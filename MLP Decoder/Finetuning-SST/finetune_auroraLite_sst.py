import numpy as np
import h5py
import xarray as xr
import pickle
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torchvision

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from aurora.batch import Batch, Metadata
from aurora.model.aurora_lite import AuroraLite
from aurora.model.decoder_lite import MLPDecoderLite


# data path
data_path = Path(f"/scratch/{os.environ['USER']}/data/finetune-data-2020-2024")
data_path = data_path.expanduser()


# create dataset and dataloader
class AuroraSSTDataset(Dataset):
    def __init__(self, data_path, history=2, years=None, cache_dir=None):
        self.data_path = Path(data_path)
        self.history = history
        self.years = years if years else [2020, 2021, 2022]
        
        # Setup cache directory in scratch
        if cache_dir is None:
            self.cache_dir = Path(f"/scratch/{os.environ['USER']}/cache/aurora_sst")
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Cache directory: {self.cache_dir}")

        # Load static data once
        self.static_ds = xr.open_dataset(self.data_path / "static.nc")
        self.static_ds = self.static_ds.sel(latitude=self.static_ds.latitude[:720])

        # Build file lists based on actual naming pattern
        self.surf_files = []
        self.atmos_files = []
        
        for year in self.years:
            for month in range(1, 13):
                m = f"{month:02d}"
                surf_file = self.data_path / f"era5_surface_{year}_{m}.nc"
                atmos_file = self.data_path / f"era5_atmospheric_{year}_{m}.nc"
                
                if surf_file.exists() and atmos_file.exists():
                    self.surf_files.append(surf_file)
                    self.atmos_files.append(atmos_file)
                else:
                    print(f"Warning: Missing files for {year}-{m}")

        print(f"Found {len(self.surf_files)} complete monthly datasets")
        
        # Try to load cached index map, or build new one
        index_cache_file = self.cache_dir / f"index_map_{'_'.join(map(str, self.years))}.pkl"
        
        if index_cache_file.exists():
            print("Loading cached index map...")
            with open(index_cache_file, 'rb') as f:
                self.index_map = pickle.load(f)
        else:
            print("Building index map (this may take a moment)...")
            self.index_map = self._build_index_map()
            # Save index map for future runs
            with open(index_cache_file, 'wb') as f:
                pickle.dump(self.index_map, f)
            print(f"Index map cached to {index_cache_file}")
        
        print(f"Dataset ready: {len(self.index_map)} samples")
        
        # Keep file handles open for efficiency (closed in __del__)
        print("Opening datasets...")
        self.surf_datasets = [xr.open_dataset(f).isel(latitude=slice(0, 720)) for f in self.surf_files]
        self.atmos_datasets = [xr.open_dataset(f).isel(latitude=slice(0, 720)) for f in self.atmos_files]
        print("Datasets opened and ready!")

    def _build_index_map(self):
        """Build mapping of global index -> (month_idx, time_idx)"""
        index_map = []
        for m_idx, file in enumerate(self.surf_files):
            # Quick check of time dimension without loading full data
            with xr.open_dataset(file) as ds:
                n = ds.dims['valid_time'] 
            
            # Valid samples start after history window
            for t in range(self.history, n):
                index_map.append((m_idx, t))
        
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        m_idx, t = self.index_map[idx]

        # Access already-open datasets (no file I/O overhead!)
        surf = self.surf_datasets[m_idx]
        atmos = self.atmos_datasets[m_idx]

        # Determine time dimension name
        time_dim = 'valid_time'

        # History slice
        surf_hist = surf.isel({time_dim: slice(t-self.history, t)})
        atmos_hist = atmos.isel({time_dim: slice(t-self.history, t)})

        # Target at time t - SST from surface variables
        target = torch.from_numpy(
            surf["sst"].isel({time_dim: t}).values
        ).float()
        
        # Create ocean mask (True where SST is valid, i.e., ocean pixels)
        ocean_mask = ~torch.isnan(target)

        # Convert inputs to tensors
        batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(surf_hist["t2m"].values[:2]),
                "10u": torch.from_numpy(surf_hist["u10"].values[:2]),
                "10v": torch.from_numpy(surf_hist["v10"].values[:2]),
                "msl": torch.from_numpy(surf_hist["msl"].values[:2]),
            },
            static_vars={
                "z":   torch.from_numpy(self.static_ds["z"].values[0]),
                "slt": torch.from_numpy(self.static_ds["slt"].values[0]),
                "lsm": torch.from_numpy(self.static_ds["lsm"].values[0]),
            },
            atmos_vars={
                "t": torch.from_numpy(atmos_hist["t"].values[:2]),
                "u": torch.from_numpy(atmos_hist["u"].values[:2]),
                "v": torch.from_numpy(atmos_hist["v"].values[:2]),
                "q": torch.from_numpy(atmos_hist["q"].values[:2]),
                "z": torch.from_numpy(atmos_hist["z"].values[:2]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_hist.latitude.values),
                lon=torch.from_numpy(surf_hist.longitude.values),        
                time=(surf_hist[time_dim].values.astype("datetime64[s]").tolist()[1],),
                atmos_levels=tuple(int(level) for level in atmos_hist.pressure_level.values),
            )
        )

        return batch, target, ocean_mask

    def __del__(self):
        """Clean up file handles"""
        if hasattr(self, 'surf_datasets'):
            for ds in self.surf_datasets + self.atmos_datasets:
                try:
                    ds.close()
                except:
                    pass


# Masked loss function for SST
def masked_loss(pred, target, mask, loss_fn=F.l1_loss):
    """Compute loss only on valid ocean pixels"""
    valid_pred = pred[mask]
    valid_target = target[mask]
    return loss_fn(valid_pred, valid_target)


# Setup
device = torch.device("cuda")

# Create datasets (Strategy 1 - temporal split)
train_dataset = AuroraSSTDataset(
    data_path=data_path,
    history=2,
    years=[2020, 2021],
    cache_dir=f"/scratch/{os.environ['USER']}/cache/aurora_sst_train"
)

val_dataset = AuroraSSTDataset(
    data_path=data_path,
    history=2,
    years=[2022],
    cache_dir=f"/scratch/{os.environ['USER']}/cache/aurora_sst_val"
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=24, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=24, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
    drop_last=True
)

# Models
modelAurora = AuroraLite(
    use_lora=False,
    autocast=True,
    surf_vars=("2t", "10u", "10v", "msl"),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("z", "u", "v", "t", "q"),
)
modelAurora.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
modelAurora = modelAurora.to(device)
modelAurora.eval()

modelDecoder = MLPDecoderLite(
    surf_vars_new=["sst"],
    patch_size=modelAurora.decoder.patch_size,
    embed_dim=2 * modelAurora.encoder.embed_dim,
    hidden_dims=[512, 512, 256],
)
checkpoint = torch.load("../aurora-lite-decoder/lite-decoder.ckpt", map_location="cpu")
modelDecoder.load_state_dict(checkpoint, strict=False)
modelDecoder = modelDecoder.to(device)

# Optimizer
opt = torch.optim.AdamW(modelDecoder.parameters(), lr=3e-4)

# TensorBoard
writer = SummaryWriter(log_dir="runs/sst_finetune")
global_step = 0

# Best model tracking
best_val_loss = float('inf')
ckpt_dir = Path(f"/scratch/{os.environ['USER']}/checkpoints/sst_finetune")
ckpt_dir.mkdir(parents=True, exist_ok=True)


print("Testing DataLoader...")
for i, (batch, target, ocean_mask) in enumerate(train_loader):
    print(f"surf_vars['2t']: {batch.surf_vars['2t'].shape}")
    print(f"atmos_vars['t']: {batch.atmos_vars['t'].shape}")
    print(f"static_vars['z']: {batch.static_vars['z'].shape}")
    print(f"target: {target.shape}")
    print(f"ocean_mask: {ocean_mask.shape}, ocean pixels: {ocean_mask.sum().item()}")
    break

# Training loop
for epoch in range(10):
    
    # ========== TRAINING ==========
    modelDecoder.train()
    train_loss = 0.0
    train_rmse = 0.0
    train_mae = 0.0
    train_r2 = 0.0
    
    # Calculate halfway point
    half_epoch = len(train_loader) // 2
    
    for batch_idx, (batch, target, ocean_mask) in enumerate(train_loader):
        batch = batch.to(device)
        target = target.to(device)
        ocean_mask = ocean_mask.to(device)
        
        # Forward Aurora encoder (frozen)
        with torch.inference_mode():
            _, latent = modelAurora.forward(batch)
            latent_decoder = latent.detach()

        # Forward decoder
        opt.zero_grad()
        preds = modelDecoder(latent_decoder, batch.metadata.lat, batch.metadata.lon)
        pred_sst = preds["sst"].squeeze(1)

        # Loss (only on ocean pixels)
        loss_value = masked_loss(pred_sst, target, ocean_mask, F.l1_loss)

        # Backprop
        loss_value.backward()
        
        # ===== GRADIENT MONITORING =====
        total_norm = 0.0
        for p in modelDecoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar("train/gradient_norm_before_clip", total_norm, global_step)
        
        # Gradient clipping
        grad_norm_after = torch.nn.utils.clip_grad_norm_(modelDecoder.parameters(), max_norm=1.0)
        writer.add_scalar("train/gradient_norm_after_clip", grad_norm_after.item(), global_step)
        
        # Log individual layer gradient norms
        for name, param in modelDecoder.named_parameters():
            if param.grad is not None and 'weight' in name:
                writer.add_scalar(f"gradient_norms/{name}", param.grad.norm().item(), global_step)
        
        opt.step()

        # ===== MULTIPLE ERROR METRICS (only on ocean pixels) =====
        with torch.no_grad():
            # Extract valid ocean predictions and targets
            valid_pred = pred_sst[ocean_mask]
            valid_target = target[ocean_mask]
            
            # MAE (L1)
            mae = F.l1_loss(valid_pred, valid_target)
            
            # MSE and RMSE
            mse = F.mse_loss(valid_pred, valid_target)
            rmse = torch.sqrt(mse)
            
            # R² Score
            ss_res = torch.sum((valid_target - valid_pred) ** 2)
            ss_tot = torch.sum((valid_target - valid_target.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            # Relative error
            relative_error = torch.mean(torch.abs(valid_pred - valid_target) / (torch.abs(valid_target) + 1e-8))
            
            # Log all metrics
            writer.add_scalar("train/loss_step", loss_value.item(), global_step)
            writer.add_scalar("train/mae_step", mae.item(), global_step)
            writer.add_scalar("train/rmse_step", rmse.item(), global_step)
            writer.add_scalar("train/r2_step", r2.item(), global_step)
            writer.add_scalar("train/relative_error", relative_error.item(), global_step)
            
            # Accumulate for epoch averages
            train_loss += loss_value.item()
            train_mae += mae.item()
            train_rmse += rmse.item()
            train_r2 += r2.item()
        
        # ===== PREDICTION STATISTICS (only ocean pixels) =====
        writer.add_scalar("train/pred_mean", valid_pred.mean().item(), global_step)
        writer.add_scalar("train/pred_std", valid_pred.std().item(), global_step)
        writer.add_scalar("train/pred_min", valid_pred.min().item(), global_step)
        writer.add_scalar("train/pred_max", valid_pred.max().item(), global_step)
        
        writer.add_scalar("train/target_mean", valid_target.mean().item(), global_step)
        writer.add_scalar("train/target_std", valid_target.std().item(), global_step)
        
        # Correlation
        correlation = torch.corrcoef(torch.stack([valid_pred, valid_target]))[0, 1]
        writer.add_scalar("train/correlation", correlation.item(), global_step)
        
        # Ocean coverage
        ocean_fraction = ocean_mask.float().mean()
        writer.add_scalar("train/ocean_fraction", ocean_fraction.item(), global_step)

        # ===== PARAMETER DISTRIBUTIONS (every half epoch) =====
        if batch_idx == half_epoch or batch_idx == 0:
            step_label = f"{epoch}.5" if batch_idx == half_epoch else f"{epoch}.0"
            step_number = epoch * 2 + (1 if batch_idx == half_epoch else 0)
            
            for name, param in modelDecoder.named_parameters():
                writer.add_histogram(f"weights/{name}", param, step_number)
                writer.add_scalar(f"weight_stats/{name}_mean", param.mean().item(), step_number)
                writer.add_scalar(f"weight_stats/{name}_std", param.std().item(), step_number)
                writer.add_scalar(f"weight_stats/{name}_min", param.min().item(), step_number)
                writer.add_scalar(f"weight_stats/{name}_max", param.max().item(), step_number)
            
            print(f"  → Parameter distributions logged at epoch {step_label}")

        global_step += 1

        # ===== SPATIAL ERROR MAP & IMAGES =====
        if batch_idx == 0:
            # For visualization, replace NaN with 0 to avoid display issues
            pred_vis = pred_sst[0].clone()
            target_vis = target[0].clone()
            pred_vis[~ocean_mask[0]] = 0
            target_vis[~ocean_mask[0]] = 0
            
            writer.add_image("train/pred_sst", pred_vis.unsqueeze(0), global_step, dataformats="CHW")
            writer.add_image("train/target_sst", target_vis.unsqueeze(0), global_step, dataformats="CHW")
            
            diff_vis = (pred_sst[0] - target[0]).clone()
            diff_vis[~ocean_mask[0]] = 0
            writer.add_image("train/diff_sst", diff_vis.unsqueeze(0), global_step, dataformats="CHW")
            
            # Spatial error map (average absolute error across batch, only ocean)
            error_map = torch.abs(pred_sst - target)
            error_map[~ocean_mask] = 0
            error_map = error_map.mean(dim=0)
            writer.add_image("train/spatial_error_map", error_map.unsqueeze(0), global_step, dataformats="CHW")
            
            # Grid of multiple samples
            n_samples = min(4, pred_sst.shape[0])
            
            # Prepare grids with NaN replaced by 0
            pred_grid_data = pred_sst[:n_samples].clone()
            target_grid_data = target[:n_samples].clone()
            diff_grid_data = (pred_sst[:n_samples] - target[:n_samples]).clone()
            
            for i in range(n_samples):
                pred_grid_data[i][~ocean_mask[i]] = 0
                target_grid_data[i][~ocean_mask[i]] = 0
                diff_grid_data[i][~ocean_mask[i]] = 0
            
            pred_grid = torchvision.utils.make_grid(
                pred_grid_data.unsqueeze(1), 
                nrow=2, 
                normalize=True
            )
            target_grid = torchvision.utils.make_grid(
                target_grid_data.unsqueeze(1), 
                nrow=2, 
                normalize=True
            )
            diff_grid = torchvision.utils.make_grid(
                diff_grid_data.unsqueeze(1), 
                nrow=2, 
                normalize=True
            )
            
            writer.add_image("train/predictions_grid", pred_grid, global_step)
            writer.add_image("train/targets_grid", target_grid, global_step)
            writer.add_image("train/diff_grid", diff_grid, global_step)

    # Compute epoch averages
    train_loss /= len(train_loader)
    train_mae /= len(train_loader)
    train_rmse /= len(train_loader)
    train_r2 /= len(train_loader)
    
    # ========== VALIDATION ==========
    modelDecoder.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_rmse = 0.0
    val_r2 = 0.0
    
    # For scatter plot
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (batch, target, ocean_mask) in enumerate(val_loader):
            batch = batch.to(device)
            target = target.to(device)
            ocean_mask = ocean_mask.to(device)
            
            # Forward Aurora encoder
            with torch.inference_mode():
                _, latent = modelAurora.forward(batch)
            
            # Forward decoder
            preds = modelDecoder(latent, batch.metadata.lat, batch.metadata.lon)
            pred_sst = preds["sst"].squeeze(1)
            
            # Extract valid ocean pixels
            valid_pred = pred_sst[ocean_mask]
            valid_target = target[ocean_mask]
            
            # Multiple error metrics
            mae = F.l1_loss(valid_pred, valid_target)
            mse = F.mse_loss(valid_pred, valid_target)
            rmse = torch.sqrt(mse)
            
            ss_res = torch.sum((valid_target - valid_pred) ** 2)
            ss_tot = torch.sum((valid_target - valid_target.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            val_loss += mae.item()
            val_mae += mae.item()
            val_rmse += rmse.item()
            val_r2 += r2.item()
            
            # Collect for scatter plot
            all_preds.append(valid_pred.cpu().flatten())
            all_targets.append(valid_target.cpu().flatten())
            
            # Log first validation batch
            if batch_idx == 0:
                # For visualization, replace NaN with 0
                pred_vis = pred_sst[0].clone()
                target_vis = target[0].clone()
                pred_vis[~ocean_mask[0]] = 0
                target_vis[~ocean_mask[0]] = 0
                
                writer.add_image("val/pred_sst", pred_vis.unsqueeze(0), epoch, dataformats="CHW")
                writer.add_image("val/target_sst", target_vis.unsqueeze(0), epoch, dataformats="CHW")
                
                diff_vis = (pred_sst[0] - target[0]).clone()
                diff_vis[~ocean_mask[0]] = 0
                writer.add_image("val/diff_sst", diff_vis.unsqueeze(0), epoch, dataformats="CHW")
                
                # Spatial error map
                error_map = torch.abs(pred_sst - target)
                error_map[~ocean_mask] = 0
                error_map = error_map.mean(dim=0)
                writer.add_image("val/spatial_error_map", error_map.unsqueeze(0), epoch, dataformats="CHW")
                
                # Prediction statistics
                writer.add_scalar("val/pred_mean", valid_pred.mean().item(), epoch)
                writer.add_scalar("val/pred_std", valid_pred.std().item(), epoch)
    
    val_loss /= len(val_loader)
    val_mae /= len(val_loader)
    val_rmse /= len(val_loader)
    val_r2 /= len(val_loader)
    
    # ========== LOGGING ==========
    print(f"Epoch {epoch+1}/{10} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
    
    writer.add_scalar("train/loss_epoch", train_loss, epoch)
    writer.add_scalar("train/mae_epoch", train_mae, epoch)
    writer.add_scalar("train/rmse_epoch", train_rmse, epoch)
    writer.add_scalar("train/r2_epoch", train_r2, epoch)
    
    writer.add_scalar("val/loss_epoch", val_loss, epoch)
    writer.add_scalar("val/mae_epoch", val_mae, epoch)
    writer.add_scalar("val/rmse_epoch", val_rmse, epoch)
    writer.add_scalar("val/r2_epoch", val_r2, epoch)
    
    writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            modelDecoder.state_dict(),
            ckpt_dir / "sst_decoder_best.ckpt"
        )
        print(f"  → New best model saved! (val_loss: {val_loss:.4f})")
    
    # Save regular checkpoint
    torch.save(
        modelDecoder.state_dict(),
        ckpt_dir / f"sst_decoder_epoch{epoch}.ckpt"
    )
    
    writer.flush()

writer.close()
print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
print(f"Checkpoints saved to: {ckpt_dir}")