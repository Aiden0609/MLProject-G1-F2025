import numpy as np
import xarray as xr
import torch
from pathlib import Path
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from aurora.batch import Batch, Metadata
from aurora.model.aurora_lite import AuroraLite
from aurora.model.decoder_lite import MLPDecoderLite


# Data path
data_path = Path(f"/scratch/{os.environ['USER']}/data/finetune-data-2020-2024")
data_path = data_path.expanduser()

# Setup
device = torch.device("cuda")

# Load static data once
static_ds = xr.open_dataset(data_path / "static.nc")
static_ds = static_ds.sel(latitude=static_ds.latitude[:720])

# Build file lists
train_surf_files = []
train_atmos_files = []
train_flux_files = []
val_surf_files = []
val_atmos_files = []
val_flux_files = []

# Training: 2020-2021
for year in [2020, 2021]:
    for month in range(1, 13):
        m = f"{month:02d}"
        surf_file = data_path / f"era5_surface_{year}_{m}.nc"
        atmos_file = data_path / f"era5_atmospheric_{year}_{m}.nc"
        flux_file = data_path / f"era5_flux_{year}_{m}.nc"
        
        if surf_file.exists() and atmos_file.exists() and flux_file.exists():
            train_surf_files.append(surf_file)
            train_atmos_files.append(atmos_file)
            train_flux_files.append(flux_file)

# Validation: 2022
for year in [2022]:
    for month in range(1, 13):
        m = f"{month:02d}"
        surf_file = data_path / f"era5_surface_{year}_{m}.nc"
        atmos_file = data_path / f"era5_atmospheric_{year}_{m}.nc"
        flux_file = data_path / f"era5_flux_{year}_{m}.nc"
        
        if surf_file.exists() and atmos_file.exists() and flux_file.exists():
            val_surf_files.append(surf_file)
            val_atmos_files.append(atmos_file)
            val_flux_files.append(flux_file)

print(f"Training files: {len(train_surf_files)} months")
print(f"Validation files: {len(val_flux_files)} months")

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
    surf_vars_new=["isshf"],
    patch_size=modelAurora.decoder.patch_size,
    embed_dim=2 * modelAurora.encoder.embed_dim,
    hidden_dims=[512, 512, 256],
)
checkpoint = torch.load("../aurora-lite-decoder/lite-decoder.ckpt", map_location="cpu")
modelDecoder.load_state_dict(checkpoint, strict=False)
modelDecoder = modelDecoder.to(device)

# Optimizer
opt = torch.optim.AdamW(modelDecoder.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=2
)

# TensorBoard
log_dir = f"/scratch/{os.environ['USER']}/runs/isshf_finetune"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
global_step = 0

# Best model tracking
best_val_loss = float('inf')
ckpt_dir = Path(f"/scratch/{os.environ['USER']}/checkpoints/isshf_finetune")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Settings
accumulation_steps = 22
history = 2

# Compute global ISSHF statistics for consistent colormap normalization
print("Computing ISSHF statistics from training data...")
isshf_values = []
for flux_file in train_flux_files[:3]:  # Sample first 3 months for efficiency
    flux_sample = xr.open_dataset(flux_file).isel(latitude=slice(0, 720))
    isshf_values.append(flux_sample["isshf"].values.flatten())
    flux_sample.close()

isshf_values = np.concatenate(isshf_values)
isshf_p1 = np.percentile(isshf_values, 1)   # 1st percentile
isshf_p99 = np.percentile(isshf_values, 99)  # 99th percentile
isshf_max_abs = max(abs(isshf_p1), abs(isshf_p99))
vmin, vmax = -isshf_max_abs, isshf_max_abs  # Symmetric range for diverging colormap

print(f"ISSHF range for colormap: [{vmin:.2f}, {vmax:.2f}] W/m²")

# Create colormap function for ISSHF visualization
def create_isshf_colormap(isshf_tensor):
    """Convert ISSHF tensor to RGB image with colormap using global range"""
    isshf_np = isshf_tensor.cpu().numpy()
    
    # Normalize using global statistics
    isshf_norm = np.clip((isshf_np - vmin) / (vmax - vmin), 0, 1)
    
    # Apply colormap (RdBu_r: blue for negative, red for positive)
    cmap = cm.get_cmap('RdBu_r')
    rgb = cmap(isshf_norm)[:, :, :3]  # Drop alpha channel
    
    return torch.from_numpy(rgb).permute(2, 0, 1).float()  # CHW format

# Load validation subset once for step-wise validation
print("Loading validation subset...")
val_surf_subset = xr.open_dataset(val_surf_files[0]).isel(latitude=slice(0, 720))
val_atmos_subset = xr.open_dataset(val_atmos_files[0]).isel(latitude=slice(0, 720))
val_flux_subset = xr.open_dataset(val_flux_files[0]).isel(latitude=slice(0, 720))

print("\nStarting training...")
print(f"Accumulation steps: {accumulation_steps}")
print(f"History window: {history}")

# Training loop
for epoch in range(10):
    
    # ========== TRAINING ==========
    modelDecoder.train()
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    
    opt.zero_grad()
    sample_count = 0
    batch_loss = 0.0
    num_batches = 0
    
    # Iterate through training files
    for file_idx, (surf_file, atmos_file, flux_file) in enumerate(zip(train_surf_files, train_atmos_files, train_flux_files)):
        
        # Load monthly datasets
        surf = xr.open_dataset(surf_file).isel(latitude=slice(0, 720))
        atmos = xr.open_dataset(atmos_file).isel(latitude=slice(0, 720))
        flux = xr.open_dataset(flux_file).isel(latitude=slice(0, 720))
        
        time_dim = 'valid_time'
        n_times = surf.sizes[time_dim]
        
        # Calculate usable samples
        valid_samples = n_times - history
        usable_samples = (valid_samples // accumulation_steps) * accumulation_steps
        end_index = history + usable_samples

        for t in range(history, end_index):
            
            # Extract history slice
            surf_hist = surf.isel({time_dim: slice(t-history, t)})
            atmos_hist = atmos.isel({time_dim: slice(t-history, t)})
            
            # Extract target ISSHF
            target = torch.from_numpy(
                flux["isshf"].isel({time_dim: t}).values
            ).float().unsqueeze(0).to(device)
            
            # Create Batch object
            batch = Batch(
                surf_vars={
                    "2t": torch.from_numpy(surf_hist["t2m"].values[:2]).unsqueeze(0),
                    "10u": torch.from_numpy(surf_hist["u10"].values[:2]).unsqueeze(0),
                    "10v": torch.from_numpy(surf_hist["v10"].values[:2]).unsqueeze(0),
                    "msl": torch.from_numpy(surf_hist["msl"].values[:2]).unsqueeze(0),
                },
                static_vars={
                    "z":   torch.from_numpy(static_ds["z"].values[0]),
                    "slt": torch.from_numpy(static_ds["slt"].values[0]),
                    "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
                },
                atmos_vars={
                    "t": torch.from_numpy(atmos_hist["t"].values[:2]).unsqueeze(0),
                    "u": torch.from_numpy(atmos_hist["u"].values[:2]).unsqueeze(0),
                    "v": torch.from_numpy(atmos_hist["v"].values[:2]).unsqueeze(0),
                    "q": torch.from_numpy(atmos_hist["q"].values[:2]).unsqueeze(0),
                    "z": torch.from_numpy(atmos_hist["z"].values[:2]).unsqueeze(0),
                },
                metadata=Metadata(
                    lat=torch.from_numpy(surf_hist.latitude.values),
                    lon=torch.from_numpy(surf_hist.longitude.values),        
                    time=(surf_hist[time_dim].values.astype("datetime64[s]").tolist()[1],),
                    atmos_levels=tuple(int(level) for level in atmos_hist.pressure_level.values),
                )
            )
            
            # Forward pass
            with torch.inference_mode():
                _, latent = modelAurora.forward(batch)
                
            latent_decoder = latent.detach().clone()
            preds = modelDecoder(latent_decoder, batch.metadata.lat, batch.metadata.lon)
            pred_isshf = preds["isshf"].squeeze(1)
            
            # Compute loss
            loss_value = F.l1_loss(pred_isshf, target) / accumulation_steps
            loss_value.backward()
            
            batch_loss += loss_value.item() * accumulation_steps
            sample_count += 1
            
            # Update weights after accumulation_steps
            if sample_count % accumulation_steps == 0:
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(modelDecoder.parameters(), max_norm=1.0)
                
                # Optimizer step
                opt.step()
                opt.zero_grad()
                
                # Average batch loss
                train_loss_step = batch_loss / accumulation_steps
                epoch_train_loss += train_loss_step
                num_batches += 1
                
                # Log training loss
                writer.add_scalar("loss/train_step", train_loss_step, global_step)
                
                # ========== VALIDATION AT EACH STEP ==========
                modelDecoder.eval()
                val_loss_step = 0.0
                val_count = 0
                
                with torch.no_grad():
                    # Validate on subset (22 samples)
                    for t_val in range(history, min(history + accumulation_steps, val_surf_subset.sizes['valid_time'])):
                        
                        surf_hist_val = val_surf_subset.isel(valid_time=slice(t_val-history, t_val))
                        atmos_hist_val = val_atmos_subset.isel(valid_time=slice(t_val-history, t_val))
                        
                        target_val = torch.from_numpy(
                            val_flux_subset["isshf"].isel(valid_time=t_val).values
                        ).float().unsqueeze(0).to(device)
                        
                        batch_val = Batch(
                            surf_vars={
                                "2t": torch.from_numpy(surf_hist_val["t2m"].values[:2]).unsqueeze(0),
                                "10u": torch.from_numpy(surf_hist_val["u10"].values[:2]).unsqueeze(0),
                                "10v": torch.from_numpy(surf_hist_val["v10"].values[:2]).unsqueeze(0),
                                "msl": torch.from_numpy(surf_hist_val["msl"].values[:2]).unsqueeze(0),
                            },
                            static_vars={
                                "z":   torch.from_numpy(static_ds["z"].values[0]),
                                "slt": torch.from_numpy(static_ds["slt"].values[0]),
                                "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
                            },
                            atmos_vars={
                                "t": torch.from_numpy(atmos_hist_val["t"].values[:2]).unsqueeze(0),
                                "u": torch.from_numpy(atmos_hist_val["u"].values[:2]).unsqueeze(0),
                                "v": torch.from_numpy(atmos_hist_val["v"].values[:2]).unsqueeze(0),
                                "q": torch.from_numpy(atmos_hist_val["q"].values[:2]).unsqueeze(0),
                                "z": torch.from_numpy(atmos_hist_val["z"].values[:2]).unsqueeze(0),
                            },
                            metadata=Metadata(
                                lat=torch.from_numpy(surf_hist_val.latitude.values),
                                lon=torch.from_numpy(surf_hist_val.longitude.values),        
                                time=(surf_hist_val.valid_time.values.astype("datetime64[s]").tolist()[1],),
                                atmos_levels=tuple(int(level) for level in atmos_hist_val.pressure_level.values),
                            )
                        )
                        
                        with torch.inference_mode():
                            _, latent_val = modelAurora.forward(batch_val)
                            
                        latent_decoder_val = latent_val.detach().clone()
                        preds_val = modelDecoder(latent_decoder_val, batch_val.metadata.lat, batch_val.metadata.lon)
                        pred_isshf_val = preds_val["isshf"].squeeze(1)
                        
                        mae_val = F.l1_loss(pred_isshf_val, target_val)
                        val_loss_step += mae_val.item()
                        val_count += 1
                
                val_loss_step /= val_count
                epoch_val_loss += val_loss_step
                
                # Log validation loss
                writer.add_scalar("loss/val_step", val_loss_step, global_step)
                
                # Visualizations every 100 steps with colormap
                if global_step % 100 == 0:
                    pred_colored = create_isshf_colormap(pred_isshf[0])
                    target_colored = create_isshf_colormap(target[0])
                    diff_colored = create_isshf_colormap(pred_isshf[0] - target[0])
                    
                    writer.add_image("isshf/prediction", pred_colored, global_step)
                    writer.add_image("isshf/target", target_colored, global_step)
                    writer.add_image("isshf/difference", diff_colored, global_step)
                    
                    print(f"Epoch {epoch+1} | Step {global_step} | Train: {train_loss_step:.4f} | Val: {val_loss_step:.4f}")
                
                modelDecoder.train()
                batch_loss = 0.0
                global_step += 1
        
        # Close monthly datasets
        surf.close()
        atmos.close()
        flux.close()
    
    # ========== EPOCH SUMMARY ==========
    epoch_train_loss /= num_batches
    epoch_val_loss /= num_batches
    
    scheduler.step(epoch_val_loss)
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch+1}/10 COMPLETE")
    print(f"{'='*60}")
    print(f"Average Train Loss: {epoch_train_loss:.4f}")
    print(f"Average Val Loss:   {epoch_val_loss:.4f}")
    print(f"Learning Rate:      {opt.param_groups[0]['lr']:.2e}")
    print(f"{'='*60}\n")
    
    writer.add_scalar("loss/train_epoch", epoch_train_loss, epoch)
    writer.add_scalar("loss/val_epoch", epoch_val_loss, epoch)
    writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
    
    # Save best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(
            modelDecoder.state_dict(),
            ckpt_dir / "isshf_decoder_best.ckpt"
        )
        print(f"✓ New best model saved! (val_loss: {epoch_val_loss:.4f})\n")
    
    # Save epoch checkpoint
    torch.save(
        modelDecoder.state_dict(),
        ckpt_dir / f"isshf_decoder_epoch{epoch}.ckpt"
    )
    
    writer.flush()

# Cleanup
val_surf_subset.close()
val_atmos_subset.close()
val_flux_subset.close()
writer.close()

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Checkpoints saved to: {ckpt_dir}")
print(f"{'='*60}")