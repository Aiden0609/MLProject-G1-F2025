import numpy as np
import h5py
import xarray as xr
import pickle
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import os

import torch.nn.functional as F

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

# Build test file lists for 2022
test_surf_files = []
test_atmos_files = []

for year in [2022]:
    for month in range(1, 9):
        m = f"{month:02d}"
        surf_file = data_path / f"era5_surface_{year}_{m}.nc"
        atmos_file = data_path / f"era5_atmospheric_{year}_{m}.nc"
        
        print(f"Checking: {surf_file}")
        print(f"  Exists: {surf_file.exists()}")
        print(f"Checking: {atmos_file}")
        print(f"  Exists: {atmos_file.exists()}")
        
        if surf_file.exists() and atmos_file.exists():
            test_surf_files.append(surf_file)
            test_atmos_files.append(atmos_file)
        else:
            print(f"Warning: Missing test files for {year}-{m}")

print(f"\nTest files found: {len(test_surf_files)} months")
if len(test_surf_files) == 0:
    print("ERROR: No test files found! Check your data path.")
    print(f"Looking in: {data_path}")
    exit(1)


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

# Load the best checkpoint
ckpt_dir = Path(f"/scratch/{os.environ['USER']}/checkpoints/sst_finetune")
checkpoint = torch.load(ckpt_dir / "sst_decoder_best.ckpt", map_location="cpu")
modelDecoder.load_state_dict(checkpoint, strict=False)
modelDecoder = modelDecoder.to(device)
modelDecoder.eval()

history = 2


# Physics-based loss functions
def compute_spatial_gradient_loss(pred, mask, lambda_grad=0.05):
    """
    Penalize unrealistic spatial gradients in SST.
    Computes finite differences between adjacent pixels.
    """
    # Only compute on ocean pixels
    masked_pred = pred.clone()
    masked_pred[~mask] = 0.0
    
    # Compute gradients in lat/lon directions
    grad_lat = torch.abs(masked_pred[:, :-1, :] - masked_pred[:, 1:, :])
    grad_lon = torch.abs(masked_pred[:, :, :-1] - masked_pred[:, :, 1:])
    
    # Only consider gradients where both pixels are ocean
    mask_lat = mask[:, :-1, :] & mask[:, 1:, :]
    mask_lon = mask[:, :, :-1] & mask[:, :, 1:]
    
    # Mean squared gradient (only on valid ocean boundaries)
    loss_lat = (grad_lat[mask_lat] ** 2).mean() if mask_lat.any() else torch.tensor(0.0, device=pred.device)
    loss_lon = (grad_lon[mask_lon] ** 2).mean() if mask_lon.any() else torch.tensor(0.0, device=pred.device)
    
    return lambda_grad * (loss_lat + loss_lon)


def compute_latitude_weighted_mae(pred, target, mask, lat):
    """
    Compute MAE weighted by latitude to account for grid distortion.
    Higher weight near equator, lower near poles.
    """
    # Convert latitude to radians and compute cos(lat) weights
    lat_rad = torch.deg2rad(lat)
    cos_weights = torch.cos(lat_rad).view(1, -1, 1)  # Shape: (1, n_lat, 1)
    cos_weights = cos_weights.expand_as(pred)  # Broadcast to pred shape
    
    # Compute weighted absolute error only on ocean pixels
    abs_error = torch.abs(pred - target)
    weighted_error = abs_error * cos_weights
    
    # Average over valid ocean pixels
    valid_weighted_error = weighted_error[mask]
    valid_weights = cos_weights[mask]
    
    return valid_weighted_error.sum() / valid_weights.sum()


def masked_loss(pred, target, mask, loss_fn=F.l1_loss):
    """Compute loss only on valid ocean pixels"""
    valid_pred = pred[mask]
    valid_target = target[mask]
    return loss_fn(valid_pred, valid_target)


# Testing loop
print("\nStarting testing on 2022 data...")
print(f"History window: {history}")
print(f"Sampling every 44th timestep")

test_mae = 0.0
test_physics_loss = 0.0
test_total_loss = 0.0
test_sample_count = 0

# Store per-sample metrics for analysis
sample_metrics = []

with torch.no_grad():
    for file_idx, (surf_file, atmos_file) in enumerate(zip(test_surf_files, test_atmos_files)):
        
        print(f"\nProcessing file {file_idx+1}/{len(test_surf_files)}: {surf_file.name}")
        
        # Load monthly datasets
        surf = xr.open_dataset(surf_file).isel(latitude=slice(0, 720))
        atmos = xr.open_dataset(atmos_file).isel(latitude=slice(0, 720))
        
        time_dim = 'valid_time'
        n_times = surf.sizes[time_dim]
        
        print(f"  Total timesteps in file: {n_times}")
        print(f"  Will sample: {list(range(history, n_times, 44))}")
        
        # Iterate through time steps (every 44th timestep)
        for t in range(history, n_times, 44):
            
            # Extract history slice
            surf_hist = surf.isel({time_dim: slice(t-history, t)})
            atmos_hist = atmos.isel({time_dim: slice(t-history, t)})
            
            # Extract target at time t
            target = torch.from_numpy(
                surf["sst"].isel({time_dim: t}).values
            ).float().unsqueeze(0).to(device)
            
            # Create ocean mask
            ocean_mask = ~torch.isnan(target)
            
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
            
            # Forward Aurora encoder
            with torch.inference_mode():
                _, latent = modelAurora.forward(batch)
                
            latent_decoder = latent.detach().clone()
            
            # Forward decoder
            preds = modelDecoder(latent_decoder, batch.metadata.lat, batch.metadata.lon)
            pred_sst = preds["sst"].squeeze(1)
            
            # Compute MAE (standard metric)
            mae = masked_loss(pred_sst, target, ocean_mask, F.l1_loss)
            
            # Compute latitude-weighted MAE (alternative, uncomment to use)
            # mae = compute_latitude_weighted_mae(pred_sst, target, ocean_mask, batch.metadata.lat)
            
            # Compute physics-based spatial gradient loss
            physics_loss = compute_spatial_gradient_loss(pred_sst, ocean_mask, lambda_grad=0.05)
            
            # Total loss
            total_loss = mae + physics_loss
            
            # Accumulate metrics
            test_mae += mae.item()
            test_physics_loss += physics_loss.item()
            test_total_loss += total_loss.item()
            test_sample_count += 1
            
            # Store sample metrics
            sample_metrics.append({
                'timestep': t,
                'file': surf_file.name,
                'mae': mae.item(),
                'physics_loss': physics_loss.item(),
                'total_loss': total_loss.item()
            })
            
            # Print per-sample metrics
            print(f"  Sample {test_sample_count} (t={t}): "
                  f"MAE={mae.item():.4f}, "
                  f"Physics={physics_loss.item():.6f}, "
                  f"Total={total_loss.item():.4f}")
        
        # Close monthly datasets
        surf.close()
        atmos.close()

# Compute test averages
if test_sample_count == 0:
    print("\n" + "="*80)
    print("ERROR: No samples were processed!")
    print("="*80)
    print("Possible reasons:")
    print("1. No 2024 data files found")
    print("2. Files exist but have < 46 timesteps (need at least history+44)")
    print("3. Data loading failed silently")
    exit(1)

test_mae /= test_sample_count
test_physics_loss /= test_sample_count
test_total_loss /= test_sample_count

# Print final results
print("\n" + "="*80)
print("FINAL TEST RESULTS ON 2022 DATA")
print("="*80)
print(f"Total samples evaluated: {test_sample_count}")
print(f"\nAverage MAE:           {test_mae:.4f} K")
print(f"Average Physics Loss:  {test_physics_loss:.6f}")
print(f"Average Total Loss:    {test_total_loss:.4f}")
print("="*80)

# Save results to file
results_dir = Path(f"/scratch/{os.environ['USER']}/results/sst_test")
results_dir.mkdir(parents=True, exist_ok=True)

# Save summary
with open(results_dir / "test_summary.txt", "w") as f:
    f.write("SST Test Results on 2024 Data\n")
    f.write("="*80 + "\n")
    f.write(f"Total samples: {test_sample_count}\n")
    f.write(f"Average MAE: {test_mae:.4f} K\n")
    f.write(f"Average Physics Loss: {test_physics_loss:.6f}\n")
    f.write(f"Average Total Loss: {test_total_loss:.4f}\n")

# Save per-sample metrics
import pandas as pd
df_metrics = pd.DataFrame(sample_metrics)
df_metrics.to_csv(results_dir / "per_sample_metrics.csv", index=False)

print(f"\nResults saved to: {results_dir}")
print("Done!")