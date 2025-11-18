from pathlib import Path

# Data will be downloaded here.
download_path = Path("/home/mridul01/scratch/data/aurora-small/inference-metric")


import torch
import xarray as xr

from aurora import Batch, Metadata

static_vars_ds = xr.open_dataset(download_path / "2025-02-01-static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2025-02-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2025-02-01-atmospheric.nc", engine="netcdf4")

sst = surf_vars_ds["sst"]  # Kelvin

# Compute median while ignoring NaNs
sst_median = float(sst.median().values)

# Fill missing values with median
sst_filled = sst.fillna(sst_median)




# Batch data
batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(sst_filled.values[:2][None]),      # gap-filled SST
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None])         
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element. Select element 1, corresponding to time
        # 06:00.
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

# Inference 
from aurora import AuroraSmallPretrained, rollout

model = AuroraSmallPretrained()
model.load_checkpoint()

model.eval()
model = model.to("cuda")



with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]

model = model.to("cpu")


import matplotlib.pyplot as plt
import numpy as np

# Ocean mask: True for ocean, False for land and land/water mix
ocean_mask = static_vars_ds["lsm"].values == 0
ocean_mask_sliced = ocean_mask[0, :-1, :] # Shape becomes (720, 1440)

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    # Predicted SST in Celsius
    sst_pred_c = pred.surf_vars["2t"][0, 0].numpy() - 273.15
    sst_pred_c_masked = np.where(ocean_mask_sliced, sst_pred_c, np.nan)
    ax[i, 0].imshow(sst_pred_c_masked, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Predicted SST")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    # Original SST in Celsius
    sst_orig_c = sst.values[2 + i] - 273.15
    #sst_orig_c_masked = np.where(ocean_mask, sst_orig_c, np.nan)
    ax[i, 1].imshow(sst_orig_c, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("SST") 
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()

# Save figure as PNG
plt.savefig("sst_comparison_small.png", dpi=300)
plt.close(fig)  # Close the figure to free memory




import numpy as np

def compute_weights(lat):
    """
    Compute cosine latitude weights for global averaging.
    Normalize to mean 1.
    
    Parameters
    ----------
    lat : [H] array of latitudes in degrees
    
    Returns
    -------
    weights : [H] array
    """
    w = np.cos(np.deg2rad(lat))
    w /= w.mean()
    return w

def masked_weighted_rmse(pred, true, mask, weights):
    """
    Compute weighted RMSE over time, latitude, and longitude, ignoring land points.

    Parameters
    ----------
    pred, true : [time, lat, lon] arrays
    mask : [lat, lon] boolean array, True = ocean
    weights : [lat] array of cosine-latitude weights
    
    Returns
    -------
    rmse : float
    """
    w = weights[:, None]  # [lat, 1] for broadcasting
    valid_mask = mask[None, :, :] & ~np.isnan(pred) & ~np.isnan(true)
    
    se = (pred - true)**2
    se[~valid_mask] = 0
    
    rmse = np.sqrt(np.sum(se * w[None, :, :]) / np.sum(w[None, :, :] * valid_mask))
    return rmse

def masked_weighted_acc(pred, true, climatology, mask, weights):
    """
    Compute weighted Anomaly Correlation Coefficient (ACC) over time.
    ACC = correlation of anomalies (pred - climatology vs true - climatology)
    
    Parameters
    ----------
    pred, true, climatology : [time, lat, lon] arrays
    mask : [lat, lon] boolean array, True = ocean
    weights : [lat] array
    
    Returns
    -------
    acc : float (mean over time)
    """
    w = weights[:, None]
    acc_list = []

    for t in range(pred.shape[0]):
        pred_anom = pred[t] - climatology[t]
        true_anom = true[t] - climatology[t]
        
        valid_mask = mask & ~np.isnan(pred_anom) & ~np.isnan(true_anom)
        
        pred_anom_masked = np.where(valid_mask, pred_anom, 0)
        true_anom_masked = np.where(valid_mask, true_anom, 0)
        
        numerator = np.sum(w * pred_anom_masked * true_anom_masked)
        denominator = np.sqrt(np.sum(w * pred_anom_masked**2) * np.sum(w * true_anom_masked**2))
        
        if denominator == 0:
            acc_list.append(np.nan)
        else:
            acc_list.append(numerator / denominator)
    
    return np.nanmean(acc_list)

# -----------------------------
lat = preds[0].metadata.lat.numpy()
weights = compute_weights(lat)


# Convert Aurora predictions → numpy array [time, lat, lon]
pred_sst = np.stack([
    pred.surf_vars["2t"][0, 0].cpu().numpy()
    for pred in preds
])

# Aurora latitudes from metadata
aurora_lat = preds[0].metadata.lat.cpu().numpy()

# Select matching time steps using xarray (not numpy)
true_sst_interp = (
    sst_filled
    .isel(valid_time=slice(2, 2 + len(preds)))  # keep as xarray
    .interp(latitude=aurora_lat)                # interpolate lat
)


# Convert to numpy array for metric computation
true_sst = true_sst_interp.values  # shape now matches pred_sst


# Build simple climatology from first 2 input frames (ERA5 times 0 and 1)
clim = np.nanmean(sst_filled.values[:2], axis=0)   # [lat, lon]


# Interpolate to Aurora latitude grid
clim_interp = xr.DataArray(clim, coords={"latitude": surf_vars_ds.latitude, "longitude": surf_vars_ds.longitude})
clim_interp = clim_interp.interp(latitude=aurora_lat)

# Repeat along time dimension
sst_climatology = np.repeat(clim_interp.values[None], len(preds), axis=0)  # [time, lat, lon]

# -----------------------------
# Compute Metrics
# -----------------------------
rmse = masked_weighted_rmse(pred_sst, true_sst, ocean_mask_sliced, weights)
acc = masked_weighted_acc(pred_sst, true_sst, sst_climatology, ocean_mask_sliced, weights)

print("Weighted RMSE:", rmse)
print("Weighted ACC:", acc)

# -----------------------------
# Save to file
# -----------------------------
with open("sst_metrics.txt", "w") as f:
    f.write(f"Weighted RMSE for SST: {rmse:.4f}\n")
    f.write(f"Weighted ACC for SST: {acc:.4f}\n")

