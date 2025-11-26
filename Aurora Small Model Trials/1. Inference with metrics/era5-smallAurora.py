from pathlib import Path

import cdsapi

# Data will be downloaded here.
download_path = Path("/home/mridul01/scratch/data/aurora-small/inference-metric")

c = cdsapi.Client()

download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)

# Download the static variables.
if not (download_path / "2025-02-01-static.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "land_sea_mask",
                "soil_type",
            ],
            "year": "2025",
            "month": "02",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        str(download_path / "2025-02-01-static.nc"),
    )

# Download the surface-level variables.
if not (download_path / "2025-02-01-surface-level.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "sea_surface_temperature"
            ],
            "year": "2025",
            "month": "02",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2025-02-01-surface-level.nc"),
    )

# Download the atmospheric variables.
if not (download_path / "2025-02-01-atmospheric.nc").exists():
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
                "geopotential",
            ],
            "pressure_level": [
                "50",
                "100",
                "150",
                "200",
                "250",
                "300",
                "400",
                "500",
                "600",
                "700",
                "850",
                "925",
                "1000",
            ],
            "year": "2025",
            "month": "02",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2025-02-01-atmospheric.nc"),
    )

import torch
import xarray as xr

from aurora import Batch, Metadata

static_vars_ds = xr.open_dataset(download_path / "2025-02-01-static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2025-02-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2025-02-01-atmospheric.nc", engine="netcdf4")

#sst = surf_vars_ds["sst"]  # Kelvin

# Compute median while ignoring NaNs
#sst_median = float(sst.median().values)

# Fill missing values with median
#sst_filled = sst.fillna(sst_median)

# Batch data
batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),      
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

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Prediction")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    ax[i, 1].imshow(surf_vars_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("ERA5")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()
plt.savefig("aurora_small_t2m.png")



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

def weighted_rmse(pred, true, weights):
    """
    Compute weighted RMSE over time, latitude, and longitude, ignoring land points.

    Parameters
    ----------
    pred, true : [time, lat, lon] arrays
    weights : [lat] array of cosine-latitude weights
    
    Returns
    -------
    rmse : float
    """
    w = weights[:, None]  # [lat, 1] for broadcasting
        
    se = (pred - true)**2
    
    rmse = np.sqrt(np.sum(se * w[None, :, :]) / np.sum(w[None, :, :]))
    return rmse

def weighted_acc(pred, true, climatology, weights):
    """
    Compute weighted Anomaly Correlation Coefficient (ACC) over time.
    ACC = correlation of anomalies (pred - climatology vs true - climatology)
    
    Parameters
    ----------
    pred, true, climatology : [time, lat, lon] arrays
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
        
        
        
        numerator = np.sum(w * pred_anom * true_anom)
        denominator = np.sqrt(np.sum(w * pred_anom**2) * np.sum(w * true_anom**2))
        
        if denominator == 0:
            acc_list.append(np.nan)
        else:
            acc_list.append(numerator / denominator)
    
    return np.mean(acc_list)

# -----------------------------
lat = preds[0].metadata.lat.cpu().numpy()
weights = compute_weights(lat)


# Convert Aurora predictions → numpy array [time, lat, lon]
pred_t2m = np.stack([
    pred.surf_vars["2t"][0, 0].cpu().numpy()
    for pred in preds
])

# Extract true ERA5 SST for matching predicted steps
# pred[0] corresponds to ERA5 index 2, pred[1] to index 3
# Aurora latitudes from metadata
aurora_lat = preds[0].metadata.lat.cpu().numpy()

# Interpolate ERA5 truth data to Aurora’s lat grid
true_t2m_interp = (
    surf_vars_ds["t2m"]
    .isel(valid_time=slice(2, 2 + len(preds)))   # match prediction steps
    .interp(latitude=aurora_lat)           # align latitudes
)

# Convert to numpy array for metric computation
true_t2m = true_t2m_interp.values  # shape now matches pred_t2m


# Build simple climatology from first 2 input frames (ERA5 times 0 and 1)
# Build climatology from first 2 input frames
clim = np.nanmean(surf_vars_ds["t2m"].isel(valid_time=slice(0, 2)), axis=0)

# Interpolate to Aurora latitude grid
clim_interp = xr.DataArray(clim, coords={"latitude": surf_vars_ds.latitude, "longitude": surf_vars_ds.longitude})
clim_interp = clim_interp.interp(latitude=aurora_lat)

# Repeat along time dimension
t2m_climatology = np.repeat(clim_interp.values[None], len(preds), axis=0)


# -----------------------------
# Compute Metrics
# -----------------------------
rmse = weighted_rmse(pred_t2m, true_t2m, weights)
acc = weighted_acc(pred_t2m, true_t2m, t2m_climatology, weights)

print("Weighted RMSE:", rmse)
print("Weighted ACC:", acc)

# -----------------------------
# Save to file
# -----------------------------
with open("t2m_metrics.txt", "w") as f:
    f.write(f"Weighted RMSE: {rmse:.4f}\n")
    f.write(f"Weighted ACC: {acc:.4f}\n")

era5_lat = surf_vars_ds.latitude.values

aurora_lat = preds[0].metadata.lat.cpu().numpy()

missing = np.setdiff1d(era5_lat, aurora_lat)
print("Missing latitudes:", missing)
print("Number missing:", missing.size)


if missing.size > 0:
    missing_lat = float(missing[0])
    idx = np.where(era5_lat == missing_lat)[0][0]
    print("Missing latitude:", missing_lat)
    print("ERA5 index:", idx)
else:
    print("No missing latitudes.")
