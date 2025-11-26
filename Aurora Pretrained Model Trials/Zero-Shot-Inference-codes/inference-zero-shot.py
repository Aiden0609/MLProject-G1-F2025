from pathlib import Path

import cdsapi

# Data will be downloaded here.
download_path = Path("~/scratch/data/inference-zero-shot")

c = cdsapi.Client()

download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)

# Download the static variables.
if not (download_path / "2025-01-01-static.nc").exists():
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
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        str(download_path / "2025-01-01-static.nc"),
    )

# Download the surface-level variables.
if not (download_path / "2025-01-01-surface-level.nc").exists():
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
            "month": "01",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2025-01-01-surface-level.nc"),
    )

# Download the atmospheric variables.
if not (download_path / "2025-01-01-atmospheric.nc").exists():
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
            "month": "01",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2025-01-01-atmospheric.nc"),
    )

import torch
import xarray as xr

from aurora import Batch, Metadata

static_vars_ds = xr.open_dataset(download_path / "2025-01-01-static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2025-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2025-01-01-atmospheric.nc", engine="netcdf4")

sst = surf_vars_ds["sst"]  # Kelvin

# Compute median while ignoring NaNs
sst_median = float(sst.median().values)

# Fill missing values with median
sst_filled = sst.fillna(sst_median)


from aurora.normalisation import locations, scales

locations["sst"] = float(sst_filled.mean().values)  # e.g.,  mean of your SST data
scales["sst"] = float(sst_filled.std().values)      # e.g., standard deviation of SST

# Batch data
batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
        "sst": torch.from_numpy(sst_filled.values[:2][None]),  # gap-filled SST
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
from aurora import Aurora, rollout

model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

model.eval()
model = model.to("cuda")

with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]

model = model.to("cpu")


import matplotlib.pyplot as plt
import numpy as np

# Ocean mask: True for ocean, False for land
ocean_mask = static_vars_ds["lsm"].values == 0

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    # Predicted SST in Celsius
    sst_pred_c = pred.surf_vars["sst"][0, 0].numpy() - 273.15
    sst_pred_c_masked = np.where(ocean_mask, sst_pred_c, np.nan)
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
plt.savefig("sst_comparison.png", dpi=300)
plt.close(fig)  # Close the figure to free memory




import numpy as np

# latitudes in degrees from the dataset
lat = surf_vars_ds.latitude.values  # shape: [H]

# Compute cos(lat) in radians
weights = np.cos(np.deg2rad(lat))

# Normalize to unit mean
weights /= weights.mean()  # shape: [H]


def weighted_rmse(pred, true, weights):
    """
    pred, true: [time, lat, lon], may contain NaNs
    weights: [lat]
    """
    w = weights[:, None]  # [lat,1]

    # Mask invalid points
    mask = ~np.isnan(pred)
    
    se = (pred - true)**2
    se[~mask] = 0  # ignore NaNs
    
    # Apply weights
    weighted_se = se * w[None, :, :]
    
    # Normalize by sum of weights over valid points
    rmse = np.sqrt(np.sum(weighted_se) / np.sum(w[None, :, :] * mask))
    
    return rmse



sst_pred = np.stack([p.surf_vars["sst"][0, 0].numpy() for p in preds])  # [time, lat, lon]
# Mask predicted SST to ocean only (land = NaN)
sst_pred = np.where(ocean_mask, sst_pred, np.nan)
sst_true = sst.values[2:2+sst_pred.shape[0]]  # align with prediction time steps

rmse = weighted_rmse(sst_pred, sst_true, weights)
print("Weighted RMSE:", rmse)


def weighted_acc(pred, true, climatology, weights):
    """
    pred, true, climatology: [time, lat, lon], may contain NaNs
    weights: [lat]
    """
    w = weights[:, None]  # [lat,1]
    acc_list = []

    for t in range(pred.shape[0]):
        pred_anom = pred[t] - climatology[t]
        true_anom = true[t] - climatology[t]

        # Ignore NaNs
        mask = ~np.isnan(pred_anom)  # NaNs in pred indicate land points
        pred_anom = np.where(mask, pred_anom, 0)
        true_anom = np.where(mask, true_anom, 0)

        numerator = np.sum(w * pred_anom * true_anom)
        denominator = np.sqrt(np.sum(w * pred_anom**2) * np.sum(w * true_anom**2))
        acc_list.append(numerator / denominator)

    return np.mean(acc_list)



# Daily climatology: same shape as sst_pred
# For simplicity, mean over available ERA5 time steps
sst_climatology = np.nanmean(sst.values, axis=0)[None, :, :]
sst_climatology = np.repeat(sst_climatology, sst_pred.shape[0], axis=0)

acc = weighted_acc(sst_pred, sst_true, sst_climatology, weights)
print("Weighted ACC:", acc)

with open("sst_metrics.txt", "w") as f:
    f.write(f"Weighted RMSE: {rmse:.4f}\n")
    f.write(f"Weighted ACC: {acc:.4f}\n")
