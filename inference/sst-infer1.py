from pathlib import Path
import datetime

import cdsapi

# Data will be downloaded here.
download_path = Path("/home/mridul01/scratch/data")

c = cdsapi.Client()

download_path = download_path.expanduser()
#download_path.mkdir(parents=True, exist_ok=True)

# Download the static variables.
if not (download_path / "2024-06-10-static.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "land_sea_mask",
                "soil_type",
            ],
            "year": "2024",
            "month": "06",
            "day": "10",
            "time": "12:00",
            "format": "netcdf",
        },
        str(download_path / "2024-06-10-static.nc"),
    )
print("Static variables downloaded!")

# Download the surface-level variables.
if not (download_path / "2024-06-10-surface-level.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
            ],
            "year": "2024",
            "month": "06",
            "day": "10",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2024-06-10-surface-level.nc"),
    )
print("Surface-level variables downloaded!")

# Download the atmospheric variables.
if not (download_path / "2024-06-10-atmospheric.nc").exists():
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
            "year": "2024",
            "month": "06",
            "day": "10",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2024-06-10-atmospheric.nc"),
    )
print("Atmospheric variables downloaded!")


# Download the SST variable.
if not (download_path / "2024-06-10-sst.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "sea_surface_temperature",
            ],
            "year": "2024",
            "month": "06",
            "day": "10",
            "time": ["12:00"],
            "format": "netcdf",
        },
        str(download_path / "2024-06-10-sst.nc"),
    )
    print("SST data downloaded!")

# Preparing a Batch 
import torch
import xarray as xr

from aurora import Batch, Metadata

static_vars_ds = xr.open_dataset(download_path / "2024-06-10-static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2024-06-10-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2024-06-10-atmospheric.nc", engine="netcdf4")
sst_ds = xr.open_dataset(download_path / "2024-06-10-sst.nc", engine="netcdf4")

batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
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

# Set to `False` to run locally and to `True` to run on Foundry.
run_on_foundry = False

if not run_on_foundry:
    from aurora import Aurora, rollout

    model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

    model.eval()
    model = model.to("cuda")

    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=1)]


    model = model.to("cpu")



import numpy as np
import matplotlib.pyplot as plt

# --- Ocean mask from land-sea mask (0 = ocean, 1 = land)
# Confirmed shape: (721, 1440)
lsm = static_vars_ds["lsm"][0].values
ocean_mask = lsm < 0.5

pred = preds[0]
# --- Aurora predicted 2m temperature (K → °C)
# Likely shape: (720, 1440)
aurora_2t = pred.surf_vars["2t"][0, 0].numpy() - 273.15

# --- ERA5 2m temperature at 12:00 UTC (K → °C)
# Likely shape: (721, 1440) - Matches mask
era5_2t = surf_vars_ds["t2m"][2].values - 273.15

# --- ERA5 SST at 12:00 UTC (K → °C)
# Likely shape: (721, 1440) - Matches mask
sst = sst_ds["sst"][0].values - 273.15

# --- FIX MASK SHAPE LOGIC ---
if ocean_mask.shape != aurora_2t.shape:
    print(f"Aligning Aurora 2m Temp ({aurora_2t.shape}) to mask shape ({ocean_mask.shape})...")

    # To match (721, 1440), we need ONE row of padding (720 + 1 = 721).
    # Assuming the missing row is at the top (North Pole):
    aurora_2t = np.pad(aurora_2t, ((1, 0), (0, 0)), mode='edge')
    
# --- APPLY MASKING ---

# 1. Mask Aurora 2m Temp (Ocean Only)
# Now both aurora_2t and ocean_mask should be the same shape (721, 1440)
aurora_2t_ocean = np.where(ocean_mask, aurora_2t, np.nan)

# 2. Mask ERA5 2m Temp (Ocean Only)
# Both era5_2t and ocean_mask are (721, 1440)
era5_2t_ocean = np.where(ocean_mask, era5_2t, np.nan)

# 3. SST (NO MASK APPLIED)
sst_ocean = sst

# --- Plot all three 

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Aurora 2m Temp (Masked)
ax[0].imshow(aurora_2t_ocean, origin="upper", cmap="coolwarm", vmin=-2, vmax=35)
ax[0].set_title("Aurora 2m Temp (Ocean Only)")
ax[0].set_xticks([])
ax[0].set_yticks([])

# Plot 2: ERA5 2m Temp (Masked)
ax[1].imshow(era5_2t_ocean, origin="upper", cmap="coolwarm", vmin=-2, vmax=35)
ax[1].set_title("ERA5 2m Temp (Ocean Only)")
ax[1].set_xticks([])
ax[1].set_yticks([])

# Plot 3: ERA5 SST (UNMASKED)
# Note: sst_ocean now holds the unmasked sst data
ax[2].imshow(sst_ocean, origin="upper", cmap="coolwarm", vmin=-2, vmax=35)
ax[2].set_title("ERA5 SST (Full Domain)")
ax[2].set_xticks([])
ax[2].set_yticks([])

plt.tight_layout()
plt.savefig("aurora_vs_era5_comparison.png", dpi=300)
print("Comparison plot saved!")

