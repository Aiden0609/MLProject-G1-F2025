import numpy as np
import h5py
import xarray as xr
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

import cdsapi

from aurora.batch import Batch, Metadata

def downloadCds(path):
    # Data will be downloaded here.
    download_path = Path("./data/downloads") if not path else path

    c = cdsapi.Client()

    download_path = download_path.expanduser()
    download_path.mkdir(parents=True, exist_ok=True)

    # Download the static variables.
    if not (download_path / "static.nc").exists():
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "land_sea_mask",
                    "soil_type",
                ],
                "year": "2020",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
            },
            str(download_path / "static.nc"),
        )
    print("Static variables downloaded!")

    # Download the surface-level variables.
    if not (download_path / "2020-01-01-surface-level.nc").exists():
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
                "year": "2020",
                "month": "01",
                "day": "01",
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(download_path / "2020-01-01-surface-level.nc"),
        )
    print("Surface-level variables downloaded!")

    # Download the atmospheric variables.
    if not (download_path / "2020-01-01-atmospheric.nc").exists():
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
                "year": "2020",
                "month": "01",
                "day": "01",
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(download_path / "2020-01-01-atmospheric.nc"),
        )
    print("Atmospheric variables downloaded!")

def getBatch(path):
    downloadCds(path)
    download_path = path
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    static_vars_ds = static_vars_ds.sel(latitude=static_vars_ds.latitude[:720])
    surf_vars_ds = xr.open_dataset(download_path / "2020-01-01-surface-level.nc", engine="netcdf4")
    surf_vars_ds = surf_vars_ds.sel(latitude=surf_vars_ds.latitude[:720])
    atmos_vars_ds = xr.open_dataset(download_path / "2020-01-01-atmospheric.nc", engine="netcdf4")
    atmos_vars_ds = atmos_vars_ds.sel(latitude=atmos_vars_ds.latitude[:720])

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
    return batch