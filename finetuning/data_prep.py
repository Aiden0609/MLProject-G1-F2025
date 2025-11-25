import numpy as np
import xarray as xr
import torch
from pathlib import Path

import cdsapi
from torch.utils.data import Dataset

from aurora.batch import Batch, Metadata


def _time_coord(ds):
    """Return the name of the time coordinate in a dataset."""
    return "valid_time" if "valid_time" in ds else "time"

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

    # Instantaneous sensible heat flux.
    if not (download_path / "2020-01-01-ishf.nc").exists():
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["instantaneous_surface_sensible_heat_flux"],
                "year": "2020",
                "month": "01",
                "day": "01",
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(download_path / "2020-01-01-ishf.nc")
        )
    print("Instanteneous sensible heat flux downloaded!")


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


class AuroraLiteFluxDataset(Dataset):
    """Dataset that returns Aurora input batches and sensible heat flux targets."""

    def __init__(self, path: Path, history: int = 2, target_var: str = "ishf") -> None:
        super().__init__()
        downloadCds(path)
        self.path = path
        self.history = history
        self.target_var = target_var

        self.static_ds = xr.open_dataset(self.path / "static.nc", engine="netcdf4")
        self.static_ds = self.static_ds.sel(latitude=self.static_ds.latitude[:720])

        self.surf_ds = xr.open_dataset(self.path / "2020-01-01-surface-level.nc", engine="netcdf4")
        self.surf_ds = self.surf_ds.sel(latitude=self.surf_ds.latitude[:720])

        self.atmos_ds = xr.open_dataset(
            self.path / "2020-01-01-atmospheric.nc", engine="netcdf4"
        )
        self.atmos_ds = self.atmos_ds.sel(latitude=self.atmos_ds.latitude[:720])

        self.ishf_ds = xr.open_dataset(self.path / "2020-01-01-ishf.nc", engine="netcdf4")
        self.ishf_ds = self.ishf_ds.sel(latitude=self.ishf_ds.latitude[:720])

        self.surf_times = self.surf_ds.valid_time.values
        self.ishf_times = self.ishf_ds[_time_coord(self.ishf_ds)].values
        self.indices = list(range(len(self.surf_times) - history + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end = start + self.history
        target_time = self.surf_times[end - 1]
        ishf_time_idx = int(np.where(self.ishf_times == target_time)[0][0])

        surf_slice = slice(start, end)

        surf_vars = {
            "2t": torch.from_numpy(self.surf_ds["t2m"].values[surf_slice]),
            "10u": torch.from_numpy(self.surf_ds["u10"].values[surf_slice]),
            "10v": torch.from_numpy(self.surf_ds["v10"].values[surf_slice]),
            "msl": torch.from_numpy(self.surf_ds["msl"].values[surf_slice]),
        }

        static_vars = {
            "z": torch.from_numpy(self.static_ds["z"].values[0]),
            "slt": torch.from_numpy(self.static_ds["slt"].values[0]),
            "lsm": torch.from_numpy(self.static_ds["lsm"].values[0]),
        }

        atmos_vars = {
            "t": torch.from_numpy(self.atmos_ds["t"].values[surf_slice]),
            "u": torch.from_numpy(self.atmos_ds["u"].values[surf_slice]),
            "v": torch.from_numpy(self.atmos_ds["v"].values[surf_slice]),
            "q": torch.from_numpy(self.atmos_ds["q"].values[surf_slice]),
            "z": torch.from_numpy(self.atmos_ds["z"].values[surf_slice]),
        }

        metadata = Metadata(
            lat=torch.from_numpy(self.surf_ds.latitude.values),
            lon=torch.from_numpy(self.surf_ds.longitude.values),
            time=(np.array(target_time, dtype="datetime64[s]").item(),),
            atmos_levels=tuple(int(level) for level in self.atmos_ds.pressure_level.values),
        )

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )

        target = torch.from_numpy(self.ishf_ds[self.target_var].values[ishf_time_idx]).float()

        return batch, target.unsqueeze(0)  # (1, H, W)


if __name__ == "__main__":
    downloadCds(Path("./data/downloads"))
