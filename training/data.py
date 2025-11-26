from pathlib import Path
from typing import Literal
from aurora import Batch, Metadata
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class SSTDataset(Dataset):
    def __init__(
        self, path: Path, history: int = 2, chunks: Literal["auto"] | None = "auto"
    ) -> None:
        super().__init__()
        self.path = path
        self.history = history
        self.static_ds = xr.open_dataset(
            self.path / "static.nc", engine="netcdf4", chunks=chunks
        )
        self.static_ds = self.static_ds.sel(latitude=self.static_ds["latitude"][:720])
        self.static_vars = {
            "z": torch.from_numpy(self.static_ds["z"].values[0]),
            "slt": torch.from_numpy(self.static_ds["slt"].values[0]),
            "lsm": torch.from_numpy(self.static_ds["lsm"].values[0]),
        }

        self.surf_ds = xr.open_dataset(
            self.path / "surface-level.nc", engine="netcdf4", chunks=chunks
        )
        self.surf_ds = self.surf_ds.sel(latitude=self.surf_ds["latitude"][:720])
        self.lat = (torch.from_numpy(self.surf_ds["latitude"].values),)
        self.lon = (torch.from_numpy(self.surf_ds["longitude"].values),)

        self.atmos_ds = xr.open_dataset(
            self.path / "atmospheric.nc", engine="netcdf4", chunks=chunks
        )
        self.atmos_ds = self.atmos_ds.sel(latitude=self.atmos_ds["latitude"][:720])
        self.atmos_levels = tuple(
            int(level) for level in self.atmos_ds["pressure_level"].values
        )

        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element. Select element 1, corresponding to time
        # 06:00.
        self.times = self.surf_ds["valid_time"].values.astype("datetime64[s]").tolist()
        # TODO why + 1
        self.len = len(self.times) - history + 1

        self.cur = None
        self.next = None  # target
        _, self.next = self[-1]

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index: int):
        index += 1
        self.cur = self.next
        start = self.times[index]
        end = self.times[index + self.history]
        target_time = self.times[index + self.history + 1]

        time_slice = slice(start, end)

        sliced_surf = self.surf_ds.sel(valid_time=time_slice)
        sliced_atmos = self.atmos_ds.sel(valid_time=time_slice)

        surf_vars = {
            key: torch.from_numpy(val.values)
            for key, val in sliced_surf.variables.items()
        }
        atmos_vars = {
            key: torch.from_numpy(val.values)
            for key, val in sliced_atmos.variables.items()
        }
        # TODO does this need to be copied?
        metadata = Metadata(
            lat=self.lat,
            lon=self.lon,
            time=(np.array(target_time, dtype="datetime64[s]").item(),),
            atmos_levels=self.atmos_levels,
        )
        batch = Batch(
            surf_vars=surf_vars,
            static_vars=self.static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )
        self.next = batch
        return self.cur, self.next


if __name__ == "__main__":
    path = Path("../data_exploration/reanalysis-era5-single-levels")
