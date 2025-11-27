from pathlib import Path
from typing import Literal
from aurora import Batch, Metadata
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class SSTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        targets: list[str],
        history: int = 2,
        static_variables: list[str] = None,
        surface_variables: list[str] = None,
        atmosphere_variables: list[str] = None,
        chunks: Literal["auto"] | None = "auto",
    ) -> None:
        super().__init__()
        if static_variables is None:
            static_variables = ["z", "slt", "lsm"]
        if surface_variables is None:
            surface_variables = ["t2m", "u10", "v10", "msl"]
        if atmosphere_variables is None:
            atmosphere_variables = ["z", "u", "v", "t", "q"]

        for target in targets:
            if target not in surface_variables and target not in atmosphere_variables:
                raise ValueError(
                    "Target must be included in surface or atmosphere variables"
                )

        self.path = path
        self.targets = targets
        self.history = history

        # ----- Static ----
        self.static_ds = xr.open_dataset(
            self.path / "static.nc", engine="netcdf4", chunks=chunks
        )
        self.static_ds = self.static_ds.sel(latitude=self.static_ds["latitude"][:720])
        self.static_variables = static_variables
        self.static_vars = {
            var_name: torch.from_numpy(self.static_ds[var_name].values[0])
            for var_name in static_variables
        }

        # ----- Surface ----
        self.surf_ds = xr.open_dataset(
            self.path / "surface-level.nc", engine="netcdf4", chunks=chunks
        )
        self.surf_ds = self.surf_ds.sel(latitude=self.surf_ds["latitude"][:720])
        self.surf_variables = surface_variables

        self.lat = torch.from_numpy(self.surf_ds["latitude"].values)
        self.lon = torch.from_numpy(self.surf_ds["longitude"].values)

        # ----- Atmosphere ----
        # TODO not use remapped
        self.atmos_ds = xr.open_dataset(
            self.path / "atmospheric.nc", engine="netcdf4", chunks=chunks
        )
        self.atmos_ds = self.atmos_ds.sel(latitude=self.atmos_ds["latitude"][:720])
        self.atmos_levels = tuple(
            int(level) for level in self.atmos_ds["pressure_level"].values
        )
        self.atmos_variables = atmosphere_variables

        # ----- Other ----
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element. Select element 1, corresponding to time
        # 06:00.
        self.times = self.surf_ds["valid_time"].values.astype("datetime64[s]").tolist()
        # TODO why + 1
        self.len = len(self.times) - history + 1

        self.cur = None
        self.next = None  # target
        # _, self.next = self[-1]

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index: int):
        """
        :param index: Must be
        :type index: int
        """
        if index == 0:
            self.cur = None
            _, self.next = self[-1]
        if (index not in (-1, 0)) and index != self.last_index + 1:
            raise ValueError("Must access data in order")
        else:
            self.last_index = index

        index += 1
        self.cur = self.next
        start = self.times[index]
        end = self.times[index + self.history]
        target_time = self.times[index + self.history + 1]

        time_slice = slice(start, end)

        sliced_surf = self.surf_ds.sel(valid_time=time_slice)
        sliced_atmos = self.atmos_ds.sel(valid_time=time_slice)

        surf_vars = {
            var_name: torch.from_numpy(sliced_surf[var_name].values)
            for var_name in self.surf_variables
        }
        atmos_vars = {
            var_name: torch.from_numpy(sliced_atmos[var_name].values)
            for var_name in self.atmos_variables
        }
        # we only train based on target vars
        for var_name, val in surf_vars.items():
            val.requires_grad = var_name in self.targets
        for var_name, val in atmos_vars.items():
            val.requires_grad = var_name in self.targets

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
    path = Path("../data_exploration/data")
    dataset = SSTDataset(
        path,
        ["sst"],
        surface_variables=["t2m", "u10", "v10", "msl", "sst", "siconc"],
        history=0,
    )
    input, target = dataset[0]
    print(input.surf_vars["sst"][-1].shape)
