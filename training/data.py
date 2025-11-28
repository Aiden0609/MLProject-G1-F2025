from pathlib import Path
from typing import Literal
from aurora import Batch, Metadata
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from torch.utils.data import DataLoader


class SSTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        targets: list[str],
        history: int = 3,
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
        # self.surf_ds = xr.open_dataset(
        #     self.path / "surface-level.nc", engine="netcdf4", chunks=chunks
        # )
        self.surf_ds = xr.open_dataset(
            self.path / "era5_surface_2020_01.nc", engine="netcdf4", chunks=chunks
        )

        self.surf_ds = self.surf_ds.sel(latitude=self.surf_ds["latitude"][:720])
        rename_dir = {"t2m": "2t", "u10": "10u", "v10": "10v"}
        self.surf_ds = self.surf_ds.rename(rename_dir)
        self.surf_variables = surface_variables

        self.lat = torch.from_numpy(self.surf_ds["latitude"].values)
        self.lon = torch.from_numpy(self.surf_ds["longitude"].values)

        # ----- Atmosphere ----
        # self.atmos_ds = xr.open_dataset(
        #     self.path / "atmospheric.nc", engine="netcdf4", chunks=chunks
        # )
        self.atmos_ds = xr.open_dataset(
            self.path / "era5_atmospheric_2020_01.nc", engine="netcdf4", chunks=chunks
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
        self.len = len(self.times) - history - 1

        self.cur = None
        self.next = None  # target
        self.loaded_batches: dict[int, Batch] = {}

    def __len__(self) -> int:
        return self.len

    def _get_batch(self, index: int) -> Batch:
        if index in self.loaded_batches:
            return self.loaded_batches[index]

        start = self.times[index]
        target_time = self.times[index + self.history]

        time_slice = slice(start, target_time)

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
            time=(target_time,),
            atmos_levels=self.atmos_levels,
        )
        batch = Batch(
            surf_vars=surf_vars,
            static_vars=self.static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )
        self.loaded_batches[index] = batch
        return batch

    def __getitem__(self, index: int) -> tuple[Batch, Batch]:
        """
        :param index:
        :type index: int
        """
        self.cur = self._get_batch(index)
        self.next = self._get_batch(index + 1)
        return self.cur, self.next

    def collate_fn(
        self,
        data: list[tuple[Batch, Batch]],
    ) -> list[tuple[Batch, Batch]] | tuple[Batch, Batch]:
        (inputs, targets) = zip(*data)
        inputs: list[Batch]
        targets: list[Batch]
        (input_surf_vars, target_surf_vars) = [
            {
                k: torch.stack([b.surf_vars[k] for b in batches], dim=0)
                for k in self.surf_variables
            }
            for batches in (inputs, targets)
        ]
        (input_atmos_vars, target_atmos_vars) = [
            {
                k: torch.stack([b.atmos_vars[k] for b in batches], dim=0)
                for k in self.atmos_variables
            }
            for batches in (inputs, targets)
        ]

        stacked_batch_metadata = Metadata(
            lat=self.lat,
            lon=self.lon,
            time=tuple(b.metadata.time[0] for b in inputs),
            atmos_levels=self.atmos_levels,
        )
        stacked_inputs = Batch(
            surf_vars=input_surf_vars,
            static_vars=self.static_vars,
            atmos_vars=input_atmos_vars,
            metadata=stacked_batch_metadata,
        )

        stacked_targets = Batch(
            surf_vars=target_surf_vars,
            static_vars=self.static_vars,
            atmos_vars=target_atmos_vars,
            metadata=stacked_batch_metadata,
        )

        return stacked_inputs, stacked_targets


if __name__ == "__main__":
    path = Path("../data_exploration/data")
    dataset = SSTDataset(
        path,
        ["sst"],
        surface_variables=["2t", "10u", "10v", "msl", "sst", "siconc"],
        history=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    print(len(dataset))
    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
        input_batch: Batch
        # input, target = dataset[0]
        # x_atmos = torch.stack(tuple(input_batch.atmos_vars.values()), dim=2)
        # print(x_atmos.size())
        print(input_batch.atmos_vars["t"].size())
