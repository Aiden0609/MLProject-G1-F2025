from matplotlib import pyplot as plt
import torch
import xarray as xr
import numpy as np
from aurora import Batch, Metadata
from mpl_toolkits import basemap

from utils import plot, plot_rectangle

filepath_instant = "reanalysis-era5-single-levels/instant.nc"
filepath_accum = "reanalysis-era5-single-levels/accum.nc"
filepath_wave_instant = "reanalysis-era5-single-levels/wave_instant.nc"


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()


def sensible_heat(
    sst: torch.Tensor,
    t2m: torch.Tensor,
    u10: torch.Tensor,
    rhoao: torch.Tensor,
    cp: float = 1.006,
    cs: float = 1e-3,
):
    return rhoao * cp * cs * u10 * (sst - t2m)


with xr.open_dataset(
    filepath_instant, engine="netcdf4"
) as era5_instant, xr.open_dataset(
    filepath_accum, engine="netcdf4"
) as era5_accum, xr.open_dataset(
    filepath_wave_instant, engine="netcdf4"
) as era5_wave_instant:
    # batch = Batch(
    #     surf_vars={
    #         k: torch.from_numpy(era5_instant.variables[k].values[0][None])
    #         for k in ("sst")
    #     }
    # )
    extent = (-60, -24, 66, 48)
    long = torch.from_numpy(era5_instant.variables["longitude"].values)
    lat = torch.from_numpy(era5_instant.variables["latitude"].values)
    # lat_grid, long_grid = torch.meshgrid(lat, long)
    lat_grid, long_grid = np.meshgrid(
        era5_instant.variables["latitude"].values,
        era5_instant.variables["longitude"].values,
    )

    sst = torch.from_numpy(era5_instant.variables["sst"].values[0])
    t2m = torch.from_numpy(era5_instant.variables["t2m"].values[0])
    u10_x = torch.from_numpy(era5_instant.variables["u10"].values[0])
    u10_y = torch.from_numpy(era5_instant.variables["v10"].values[0])
    u10 = torch.sqrt(u10_x**2 + u10_y**2)
    # TODO do this
    long_wave = torch.from_numpy(era5_wave_instant.variables["longitude"].values)
    lat_wave = torch.from_numpy(era5_wave_instant.variables["latitude"].values)
    lat_wave_grid, long_wave_grid = torch.meshgrid(lat_wave, long_wave)
    rhoao_regridded = basemap.interp(
        era5_wave_instant.variables["rhoao"].values[0, ::-1],
        # long_wave_grid,
        # torch.flip(lat_wave_grid, dims=(0,)),
        long_wave,
        torch.flip(lat_wave, dims=(0,)),
        long_grid,
        # torch.flip(lat_grid, dims=(0,)),
        np.flip(lat_grid, axis=0),
        # long,
        # lat,
        order=1,
    )[::-1]
    rhoao = torch.from_numpy(rhoao_regridded)
    # rhoao = torch.nn.functional.interpolate(
    #     torch.from_numpy(era5_wave_instant.variables["rhoao"].values), sst.shape
    # )[0]
    print(sst.shape, t2m.shape, u10.shape, rhoao.shape)
    quit()
    # custom_sh = sensible_heat(sst, t2m, u10, rhoao)
    # ishf = torch.from_numpy(era5_accum.variables["ishf"].values[0])
    # plot(
    #     long,
    #     lat,
    #     custom_sh - ishf,
    #     fig=fig,
    #     ax=ax,
    #     # title="Surface 2m air temperature",
    #     title="Surface sea temperature",
    #     cbar_label=r"Temperature $[C\degree]$",
    #     continent_overlay=True,
    # )
    # plot_rectangle(extent, ax)
    # metadata = Metadata(
    #     lat=torch.from_numpy(era5_instant.variables["latitude"].values),
    #     long=torch.from_numpy(era5_instant.variables["longitude"].values),
    #     time=(
    #         era5_instant.variables["valid_time"]
    #         .values.astype("datetime64[s]")
    #         .tolist()[1],
    #     ),
    # )
