from matplotlib import pyplot as plt
import torch
import xarray as xr
import numpy as np
from aurora import Batch, Metadata

from utils import plot, plot_rectangle, filter_all_data

filepath_instant = "reanalysis-era5-single-levels/instant.nc"
filepath_accum = "reanalysis-era5-single-levels/accumulated.nc"
filepath_wave_instant = "reanalysis-era5-single-levels/bil_remapped_wave_instant.nc"


def sensible_heat(
    sst: torch.Tensor,
    t2m: torch.Tensor,
    u10: torch.Tensor,
    rhoao: torch.Tensor,
    cp: float = 1.006e3,  # 1.006kJ
    cs: float = 1e-3,
    # cs: float = 0.9e-3,
):
    return rhoao * cp * cs * u10 * (sst - t2m)


with xr.open_dataset(
    filepath_instant, engine="netcdf4"
) as era5_instant, xr.open_dataset(
    filepath_accum, engine="netcdf4"
) as era5_accum, xr.open_dataset(
    filepath_wave_instant, engine="netcdf4"
) as era5_wave_instant:
    era5_instant, era5_accum, era5_wave_instant = filter_all_data(
        era5_instant,
        era5_accum,
        era5_wave_instant,
        longitude=slice(-60, -24),
        latitude=slice(66, 48),
        valid_time="2023-01-01T00",
    )
    # batch = Batch(
    #     surf_vars={
    #         k: torch.from_numpy(era5_instant.variables[k].values[0][None])
    #         for k in ("sst")
    #     }
    # )
    extent = (-60, -24, 66, 48)
    # long = torch.from_numpy(era5_instant.variables["longitude"].values)
    # lat = torch.from_numpy(era5_instant.variables["latitude"].values)
    long = era5_instant.variables["longitude"].values
    lat = era5_instant.variables["latitude"].values
    # lat_grid, long_grid = torch.meshgrid(lat, long)
    lat_grid, long_grid = np.meshgrid(
        era5_instant.variables["latitude"].values,
        era5_instant.variables["longitude"].values,
    )

    sst = torch.from_numpy(era5_instant.variables["sst"].values)
    skt = torch.from_numpy(era5_instant.variables["skt"].values)

    # sst = (sst + skt) / 2
    # sst = skt
    t2m = torch.from_numpy(era5_instant.variables["t2m"].values)
    u10_x = torch.from_numpy(era5_instant.variables["u10"].values)
    u10_y = torch.from_numpy(era5_instant.variables["v10"].values)
    u10 = torch.sqrt(u10_x**2 + u10_y**2)
    # TODO do this
    long_wave = torch.from_numpy(era5_wave_instant.variables["longitude"].values)
    lat_wave = torch.from_numpy(era5_wave_instant.variables["latitude"].values)
    lat_wave_grid, long_wave_grid = torch.meshgrid(lat_wave, long_wave)
    rhoao = torch.from_numpy(era5_wave_instant.variables["rhoao"].values)
    custom_sh = sensible_heat(sst, t2m, u10, rhoao)
    custom_sh_w_skt = sensible_heat(skt, t2m, u10, rhoao)

    ishf = torch.from_numpy(era5_instant.variables["ishf"].values)
    ishf = torch.where(custom_sh.isnan(), torch.nan, ishf)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    plot(
        long,
        lat,
        (abs(-custom_sh - ishf) / ishf).numpy(),
        # (-custom_sh_w_skt / ishf).numpy(),
        # (custom_sh - ishf).numpy(),
        fig=fig,
        ax=ax,
        # title="Surface 2m air temperature",
        # title="Sensible heat rel error",
        title="Given sensible heat",
        # cbar_label=r"Temperature $[C\degree]$",
        continent_overlay=True,
        # cbar_limits=(100, -300),
        cbar_limits=(1, -1),
    )
    plot_rectangle(extent, ax)
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    # plot(
    #     long,
    #     lat,
    #     # (abs(custom_sh - ishf) / ishf).numpy(),
    #     # (custom_sh - ishf).numpy(),
    #     -custom_sh.numpy(),
    #     fig=fig,
    #     ax=ax,
    #     # title="Surface 2m air temperature",
    #     # title="Sensible heat rel error",
    #     title="Calculated sensible heat",
    #     # cbar_label=r"Temperature $[C\degree]$",
    #     continent_overlay=True,
    #     cbar_limits=(100, -300),
    # )
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    # custom_sh = sensible_heat(skt, t2m, u10, rhoao)
    # plot(
    #     long,
    #     lat,
    #     # (abs(custom_sh - ishf) / ishf).numpy(),
    #     # (custom_sh - ishf).numpy(),
    #     -custom_sh_w_skt.numpy(),
    #     fig=fig,
    #     ax=ax,
    #     # title="Surface 2m air temperature",
    #     # title="Sensible heat rel error",
    #     title="Calculated sensible heat (using skin temperature)",
    #     # cbar_label=r"Temperature $[C\degree]$",
    #     continent_overlay=True,
    #     cbar_limits=(100, -300),
    # )
    # plot_rectangle(extent, ax)
    plt.show()
