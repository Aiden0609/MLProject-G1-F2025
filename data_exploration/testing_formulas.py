from typing import Any
from matplotlib import pyplot as plt
import torch
import xarray as xr
import numpy as np

# from aurora import Batch, Metadata

from utils import plot, plot_rectangle, filter_all_data

extra_name = ""
extra_name = "_full_year"

filepath_instant = f"reanalysis-era5-single-levels/instant{extra_name}.nc"
# filepath_accum = "reanalysis-era5-single-levels/accumulated.nc"
filepath_accum = f"reanalysis-era5-single-levels/flux{extra_name}.nc"
filepath_wave_instant = (
    f"reanalysis-era5-single-levels/bil_remapped_wave_instant{extra_name}.nc"
)


def accumulated_flux_to_watts(
    data: xr.Dataset, keys: dict[str, str], time_period: int = 6 * 60 * 60
) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
    new_vars = {}
    for in_key, out_key in keys.items():
        cur = np.roll(data.variables[in_key].values, 1, axis=0)
        prev = data.variables[in_key]
        diff = prev - cur
        flux_in_watts = diff / time_period

        new_vars[out_key] = flux_in_watts

    return data.assign(**new_vars)


def prepare_data(
    era5_instant: xr.Dataset,
    era5_accum: xr.Dataset,
    era5_wave_instant: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    # era5_accum = accumulated_flux_to_watts(
    #     era5_accum, {"sshf": "instant_sshf"}, time_period=60 * 60
    # )
    # Need to drop the very first time point

    era5_instant, era5_accum, era5_wave_instant = filter_all_data(
        # era5_instant, era5_accum, era5_wave_instant = filter_all_data(
        era5_instant,
        era5_accum,
        era5_wave_instant,
        # longitude=slice(-60, -24),
        # latitude=slice(66, 48),
        # valid_time=slice("2023-01-01T06", "2023-01-03T18"),
        # valid_time="2023-01-01T06",
        # valid_time="2023-01-01T18",
        # valid_time="2023-01",
        # valid_time=slice("2023-07", "2023-08"),
    )
    era5_instant = era5_instant.mean("valid_time")
    era5_wave_instant = era5_wave_instant.mean("valid_time")
    era5_accum = era5_accum.mean("valid_time")
    return era5_instant, era5_accum, era5_wave_instant


def sensible_heat(
    sst: torch.Tensor,
    t2m: torch.Tensor,
    u10: torch.Tensor,
    rhoao: torch.Tensor,
    cp: float = 1.006e3,  # 1.006kJ
    cs: float = 1e-3,
    # cs: float = 0.9e-3,
):
    return -rhoao * cp * cs * u10 * (sst - t2m)


with xr.open_dataset(
    filepath_instant, engine="netcdf4", chunks="auto"
) as era5_instant, xr.open_dataset(
    filepath_accum, engine="netcdf4", chunks="auto"
) as era5_accum, xr.open_dataset(
    filepath_wave_instant, engine="netcdf4", chunks="auto"
) as era5_wave_instant:
    era5_instant, era5_accum, era5_wave_instant = prepare_data(
        era5_instant, era5_accum, era5_wave_instant
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
    istl1 = torch.from_numpy(era5_instant.variables["istl1"].values)
    ci = torch.from_numpy(era5_instant.variables["siconc"].values)  # sea ice cover

    sst = torch.from_numpy(era5_instant.variables["sst"].values) - 273
    skt = torch.from_numpy(era5_instant.variables["skt"].values) - 273

    sst = torch.where(ci > 0.5, torch.nan, sst)
    # sst = ci * istl1 + (1 - ci) * sst

    # sst = (sst + skt) / 2
    # sst = skt
    t2m = torch.from_numpy(era5_instant.variables["t2m"].values) - 273
    u10_x = torch.from_numpy(era5_instant.variables["u10"].values)
    u10_y = torch.from_numpy(era5_instant.variables["v10"].values)
    u10 = torch.sqrt(u10_x**2 + u10_y**2)

    rhoao = torch.from_numpy(era5_wave_instant.variables["rhoao"].values)
    custom_sh = sensible_heat(sst, t2m, u10, rhoao)
    custom_sh_w_skt = sensible_heat(skt, t2m, u10, rhoao)

    # ishf = torch.from_numpy(era5_instant.variables["ishf"].values)
    # ishf = torch.from_numpy(era5_accum.variables["instant_sshf"].values)
    ishf = torch.from_numpy(era5_accum.variables["avg_ishf"].values)
    ishf = torch.where(custom_sh.isnan(), torch.nan, ishf)
    print(torch.nanmean(ishf))
    print(np.nanstd(ishf.numpy()))
    delta_ishf = ishf - torch.roll(ishf, 1)

    fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    # ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = fig.subplots(3, 2, sharex=True, sharey=True)
    ((ax1, ax2), (ax3, ax4), (ax5, _)) = fig.subplots(3, 2, sharex=True, sharey=True)
    # plot(
    #     long,
    #     lat,
    #     # (abs(-custom_sh - ishf) / ishf).numpy(),
    #     # (-custom_sh_w_skt / ishf).numpy(),
    #     # (custom_sh - ishf).numpy(),
    #     # 10 * ishf.numpy(),
    #     # ishf.numpy(),
    #     # rhoao.numpy(),
    #     sst.numpy(),
    #     # delta_ishf.numpy(),
    #     fig=fig,
    #     ax=ax1,
    #     # title="Surface 2m air temperature",
    #     # title="Sensible heat rel error",
    #     # title="Given sensible heat",
    #     # title="Air density",
    #     title="sst",
    #     # cbar_label=r"Temperature $[C\degree]$",
    #     continent_overlay=True,
    #     # cbar_limits=(100, -300),
    #     # cbar_limits=(30, -30),
    #     # cbar_limits=(10, -10),
    #     # cbar_limits=(150, -500),
    # )
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    # plot(
    #     long,
    #     lat,
    #     # (abs(-custom_sh - ishf) / ishf).numpy(),
    #     # (-custom_sh_w_skt / ishf).numpy(),
    #     # (custom_sh - ishf).numpy(),
    #     # 10 * ishf.numpy(),
    #     # ishf.numpy(),
    #     # rhoao.numpy(),
    #     t2m.numpy(),
    #     # delta_ishf.numpy(),
    #     fig=fig,
    #     ax=ax2,
    #     # title="Surface 2m air temperature",
    #     # title="Sensible heat rel error",
    #     # title="Given sensible heat",
    #     # title="Air density",
    #     title="t2m",
    #     # cbar_label=r"Temperature $[C\degree]$",
    #     continent_overlay=True,
    #     # cbar_limits=(100, -300),
    #     # cbar_limits=(30, -30),
    #     # cbar_limits=(10, -10),
    #     # cbar_limits=(150, -500),
    # )
    # ishf = torch.from_numpy(era5_instant.variables["ishf"].values)
    # ishf = torch.where(custom_sh.isnan(), torch.nan, ishf)
    # plot_rectangle(extent, ax)
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    delta_custom_sh = custom_sh - torch.roll(custom_sh, 1)
    plot(
        long,
        lat,
        # (abs(custom_sh - ishf) / ishf).numpy(),
        # (custom_sh - ishf).numpy(),
        # custom_sh.numpy(),
        u10.numpy(),
        # -delta_custom_sh.numpy(),
        # ishf.numpy(),
        fig=fig,
        # ax=ax3,
        ax=ax1,
        # title="Surface 2m air temperature",
        # title="Sensible heat rel error",
        # title="Calculated sensible heat",
        title="Wind speed",
        # cbar_label=r"Temperature $[C\degree]$",
        continent_overlay=True,
        # cbar_limits=(150, -500),
        # cbar_limits=(10, -10),
        cbar_discrete=True,
    )
    # plt.show()
    # quit()
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    # custom_sh = sensible_heat(skt, t2m, u10, rhoao)
    plot(
        long,
        lat,
        (sst - t2m).numpy(),
        fig=fig,
        # ax=ax4,
        ax=ax3,
        title="sst and t2m diff",
        continent_overlay=True,
        # cbar_limits=(-3, 3),
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        # (abs(-custom_sh - ishf) / ishf).numpy(),
        # (-custom_sh_w_skt / ishf).numpy(),
        # (custom_sh - ishf).numpy(),
        # 10 * ishf.numpy(),
        # ishf.numpy(),
        # rhoao.numpy(),
        rhoao.numpy(),
        # delta_ishf.numpy(),
        fig=fig,
        # ax=ax4,
        ax=ax5,
        # title="Surface 2m air temperature",
        # title="Sensible heat rel error",
        # title="Given sensible heat",
        # title="Air density",
        title="Air pressure",
        # cbar_label=r"Temperature $[C\degree]$",
        continent_overlay=True,
        # cbar_limits=(100, -300),
        # cbar_limits=(30, -30),
        # cbar_limits=(10, -10),
        # cbar_limits=(1.2, 1.3),
        # cbar_limits=(150, -500),
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        (abs(custom_sh - ishf) / abs(ishf)).numpy(),
        # (abs(delta_custom_sh - delta_ishf) / delta_ishf).numpy(),
        # (custom_sh - ishf).numpy(),
        # -custom_sh_w_skt.numpy(),
        fig=fig,
        # ax=ax5,
        ax=ax4,
        # title="Surface 2m air temperature",
        title="Sensible heat rel error",
        # title="Calculated sensible heat (using skin temperature)",
        # cbar_label=r"Temperature $[C\degree]$",
        continent_overlay=True,
        # cbar_limits=(100, -300),
        # cbar_limits=(5, -5),
        cbar_limits=(1, 0),
        # cbar_limits=(10, -10),
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        # (abs(-custom_sh - ishf) / ishf).numpy(),
        # (-custom_sh_w_skt / ishf).numpy(),
        # (custom_sh - ishf).numpy(),
        # 10 * ishf.numpy(),
        # ishf.numpy(),
        # rhoao.numpy(),
        # (abs(custom_sh - ishf)).numpy(),
        (custom_sh - ishf).numpy(),
        # delta_ishf.numpy(),
        fig=fig,
        # ax=ax4,
        ax=ax2,
        # title="Surface 2m air temperature",
        # title="Sensible heat rel error",
        # title="Given sensible heat",
        # title="Air density",
        # title="Sensible heat abs error",
        title="Sensible heat error",
        # cbar_label=r"Temperature $[C\degree]$",
        continent_overlay=True,
        # cbar_limits=(100, -300),
        # cbar_limits=(30, -30),
        # cbar_limits=(10, -10),
        # cbar_limits=(0, 10),
        # cbar_limits=(150, -500),
        cbar_discrete=True,
    )
    # plot_rectangle(extent, ax)
    plt.show()
