from typing import Any
from matplotlib import pyplot as plt

# import matplotlib.gridspec as gridspec
import torch
import xarray as xr
import numpy as np

# from aurora import Batch, Metadata

from utils import plot, filter_all_data

extra_name = ""
extra_name = "_full_year"

filepath_instant = f"reanalysis-era5-single-levels/instant{extra_name}.nc"
filepath_accum = f"reanalysis-era5-single-levels/flux{extra_name}.nc"
filepath_wave_instant = (
    f"reanalysis-era5-single-levels/bil_remapped_wave_instant{extra_name}.nc"
)
filepath_invarariant = f"reanalysis-era5-single-levels/invariant{extra_name}.nc"


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
    *data: xr.Dataset,
) -> list[xr.Dataset]:
    # era5_accum = accumulated_flux_to_watts(
    #     era5_accum, {"sshf": "instant_sshf"}, time_period=60 * 60
    # )
    # Need to drop the very first time point

    data = filter_all_data(
        # era5_instant, era5_accum, era5_wave_instant = filter_all_data(
        *data,
        # longitude=slice(-60, -24),
        latitude=slice(90, -60),
        # valid_time=slice("2023-07-01T06", "2023-07-01T18"),
        valid_time=slice("2023-07-01T18", "2023-07-08T06"),
        # valid_time="2023-01-01T06",
        # valid_time="2023-01-01",
        # valid_time="2023-07-01",
        # valid_time=slice("2023-07", "2023-08"),
    )
    return data
    # mean_data = []
    # for i in range(len(data)):
    #     mean_data.append(data[i].mean("valid_time"))
    # return mean_data


def diff(tensor: torch.Tensor):
    return tensor[1] - tensor[0]


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
    filepath_instant, engine="netcdf4"
) as era5_instant, xr.open_dataset(
    filepath_accum, engine="netcdf4"
) as era5_accum, xr.open_dataset(
    filepath_invarariant, engine="netcdf4"
) as era5_invariant, xr.open_dataset(
    filepath_wave_instant, engine="netcdf4"
) as era5_wave_instant:
    era5_instant, era5_accum, era5_invariant, era5_wave_instant = prepare_data(
        era5_instant,
        era5_accum,
        era5_invariant,
        era5_wave_instant,
        # valid_time="2023-01-01T06",
    )
    long = era5_instant.variables["longitude"].values
    lat = era5_instant.variables["latitude"].values

    sst = torch.from_numpy(era5_instant.variables["sst"].values) - 273

    t2m = torch.from_numpy(era5_instant.variables["t2m"].values) - 273
    u10_x = torch.from_numpy(era5_instant.variables["u10"].values)
    u10_y = torch.from_numpy(era5_instant.variables["v10"].values)
    u10 = torch.sqrt(u10_x**2 + u10_y**2)

    rhoao = torch.from_numpy(era5_wave_instant.variables["rhoao"].values)

    d_sst = diff(sst)
    d_t2m = diff(t2m)
    d_u10 = diff(u10)
    # d_rhoao = diff(rhoao)
    d_rhoao = rhoao[0]

    # d_sh = sensible_heat(d_sst, d_t2m, d_u10, d_rhoao)
    sh = sensible_heat(sst, t2m, u10, rhoao)
    d_sh = diff(sh)

    ishf = torch.from_numpy(era5_accum.variables["avg_ishf"].values)
    d_ishf = diff(ishf)
    d_ishf = torch.where(d_sh.isnan(), torch.nan, d_ishf)

    fig = plt.figure(figsize=(16, 9))
    ((ax1, ax2), (ax3, ax4), (ax5, _)) = fig.subplots(3, 2, sharex=True, sharey=True)

    plot(
        long,
        lat,
        d_u10.numpy(),
        fig=fig,
        ax=ax1,
        title="Wind speed",
        continent_overlay=True,
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        (d_sst - d_t2m).numpy(),
        fig=fig,
        ax=ax3,
        title="sst and t2m diff",
        continent_overlay=True,
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        d_rhoao.numpy(),
        fig=fig,
        ax=ax5,
        title="Air pressure",
        continent_overlay=True,
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        # (abs(d_sh - d_ishf) / abs(d_ishf) - 1).numpy(),
        (abs(d_ishf - d_sh) / abs(d_sh)).numpy(),
        fig=fig,
        ax=ax4,
        title="Sensible heat rel error",
        continent_overlay=True,
        cbar_limits=(1, 0),
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        (d_sh - d_ishf).numpy(),
        fig=fig,
        ax=ax2,
        title="Sensible heat error",
        continent_overlay=True,
        cbar_discrete=True,
    )
    plt.show()
