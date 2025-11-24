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
        # valid_time=slice("2023-01-01T06", "2023-01-03T18"),
        # valid_time="2023-01-01T06",
        valid_time="2023-01-01",
        # valid_time="2023-07-01",
        # valid_time=slice("2023-07", "2023-08"),
    )
    mean_data = []
    for i in range(len(data)):
        mean_data.append(data[i].mean("valid_time"))
    return mean_data
    # return data


def energy_balance(
    sst: torch.Tensor,
    skt: torch.Tensor,
    sw: torch.Tensor,
    lw: torch.Tensor,
    sh: torch.Tensor,
    lh: torch.Tensor,
    conductivity: float,
) -> torch.Tensor:
    return conductivity * (skt - sst) - (sw + lw + sh + lh)
    # return sw + lw + sh + lh


with xr.open_dataset(
    filepath_instant, engine="netcdf4"
) as era5_instant, xr.open_dataset(
    filepath_accum, engine="netcdf4"
) as era5_accum, xr.open_dataset(
    filepath_invarariant, engine="netcdf4"
) as era5_invariant:
    era5_instant, era5_accum, era5_invariant = prepare_data(
        era5_instant,
        era5_accum,
        era5_invariant,
        # valid_time="2023-01-01T06",
    )
    long = era5_instant.variables["longitude"].values
    lat = era5_instant.variables["latitude"].values
    ishf = torch.from_numpy(era5_instant.variables["ishf"].values)
    # sst = torch.from_numpy(era5_instant.variables["sst"].values)
    sst = torch.from_numpy(era5_instant.variables["stl1"].values)
    skt = torch.from_numpy(era5_instant.variables["skt"].values)

    # accum_sshf = torch.from_numpy(era5_accum.variables["instant_sshf"].values)
    sh = torch.from_numpy(era5_accum.variables["avg_ishf"].values)
    lh = torch.from_numpy(era5_accum.variables["avg_slhtf"].values)
    lw = torch.from_numpy(era5_accum.variables["avg_snlwrf"].values)
    # lw = torch.from_numpy(era5_accum.variables["avg_snlwrfcs"].values)
    sw = torch.from_numpy(era5_accum.variables["avg_snswrf"].values)
    # sw = torch.from_numpy(era5_accum.variables["avg_snlwrfcs"].values)

    lsm = torch.from_numpy(era5_invariant.variables["lsm"].values)

    sh = torch.where(lsm < 0.8, torch.nan, sh)
    lh = torch.where(lsm < 0.8, torch.nan, lh)
    lw = torch.where(lsm < 0.8, torch.nan, lw)
    sw = torch.where(lsm < 0.8, torch.nan, sw)

    # energy_surplus = energy_balance(sst, skt, sh, lh, lw, sw, 0.6)
    conductivity = (
        7 * 1
    )  # W m^-2 K^-1 from https://journals.ametsoc.org/view/journals/clim/8/11/1520-0442_1995_008_2716_ailsps_2_0_co_2.xml?tab_body=pdf
    energy_surplus = energy_balance(sst, skt, sh, lh, lw, sw, conductivity)
    print(torch.nanmean(energy_surplus))
    fig = plt.figure(figsize=(16, 9))
    # gs1 = gridspec.GridSpec(3, 1)
    # ax1 = fig.add_subplot(gs1[0])
    # ax2 = fig.add_subplot(gs1[1])
    # ax3 = fig.add_subplot(gs1[2])
    # ax1 = fig.add_subplot()
    # ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True, sharey=True)
    ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = fig.subplots(3, 2, sharex=True, sharey=True)
    plot(
        long,
        lat,
        conductivity * (skt - sst).numpy(),
        fig=fig,
        ax=ax1,
        title="RHS",
        continent_overlay=True,
        # cbar_limits=(-3, 3),
        # cbar_discrete=True,
    )
    plot(
        long,
        lat,
        # abs(energy_surplus).numpy(),
        energy_surplus.numpy(),
        fig=fig,
        ax=ax2,
        title="surplus",
        continent_overlay=True,
        # cbar_limits=(-3, 3),
        cbar_discrete=True,
    )
    plot(
        long,
        lat,
        lh.numpy(),
        fig=fig,
        ax=ax3,
        title="LH",
        continent_overlay=True,
        # cbar_limits=(-10, 10),
        # cbar_discrete=True,
    )
    plot(
        long,
        lat,
        sh.numpy(),
        fig=fig,
        ax=ax4,
        title="SH",
        continent_overlay=True,
        # cbar_limits=(-10, 10),
        # cbar_discrete=True,
    )
    plot(
        long,
        lat,
        lw.numpy(),
        fig=fig,
        ax=ax5,
        title="LW",
        continent_overlay=True,
        # cbar_limits=(-10, 10),
        # cbar_discrete=True,
    )
    plot(
        long,
        lat,
        sw.numpy(),
        fig=fig,
        ax=ax6,
        title="SW",
        continent_overlay=True,
        # cbar_limits=(-10, 10),
        # cbar_discrete=True,
    )

plt.show()
