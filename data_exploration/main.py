import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import plot, plot_rectangle

filepath = "reanalysis-era5-single-levels/instant.nc"
# filepath = "reanalysis-era5-single-levels/con_remapped_wave_instant.nc"
filepath2 = "reanalysis-era5-single-levels/bil_remapped_wave_instant.nc"

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()

with xr.open_dataset(filepath, engine="netcdf4") as era5, xr.open_dataset(
    filepath2, engine="netcdf4"
) as era5_2:
    # print(era5.coords["valid_time"].values)
    # quit()
    extent = (-60, -24, 66, 48)
    # print(era5.variables)
    # quit()
    era5_narrow = era5.sel(
        # longitude=slice(-60, -24),
        # latitude=slice(66, 48),
        # latitude=slice(63, 48),
        # longitude=slice(-65, -19),
        # latitude=slice(70, 44),
        latitude=slice(73, -60),
        valid_time="2023-01-01T00",
        # pressure_level=1000,
    )
    era5_narrow_2 = era5_2.sel(
        # longitude=slice(-60, -24),
        # latitude=slice(66, 48),
        # longitude=slice(-65, -19),
        # latitude=slice(70, 44),
        valid_time="2023-01-01T00",
        # pressure_level=1000,
    )
    long = era5_narrow.variables["longitude"]
    lat = era5_narrow.variables["latitude"]
    t = era5_narrow.variables["skt"].to_numpy()  # - 273
    t2 = era5_narrow.variables["sst"].to_numpy()
    diff = t - t2
    std = np.nanstd(diff)
    diff = np.where(abs(diff) < 2 * std, diff, np.nan)
    plot(
        long,
        lat,
        # t,
        diff,
        fig=fig,
        ax=ax,
        # title="Surface 2m air temperature",
        title="Surface sea temperature and skin temperature diff",
        # title="Air density over the oceans",
        cbar_label=r"Temperature $[C\degree]$",
        # cbar_label=r"Air density $[kg m^{-3}]$",
        continent_overlay=True,
    )
    plot_rectangle(extent, ax)

plt.show()
