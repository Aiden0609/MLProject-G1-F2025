import matplotlib.pyplot as plt
import xarray as xr
from utils import plot, plot_rectangle

filepath = "reanalysis-era5-single-levels/instant.nc"

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()

with xr.open_dataset(filepath, engine="netcdf4") as era5:
    # print(era5.coords["valid_time"].values)
    # quit()
    extent = (-60, -24, 66, 48)
    # print(era5.variables)
    # quit()
    era5_narrow = era5.sel(
        # longitude=slice(-60, -24),
        # latitude=slice(66, 48),
        # longitude=slice(-65, -19),
        # latitude=slice(70, 44),
        valid_time="2023-01-01T00",
        # pressure_level=1000,
    )
    long = era5_narrow.variables["longitude"]
    lat = era5_narrow.variables["latitude"]
    t = era5_narrow.variables["t2m"] - 273
    plot(
        long,
        lat,
        t.to_numpy(),
        fig=fig,
        ax=ax,
        # title="Surface 2m air temperature",
        title="Surface sea temperature",
        cbar_label=r"Temperature $[C\degree]$",
        continent_overlay=True,
    )
    plot_rectangle(extent, ax)

plt.show()
