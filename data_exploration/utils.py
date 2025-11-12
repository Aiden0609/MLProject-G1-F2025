import cmocean
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import geopandas as gpd


def _make_Ramp(ramp_colors):
    from matplotlib.colors import LinearSegmentedColormap

    color_ramp = LinearSegmentedColormap.from_list("my_list", ramp_colors)
    return color_ramp


VE = 0.00009000  # "vertical exxageration". this number will change depending on your DEM resolution and units
LS = LightSource(azdeg=270, altdeg=45)


def get_extent(x: np.ndarray, y: np.ndarray):
    extent: tuple[float, float, float, float] = (
        np.min(x),
        np.max(x),
        np.min(y),
        np.max(y),
    )
    return extent


def shaded_image(plot_func, ve: float = VE):
    gray_ramp = _make_Ramp(["#000000", "#ffffff00"])

    def wrapper(x: np.ndarray, y: np.ndarray, data: np.ndarray, *args, **kwargs):
        hillshade = LS.hillshade(data, vert_exag=ve)  # dx=?, dy=?
        # unsafe
        ax: plt.Axes = kwargs["ax"]
        ax.imshow(hillshade, origin="lower", extent=get_extent(x, y), cmap=gray_ramp)
        plot_output = plot_func(x, y, data, *args, **kwargs)
        return plot_output

    return wrapper


filepath = "reanalysis-era5-pressure-levels.nc"
world = None
# print(world)


@shaded_image
def plot(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    *,
    fig: plt.Figure,
    ax: plt.Axes,
    continent_overlay: bool = False
):
    img = ax.imshow(
        data, origin="upper", extent=get_extent(x, y), cmap=cmocean.cm.thermal
    )
    fig.colorbar(img)
    if continent_overlay:
        global world
        if world is None:
            # Land -> https://www.naturalearthdata.com/downloads/10m-physical-vectors/
            world = gpd.read_file("ne_10m_land/ne_10m_land.shp")
        world.plot(ax=ax, edgecolor="black", color="None")
    ax.set_xlim((-180, 180))
    ax.set_ylim((-90, 90))


if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()

    with xr.open_dataset(filepath, engine="netcdf4") as era5:
        long = era5.variables["longitude"]
        lat = era5.variables["latitude"]
        t = era5.variables["t"][0, 0] - 273
        plot(long, lat, t.to_numpy(), fig=fig, ax=ax, continent_overlay=True)

    plt.show()
