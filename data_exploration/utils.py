import cmocean
import numpy as np
import xarray as xr
from xarray import Dataset, Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd


def filter_all_data(*datasets: Dataset, **filters):
    filtered_datasets = []
    for dataset in datasets:
        filtered_datasets.append(dataset.sel(**filters))

    return filtered_datasets


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


world = None


@shaded_image
def plot(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    *,
    fig: plt.Figure,
    ax: plt.Axes,
    title: str = None,
    cbar_label: str = None,
    cbar_limits: tuple[float, float] = None,
    continent_overlay: bool = False,
):
    extent = get_extent(x, y)
    norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    if cbar_limits is not None:
        cbar_min, cbar_max = cbar_limits
        norm = Normalize(vmin=cbar_min, vmax=cbar_max)
    img = ax.imshow(
        data, origin="upper", extent=extent, cmap=cmocean.cm.thermal, norm=norm
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmocean.cm.thermal), cax=cax)
    if continent_overlay:
        global world
        if world is None:
            # Land -> https://www.naturalearthdata.com/downloads/10m-physical-vectors/
            world = gpd.read_file("ne_10m_land/ne_10m_land.shp")
        world.plot(ax=ax, edgecolor="black", color="None")

    ax.set_xlim(extent[:2])
    ax.set_xlabel("Longitude")
    ax.set_ylim(extent[2:])
    ax.set_ylabel("Latitude")

    if title is not None:
        ax.set_title(title)
    if cbar_label is not None:
        cbar.set_label(cbar_label)


def plot_rectangle(extent: tuple[float, float, float, float], ax: plt.Axes):
    x1, x2, y1, y2 = extent
    return ax.fill(
        (x1, x1, x2, x2),
        (y1, y2, y2, y1),
        edgecolor="w",
        facecolor="None",
        linewidth=2,
        zorder=1,
    )
