import cmocean
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    title: str = None,
    cbar_label: str = None,
    continent_overlay: bool = False,
):
    extent = get_extent(x, y)
    img = ax.imshow(data, origin="upper", extent=extent, cmap=cmocean.cm.thermal)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img, cax=cax)
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


def _value_to_index(value_slice: tuple[float, float], variable: xr.Variable) -> slice:
    idx1 = np.argmin(abs(value_slice[0] - variable[:]))
    idx2 = np.argmin(abs(value_slice[1] - variable[:]))
    return slice(int(min(idx1, idx2)), int(max(idx1, idx2)))


def _narrow_variables(data: Dataset, axis: int, mask: slice | int):
    for key, variable in data.variables.items():
        match axis:
            case 0:
                narrowed_var = variable[mask]
            case 1:
                narrowed_var = variable[:, mask]
            case 2:
                narrowed_var = variable[:, :, mask]
            case 3:
                narrowed_var = variable[:, :, :, mask]
            case _:
                raise ValueError("axis is wrong")

        data.variables[key] = narrowed_var


def narrow_data(
    data: Dataset, extent_dict: dict[str, tuple[float, float] | int]
) -> Dataset:
    print(data)
    # _value_to_index needs to deal with grid variables
    if "longitude" in extent_dict:
        extent = extent_dict["longitude"]
        if isinstance(extent, int):
            mask = extent
        else:
            mask = _value_to_index(extent, data.variables["longitude"])
        _narrow_variables(data, axis=3, mask=mask)

    if "latitude" in extent_dict:
        extent = extent_dict["latitude"]
        if isinstance(extent, int):
            mask = extent
        else:
            mask = _value_to_index(extent, data.variables["latitude"])
        _narrow_variables(data, axis=2, mask=mask)

    if "pressure_level" in extent_dict:
        extent = extent_dict["pressure_level"]
        if isinstance(extent, int):
            mask = extent
        else:
            mask = _value_to_index(extent, data.variables["pressure_level"])
        _narrow_variables(data, axis=1, mask=mask)

    if "valid_time" in extent_dict:
        extent = extent_dict["valid_time"]
        if isinstance(extent, int):
            mask = extent
        else:
            mask = _value_to_index(extent, data.variables["valid_time"])
        _narrow_variables(data, axis=0, mask=mask)

    quit()


if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()

    with xr.open_dataset(filepath, engine="netcdf4") as era5:
        narrow_data(era5, {"longitude": (-60, -24)})
        long = era5.variables["longitude"]
        lat = era5.variables["latitude"]
        t = era5.variables["t"][0, 0] - 273
        plot(
            long,
            lat,
            t.to_numpy(),
            fig=fig,
            ax=ax,
            title="Surface 2m air temperature",
            cbar_label=r"Temperature $[C\degree]$",
            continent_overlay=True,
        )

    plt.show()
