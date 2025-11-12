import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np


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
