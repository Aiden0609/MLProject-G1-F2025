def make_Ramp(ramp_colors):
    from matplotlib.colors import LinearSegmentedColormap

    color_ramp = LinearSegmentedColormap.from_list("my_list", ramp_colors)
    # uncommenting these will display the colorramp
    # plt.figure(figsize=(15, 3))
    # plt.imshow(
    #     [list(np.arange(0, len(ramp_colors), 0.1))],
    #     interpolation="nearest",
    #     origin="lower",
    #     cmap=color_ramp,
    # )
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    return color_ramp
