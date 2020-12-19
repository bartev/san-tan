import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def prepareSubplot(
    xticks,
    yticks,
    figsize=(10.5, 6),
    hideLabels=False,
    gridColor="#999999",
    gridWidth=1.0,
    subplots=(1, 1),
):
    """Template for generating the plot layout.
    Used in plotting the k-means results"""
    plt.close()
    fig, axList = plt.subplots(
        nrows=subplots[0],
        ncols=subplots[1],
        figsize=figsize,
        facecolor="white",
        edgecolor="white",
    )
    if not isinstance(axList, np.ndarray):
        axList = np.array([axList])

    for ax in axList.flatten():
        ax.axes.tick_params(labelcolor="#999999", labelsize="10")
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position("none")
            axis.set_ticks(ticks)
            axis.label.set_color("#999999")
            if hideLabels:
                axis.set_ticklabels([])
        ax.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
        map(
            lambda position: ax.spines[position].set_visible(False),
            ["bottom", "top", "left", "right"],
        )

    if axList.size == 1:
        axList = axList[0]  # Just return a single axes object for a regular plot
    return fig, axList
