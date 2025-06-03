import matplotlib.pyplot as plt
import math
from typing import List

from src.utils.plotter import Plotter


class CombinedPlotter:
    """
    A class to combine multiple Plotter objects into a single figure with a defined number of columns.
    """

    def __init__(self, plotters: List[Plotter], columns: int) -> None:
        """
        Initialize the CombinedPlotter.

        :param plotters: A list of Plotter objects to be combined.
        :param columns: The number of columns in the grid layout.
        """
        self.plotters = plotters
        self.columns = columns
        self.rows = math.ceil(len(plotters) / columns)

    def plot_combined(self, fig_width: int = 15, fig_height: int = 10, disable_show: bool = False) -> plt.Figure:
        """
        Create a combined figure with subplots for each Plotter object.

        :param fig_width: Width of the combined figure.
        :param fig_height: Height of each subplot.
        :param disable_show: Whether to disable showing the figure.
        :return: The combined figure with all subplots.
        """
        num_plots = len(self.plotters)
        total_height = fig_height * self.rows

        # Create the figure and subplots
        fig, axs = plt.subplots(self.rows, self.columns, figsize=(fig_width, total_height))

        # Flatten axes if there are multiple rows and columns
        if self.rows == 1 and self.columns == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for i, plotter in enumerate(self.plotters):
            if i < len(axs):
                plotter.plot(disable_show=True, fig_width=fig_width, fig_height_per_plot=fig_height)
                axs[i].set_title(plotter.title)

        # Hide any extra empty subplots
        for j in range(num_plots, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()

        if not disable_show:
            plt.show()

        return fig