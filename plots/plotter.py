import textwrap

import matplotlib.pyplot as plt
import colorsys
from typing import List, Tuple, Optional, Callable, Literal, TypeVar

T = TypeVar('T')


class ColorIterator:
    """
    A class to iterate over colors
    """

    def __init__(self, start_hue: int = 0, saturation: float = 0.5, lightness: float = 0.5) -> None:
        """
        Initialize the color iterator
        :param start_hue: The starting hue
        :param saturation: The saturation
        :param lightness: The lightness
        """
        self.current_hue = start_hue
        self.saturation = saturation
        self.lightness = lightness

    def __iter__(self) -> 'ColorIterator':
        return self

    def __next__(self) -> str:
        """
        Get the next color
        :return: str: The color code in hex format
        """
        self.current_hue = (self.current_hue + 10) % 360
        r, g, b = colorsys.hls_to_rgb(self.current_hue / 360.0, self.lightness, self.saturation)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


class Plotter:
    def __init__(self,
                 title: str = "",
                 y_axis_start_at_zero: bool = True,
                 columns: Optional[int] = None) -> None:
        """
        A plotter class to plot data
        :param title: The title of the plot
        :param y_axis_start_at_zero: Boolean flag to indicate if the y-axis should always start at zero
        :param columns: Number of columns for subplots
        """
        self.data_series: List[
            Tuple[List[Tuple[str, List[float], Optional[str], Optional[float]]], bool, str, List[float], Optional[str]]] = []
        self.marked_areas: List[Tuple[int, int, str, float, Optional[str], Literal["top", "middle", "bottom"], Literal[
            "horizontal", "vertical"]]] = []
        self.vertical_lines: List[Tuple[int, str, float, Optional[str], Literal["top", "middle", "bottom"]]] = []
        self.marked_points: List[Tuple[int, str]] = []
        self.tables: List[Tuple[List[List[T]], Optional[List[str]], Optional[Callable[[T], Optional[str]]]]] = []
        self.color_iterator: ColorIterator = ColorIterator()
        self.title: str = title
        self.y_axis_start_at_zero: bool = y_axis_start_at_zero
        self.DEFAULT_COLOR: str = "#445469"
        self.DEFAULT_THICKNESS: float = 2.0
        self.columns: int = columns if columns else 1

    def add_subplot(self,
                    series: List[Tuple[str, List[float], Optional[str], Optional[float]]],
                    x_range: List[float],
                    include_extras: bool = True,
                    y_label: str = "Y-Axis",
                    vertical_lines: Optional[List[
                        Tuple[int, str, float, Optional[str], Literal["top", "middle", "bottom"]]]] = None,
                    title: Optional[str] = None) -> None:
        """
        Add a subplot to the plot with a specific x-range and optional vertical lines.
        :param series: The (independent) data series to be added.
        :param x_range: The x-axis data specific to this subplot.
        :param include_extras: Boolean flag to include additional visual elements.
        :param y_label: The label for the y-axis specific to this subplot.
        :param vertical_lines: List of tuples defining vertical lines for this subplot.
        :param title: The title for the subplot.
        :return: None.
        """
        updated_series = []
        for entry in series:
            if len(entry) == 2:
                label, data_series = entry
                color = self.DEFAULT_COLOR
                thickness = self.DEFAULT_THICKNESS
            elif len(entry) == 3:
                label, data_series, color = entry
                thickness = self.DEFAULT_THICKNESS
            else:
                label, data_series, color, thickness = entry

            if color is None:
                color = self.DEFAULT_COLOR
            if thickness is None:
                thickness = self.DEFAULT_THICKNESS

            updated_series.append((label, data_series, color, thickness))

        # Append the series and vertical lines specific to this subplot along with its title
        self.data_series.append((updated_series, include_extras, y_label, x_range, vertical_lines, title))

    def shade_regions(self,
                      start_x_idx: int,
                      end_x_idx: int,
                      alpha: float = 0.5,
                      text: Optional[str] = None,
                      position: Literal["top", "middle", "bottom"] = 'middle',
                      orientation: Literal["horizontal", "vertical"] = 'horizontal') -> None:
        """
        Shade a region in the plot
        :param start_x_idx: The start x index of the region to be shaded
        :param end_x_idx: The end x index of the region to be shaded
        :param alpha: The alpha value of the shaded region
        :param text: The text to be displayed in the middle of the shaded region
        :param position: The position of the text in the shaded region (top, middle, bottom)
        :param orientation: The orientation of the text in the shaded region (horizontal, vertical)
        :return: None
        """
        color = next(self.color_iterator)
        self.marked_areas.append((start_x_idx, end_x_idx, color, alpha, text, position, orientation))

    def add_table(self,
                  table_data: List[List[T]],
                  headers: Optional[List[str]] = None,
                  condition: Optional[Callable[[T], Optional[str]]] = None) -> None:
        """
        Add a table to the plot
        :param table_data: The data to be displayed in the table
        :param headers: A list of column headers
        :param condition: A function that takes a cell value and returns a color code
        :return: None
        """
        self.tables.append((table_data, headers, condition))

    def draw_vertical_line(self,
                           x_index: int,
                           color: str = "#8B0000",
                           thickness: float = 1.0,
                           label: Optional[str] = None,
                           label_position: Literal["top", "middle", "bottom"] = "top") -> None:
        """
        Draw a vertical line at the specified x index
        :param x_index: The x index where the vertical line should be drawn
        :param color: The color of the vertical line
        :param thickness: The thickness of the vertical line
        :param label: The label for the vertical line
        :param label_position: The position of the label (top, middle, bottom)
        :return: None
        """
        self.vertical_lines.append((x_index, color, thickness, label, label_position))

    def mark_point_on_x_axis(self, x_index: int, color: str = "red") -> None:
        """
        Mark a point on the x-axis with a rectangle
        :param x_index: The x index where the point should be marked
        :param color: The color of the rectangle
        :return: None
        """
        self.marked_points.append((x_index, color))

    def plot(self,
             fig_width: int = 15,
             fig_height_per_plot: int = 10,
             x_label: str = "",
             font_size: int = 24,
             save_path: Optional[str] = None,
             disable_show: bool = False,
             title_wrap_length: int = 50) -> plt.Figure:
        """
        Plot the data
        :param fig_width: Figure width.
        :param fig_height_per_plot: Figure height per plot.
        :param x_label: Label for the x-axis.
        :param font_size: The font size of the plot.
        :param save_path: Path to save the figure as a PNG file.
        :param disable_show: Boolean flag to disable plot showing.
        :param title_wrap_length: The maximum character length before wrapping the title.
        :return: The figure object.
        """
        plt.rcParams.update({'font.size': font_size})
        num_plots = len(self.data_series)
        total_plots = num_plots + (1 if self.tables else 0)
        num_rows = (total_plots + self.columns - 1) // self.columns  # Calculate rows needed based on columns

        fig, axs = plt.subplots(num_rows, self.columns, figsize=(fig_width, fig_height_per_plot * num_rows))
        axs = axs.flatten() if num_rows * self.columns > 1 else [axs]

        for ax, (series, include_extras, y_label, x_range, vertical_lines, subplot_title) in zip(axs[:num_plots],
                                                                                                 self.data_series):
            for label, data_series, color, thickness in series:
                ax.plot(x_range, data_series, label=label, color=color, linewidth=thickness)

            if include_extras:
                self._mark_areas(ax, x_range)
                self._mark_line(ax, x_range, vertical_lines)  # Pass vertical lines for the subplot
                self._color_x_axis(ax, series[0][1])

            # Wrap the title if it's too long
            wrapped_title = "\n".join(textwrap.wrap(subplot_title if subplot_title else self.title, title_wrap_length))
            ax.set_title(wrapped_title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            if self.y_axis_start_at_zero:
                ax.set_ylim(bottom=0)

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
            ax.grid(True, linestyle='--', alpha=0.7)

        self._handle_tables(axs)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format='png')

        if not disable_show:
            plt.show()

        return fig

    def _mark_areas(self, ax: plt.Axes, data_x: List[float]) -> None:
        """
        Mark shaded areas on the plot
        :param ax: The axis of the plot
        :return: None
        """
        for start_x_idx, end_x_idx, color, alpha, text, position, orientation in self.marked_areas:
            ax.axvspan(data_x[start_x_idx], data_x[end_x_idx], color=color, alpha=alpha)
            if text:
                x_text = (data_x[start_x_idx] + data_x[end_x_idx]) / 2
                y_min, y_max = ax.get_ylim()
                y_positions = {
                    'top': y_max - (y_max - y_min) * 0.1,
                    'middle': (y_max + y_min) / 2,
                    'bottom': y_min + (y_max - y_min) * 0.1
                }
                rotation_angle = 0 if orientation == 'horizontal' else 90
                ax.text(x_text, y_positions[position], text, ha='center', va='center', rotation=rotation_angle)

    def _mark_line(self, ax: plt.Axes, data_x: List[float], vertical_lines: Optional[
        List[Tuple[int, str, float, Optional[str], Literal["top", "middle", "bottom"]]]]) -> None:
        """
        Mark vertical lines on the plot specific to each subplot.
        :param ax: The axis of the plot.
        :param data_x: X-axis data for the plot.
        :param vertical_lines: The list of vertical lines specific to this subplot.
        :return: None.
        """
        if vertical_lines:
            for x_index, color, thickness, label, label_position in vertical_lines:
                ax.axvline(data_x[x_index], color=color, linewidth=thickness, ymax=5)
                if label:
                    y_min, y_max = ax.get_ylim()
                    y_positions = {
                        'top': y_max + (y_max - y_min) * 0.05,
                        'middle': (y_max + y_min) / 2,
                        'bottom': y_min - (y_max - y_min) * 0.05
                    }
                    ax.text(data_x[x_index], y_positions[label_position], label, ha='center', va='center')

    def _color_x_axis(self, ax: plt.Axes, y_data: List) -> None:
        """
        Mark points on the x-axis with rectangles below the plot
        :param ax: The axis of the plot
        :return: None
        """
        # Create custom x-axis
        ax.spines['bottom'].set_color('none')  # Hide the default x-axis
        for (xidx, color) in self.marked_points:
            segment = plt.Line2D((xidx, xidx + 1), (0, 0), color=color, linewidth=20)
            ax.add_line(segment)

    def _handle_tables(self, axs: List[plt.Axes]) -> None:
        """
        Add tables to the plot
        :param axs: The axes of the plt plot
        :return: None
        """
        if not self.tables:
            return

        ax_table = axs[-1]  # Use the last subplot for the table
        ax_table.axis('tight')
        ax_table.axis('off')

        for table_data, headers, condition in self.tables:
            table = ax_table.table(cellText=table_data, colLabels=headers, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            table.auto_set_column_width(col=list(range(len(headers))))  # Adjust column width based on content

            for i, row in enumerate(table_data):
                for j, cell in enumerate(row):
                    if condition:
                        color = condition(cell)
                        if color:
                            table[(i + 1, j)].set_facecolor(color)
