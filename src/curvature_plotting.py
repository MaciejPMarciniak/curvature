import matplotlib.collections as lc
import matplotlib.lines as ln
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class PlottingCurvature:
    """Class for plotting curvature of a contour"""

    def __init__(
        self, contour_points_x: NDArray, contour_points_y: NDArray, curvature_values: NDArray
    ):
        self.x = contour_points_x
        self.y = contour_points_y
        self.curvature_values = curvature_values
        self.curvature_norm = plt.Normalize(-0.125, 0.125)
        self._center_contour()

    def plot_contour_with_curvature(self) -> None:
        """Creates a plot of the contour and the curvature."""
        fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 5]}, figsize=(13, 8))

        ax0 = self._generate_contour_ax(ax0)
        ax0.scatter(
            self.x, self.y, c=self.curvature_values, cmap="seismic", norm=self.curvature_norm
        )
        ax0.plot(self.x, self.y, c="gray", alpha=0.1, linewidth=5)

        ax1 = self._generate_curvature_ax(ax1)
        ax1.add_collection(self._get_curvature_plot())

        fig.tight_layout()
        plt.show()

    def _center_contour(self) -> None:
        self.x -= np.mean(self.x)
        self.y -= np.mean(self.y)

    def _generate_contour_ax(self, contour_ax: plt.Axes) -> plt.Axes:
        contour_ax.set_xlim(-45, 45)
        contour_ax.set_ylim(-45, 45)
        contour_ax.set_xlabel("Short axis")
        contour_ax.set_ylabel("Long axis")
        contour_ax.set_title("Endocardial contour")

        return contour_ax

    def _generate_curvature_ax(self, curvature_ax: plt.Axes) -> plt.Axes:
        """Generates an axis object with curvature related parameters and features"""
        curvature_ax.set_title("Mean geometric point-to-point curvature")
        curvature_ax.axhline(y=0, c="k", ls="-.", lw=2)
        curvature_ax.set_xlim(-1, len(self.curvature_values) + 1)
        curvature_ax.set_ylim(-0.07, 0.13)
        curvature_ax.set_xlabel("Point number")
        curvature_ax.set_ylabel("Curvature $[dm^{-1}]$")
        curvature_ax.legend(
            handles=[
                ln.Line2D([0], [0], c="b", lw=2, label="Negative curvature"),
                ln.Line2D([0], [0], c="r", lw=2, label="Positive curvature"),
            ],
            loc="upper right",
            title="Curvature",
        )
        return curvature_ax

    def _get_curvature_plot(self) -> lc.LineCollection:
        points = np.array([np.arange(len(self.curvature_values)), self.curvature_values]).T.reshape(
            1, 2
        )
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        curvature_plot = lc.LineCollection(segments, cmap="seismic", norm=self.curvature_norm)
        curvature_plot.set_array(self.curvature_values)
        curvature_plot.set_linewidth(5)
        return curvature_plot
