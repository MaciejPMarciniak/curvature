from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import Rbf


class SmallValueError(ValueError):
    """Raised if the provided value is too small."""


class ContourAdjustment:
    """The class for interpolating and smoothing a 2D curve.

    Attributes:
        contour:
            The contour to be adjusted, with 2 coordinates per point.
        adj_contour:
            The result of the contour adjustment, same structure as contour.
        interpolation_resolution:
            The number of points that the adjusted contour will have after
            the interpolation.
        smoothing_factor:
            The impact of smoothing on the interpolated contour.
        interpolation_methods:
            The implemented methods that can be used for interpolating the
            contour.
    """

    def __init__(
        self, contour: NDArray, interpolation_resolution: int = 500, smoothing_factor: float = 20000
    ) -> None:
        self.contour = contour
        self.interpolation_resolution = interpolation_resolution
        self.smoothing_factor = smoothing_factor
        self.interpolation_methods: dict[str, Callable] = {"rbf": self._rbf}
        self.adj_contour: NDArray

        if self.smoothing_factor < 0:
            raise SmallValueError(
                f"Smoothing factor must be greater than 0 ({self.smoothing_factor} " "provided)"
            )
        if self.interpolation_resolution < len(self.contour):
            raise SmallValueError(
                "The resolution must be greater or equal the length of the "
                f"original contour ({self.interpolation_resolution=}, "
                f"{len(self.contour)=})"
            )

    def interpolate(self, interpolation_method: str = "rbf") -> None:
        """Calls the interpolation function to be applied on the contour.

        Args:
            interpolation_method: An interpolation method to be called.
        """
        if interpolation_method not in self.interpolation_methods:
            raise NotImplementedError(
                f"Method {interpolation_method} is not available. Implemented methods: "
                f"{self.interpolation_methods.keys()}"
            )
        self.interpolation_methods[interpolation_method]()

    def _rbf(self) -> None:
        """Radial basis function interpolation.

        The 'quintic' function optimizes the interpolation value as r**5 where r is the
        distance from the next point. Such interpolation ensures smooth higher derivatives.

        Set smoothing to a very high value (>10000) if the contour is quantized (e.g. coming from
        a border of a low resolution image).
        """
        # Split contour to 1D values
        x, y = self.contour[:, 0], self.contour[:, 1]

        # Calculate the interpolation function parameters
        positions = np.arange(len(x))
        rbf_x = Rbf(positions, x, smooth=self.smoothing_factor, function="quintic")
        rbf_y = Rbf(positions, y, smooth=self.smoothing_factor, function="quintic")

        # Define the new resolution for the adjusted contour
        interpolation_base = np.linspace(
            0, len(self.contour) - 1, self.interpolation_resolution + 1
        )

        # Generate the adjusted contour
        x_interpolated = rbf_x(interpolation_base)
        y_interpolated = rbf_y(interpolation_base)

        self.adj_contour = np.column_stack([x_interpolated, y_interpolated])
