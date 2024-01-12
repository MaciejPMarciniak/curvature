import os
import sys
import time
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import Rbf, interp1d, pchip_interpolate


class Curvature:
    """
    Class for computing curvature of ordered list of points on a plane
    """

    def __init__(self, trace: list[NDArray], interpolation_function: Callable) -> None:
        self.trace = np.array(trace)
        self.interpolation_function = interpolation_function
        self.curvature: NDArray

    @staticmethod
    def _get_twice_triangle_area(a: NDArray, b: NDArray, c: NDArray) -> float:
        """Calculates the doubled triangle area from a set of 2D points.

        Args:
            a: 2D point
            b: 2D point
            c: 2D point

        Returns:
            Doubled triangle area.
        """
        if np.all(a == b) or np.all(b == c) or np.all(c == a):
            sys.exit("CURVATURE:\nAt least two points are at the same position")

        twice_triangle_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        if twice_triangle_area == 0:
            warnings.warn(f"Collinear consecutive points found: \na: {a}\t b: {b}\t c: {c}")

        return twice_triangle_area

    def _get_menger_curvature(self, a: NDArray, b: NDArray, c: NDArray) -> float:
        menger_curvature = (
            2
            * self._get_twice_triangle_area(a, b, c)
            / (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a))
        )
        return menger_curvature

    def calculate_curvature(self, interpolation_target_n: int = 500) -> NDArray:
        self.trace = interpolate_trace(
            self.trace, self.interpolation_function, interpolation_target_n
        )

        self.curvature = np.zeros(len(self.trace) - 2)
        for point_index in range(len(self.curvature)):
            triplet = self.trace[point_index : point_index + 3]
            self.curvature[point_index - 1] = self._get_menger_curvature(*triplet)
        return self.curvature

    def plot_curvature(self) -> None:
        fig, _ = plt.subplots(figsize=(8, 7))
        _.plot(self.trace[1:-1, 0], self.curvature, "r-", lw=2)
        _.set_title(f"Corresponding Menger's curvature: {len(self.curvature)}")
        plt.show()
        fig.savefig(os.path.join("images", "Curvature.png"))
        return _


class GradientCurvature:
    """Class for calculating the gradient curvature"""

    def __init__(
        self, trace: list[NDArray], interpolation_function: Callable, plot_derivatives: bool = True
    ) -> None:
        self.trace = trace
        self.plot_derivatives = plot_derivatives
        self.interpolation_function = interpolation_function
        self.curvature = None
        self.x_trace: NDArray
        self.y_trace: NDArray

    def _get_gradients(self) -> tuple[NDArray, ...]:
        """Calculates the gradients of the provided trace in 2 dimensions.

        Both the first and second derivatives are calculated.

        Returns:
            The gradients of the trace.
        """
        self.x_trace = [x[0] for x in self.trace]
        self.y_trace = [y[1] for y in self.trace]

        x_prime = np.gradient(self.x_trace)
        y_prime = np.gradient(self.y_trace)
        x_bis = np.gradient(x_prime)
        y_bis = np.gradient(y_prime)

        if self.plot_derivatives:
            plt.subplot(411)
            plt.plot(self.y_trace, label="y")
            plt.title("Function")

            plt.subplot(412)
            # plt.plot(x_prime, label='x\'')
            plt.plot(y_prime, label="y'")
            plt.title("First spatial derivative")
            plt.legend()
            plt.subplot(413)
            # plt.plot(x_bis, label='x\'\'')
            plt.plot(y_bis, label="y''")
            plt.title("Second spatial derivative")
            plt.legend()

        return x_prime, y_prime, x_bis, y_bis

    def calculate_curvature(self, interpolation_target_n: int = 500) -> NDArray:
        """Calculates the curvature of an interpolated trace using the gradients.

        Interpolation is used to smooth out the first and second derivatives of the trace.

        Args:
            interpolation_target_n: The optimal number of points to calculate the curvature.
            Defaults to 500.

        Returns:
            Gradient curvature of the trace.
        """
        self.trace = interpolate_trace(
            self.trace, self.interpolation_function, interpolation_target_n
        )
        x_prime, y_prime, x_bis, y_bis = self._get_gradients()
        curvature = x_prime * y_bis / (
            (x_prime**2 + y_prime**2) ** (3 / 2)
        ) - y_prime * x_bis / (
            (x_prime**2 + y_prime**2) ** (3 / 2)
        )  # Numerical trick to get accurate values
        self.curvature = curvature
        return curvature


def sigmoid(t: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-t))


def interpolate_trace(
    trace: list[NDArray], interpolation_function: Callable, target_n: int = 500
) -> NDArray:
    """Interpolates the trace with provided function to desired number of points.

    Args:
        trace:
            2D curve.
        interpolation_function:
            A predefined method for interpolating the trace.
        target_n:
            The optimal number of points to calculate the curvature.
            Defaults to 500.

    Returns:
        The interpolated trace.
    """
    n_trace_points = len(trace)
    x_points = [x[0] for x in trace]
    y_points = [y[1] for y in trace]

    positions = np.arange(n_trace_points)  # strictly monotonic, number of points in single trace
    interpolation_base = np.linspace(0, n_trace_points - 1, target_n + 1)

    x_interpolated, y_interpolated = interpolation_function(
        x_points, y_points, positions, interpolation_base
    )

    interpolated_trace = np.array([[x, y] for x, y in zip(x_interpolated, y_interpolated)])

    return interpolated_trace


def rbf_interpolation(
    x: NDArray, y: NDArray, positions: NDArray, interpolation_base: NDArray
) -> tuple[NDArray, ...]:
    """Radial basis function interpolation.

    The 'quintic' function optimizes the interpolation value as r**5 where r is the
    distance from the next point. Such interpolation ensures smooth higher derivatives.

    Args:
        x:
            Horizontal coordinates.
        y:
            Vertical coordinates.
        positions:
            Ordered numbers used for interpolation model generation.
        interpolation_base:
            The desired interpolation density.

    Returns:
        Coordinates interpolated in both dimensions.
    """

    rbf_x = Rbf(positions, x, smooth=len(positions), function="quintic")
    rbf_y = Rbf(positions, y, smooth=len(positions), function="quintic")

    x_interpolated = rbf_x(interpolation_base)
    y_interpolated = rbf_y(interpolation_base)

    return x_interpolated, y_interpolated


def interp1d_interpolation(
    x: NDArray,
    y: NDArray,
    positions: NDArray,
    interpolation_base: NDArray,
    interpolation_kind: str = "cubic",
) -> tuple[NDArray, ...]:
    """Polynomial interpolation.
    Args:
        x:
            Horizontal coordinates.
        y:
            Vertical coordinates.
        positions:
            Ordered numbers used for interpolation model generation.
        interpolation_base:
            The desired interpolation density.
        interpolation_kind:
            The degree of the interpolation function.

    Returns:
        Coordinates interpolated in both dimensions.
    """
    interp1d_x = interp1d(positions, x, kind=interpolation_kind)
    interp1d_y = interp1d(positions, y, kind=interpolation_kind)

    x_interpolated = interp1d_x(interpolation_base)
    y_interpolated = interp1d_y(interpolation_base)

    return x_interpolated, y_interpolated


def pchip_interpolation(
    x: NDArray,
    y: NDArray,
    positions: NDArray,
    interpolation_base: NDArray,
) -> tuple[NDArray, ...]:
    """Cubic Hermite interpolation.

    Args:
        x:
            Horizontal coordinates.
        y:
            Vertical coordinates.
        positions:
            Ordered numbers used for interpolation model generation.
        interpolation_base:
            The desired interpolation density.
        interpolation_kind:
            The degree of the interpolation function.

    Returns:
        Coordinates interpolated in both dimensions.
    """
    pchip_x = pchip_interpolate(positions, x, interpolation_base)
    pchip_y = pchip_interpolate(positions, y, interpolation_base)

    return pchip_x, pchip_y


if __name__ == "__main__":
    k = 20  # Resolution
    independent = np.linspace(-5, 5, k + 1)

    # ____Testing functions____
    # y = sigmoid(x**3)
    dependent = independent**2
    # y = np.sin(x)
    # y = (np.sin(x**2))

    ab = list(zip(independent, dependent))  # list of points in 2D space

    plt.scatter(independent, dependent)
    plt.show()
    ifunc = rbf_interpolation
    curv1 = GradientCurvature(trace=ab, interpolation_function=ifunc)
    start = time.time()
    curv1.calculate_curvature()
    end = time.time()
    print(f"Gradient curvature execution time: {end - start}")

    curv2 = Curvature(trace=ab, interpolation_function=ifunc)
    start = time.time()
    curv2.calculate_curvature()
    end = time.time()
    print(f"Menger curvature execution time: {end - start}")

    print(k)
    print("Menger")
    print(f"Maximum curvature: {max(curv2.curvature)}")
    print(f"Minimum curvature: {min(curv2.curvature)}")

    print("Gradient")
    print(f"Maximum curvature: {np.max(curv1.curvature)}")
    print(f"Minimum curvature: {np.min(curv1.curvature)}")

    plt.subplot(414)
    plt.plot(range(2, len(curv2.curvature) + 2), curv2.curvature, "d-", label="Menger curvature")
    plt.plot(curv1.curvature, "g.-", label="Gradient curvature")
    plt.legend()
    plt.show()
