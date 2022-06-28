import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import time
from scipy.interpolate import Rbf, interp1d, pchip_interpolate


class Curvature:
    """
    Class for computing curvature of ordered list of points on a plane
    """
    def __init__(self, trace, interpolation_function):

        self.trace = np.array(trace)
        self.interpolation_function = interpolation_function
        self.curvature = None

    @staticmethod
    def _get_twice_triangle_area(a, b, c):

        if np.all(a == b) or np.all(b == c) or np.all(c == a):
            exit('CURVATURE:\nAt least two points are at the same position')

        twice_triangle_area = (b[0] - a[0])*(c[1] - a[1]) - (b[1]-a[1]) * (c[0]-a[0])

        if twice_triangle_area == 0:
            warnings.warn('Collinear consecutive points found: '
                          '\na: {}\t b: {}\t c: {}'.format(a, b, c))

        return twice_triangle_area

    def _get_menger_curvature(self, a, b, c):

        menger_curvature = (2 * self._get_twice_triangle_area(a, b, c) /
                            (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)))
        return menger_curvature

    def calculate_curvature(self, interpolation_target_n=500):
        self.trace = interpolate_trace(self.trace, self.interpolation_function, interpolation_target_n)

        self.curvature = np.zeros(len(self.trace) - 2)
        for point_index in range(len(self.curvature)):
            triplet = self.trace[point_index:point_index+3]
            self.curvature[point_index-1] = self._get_menger_curvature(*triplet)
        return self.curvature

    def plot_curvature(self):
        fig, _ = plt.subplots(figsize=(8, 7))
        _.plot(self.trace[1:-1, 0], self.curvature, 'r-', lw=2)
        _.set_title('Corresponding Menger\'s curvature'.format(len(self.curvature)))
        plt.show()
        fig.savefig(os.path.join('images', 'Curvature.png'))
        return _


class GradientCurvature:

    def __init__(self, trace, interpolation_function, plot_derivatives=True):
        self.trace = trace
        self.plot_derivatives = plot_derivatives
        self.interpolation_function = interpolation_function
        self.curvature = None

    def _get_gradients(self):
        self.x_trace = [x[0] for x in self.trace]
        self.y_trace = [y[1] for y in self.trace]

        x_prime = np.gradient(self.x_trace)
        y_prime = np.gradient(self.y_trace)
        x_bis = np.gradient(x_prime)
        y_bis = np.gradient(y_prime)

        if self.plot_derivatives:
            plt.subplot(411)
            plt.plot(self.y_trace, label='y')
            plt.title('Function')

            plt.subplot(412)
            # plt.plot(x_prime, label='x\'')
            plt.plot(y_prime, label='y\'')
            plt.title('First spatial derivative')
            plt.legend()
            plt.subplot(413)
            # plt.plot(x_bis, label='x\'\'')
            plt.plot(y_bis, label='y\'\'')
            plt.title('Second spatial derivative')
            plt.legend()

        return x_prime, y_prime, x_bis, y_bis

    def calculate_curvature(self, interpolation_target_n=500):

        self.trace = interpolate_trace(self.trace, self.interpolation_function, interpolation_target_n)
        x_prime, y_prime, x_bis, y_bis = self._get_gradients()
        curvature = x_prime * y_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2)) - \
            y_prime * x_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2))  # Numerical trick to get accurate values
        self.curvature = curvature
        return curvature


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def interpolate_trace(trace, interpolation_function, target_n=500):

    n_trace_points = len(trace)
    x_points = [_x[0] for _x in trace]
    y_points = [_y[1] for _y in trace]

    positions = np.arange(n_trace_points)  # strictly monotonic, number of points in single trace
    interpolation_base = np.linspace(0, n_trace_points-1, target_n+1)

    x_interpolated, y_interpolated = interpolation_function(x_points, y_points, positions, interpolation_base)

    interpolated_trace = np.array([[x, y] for x, y in zip(x_interpolated, y_interpolated)])

    return interpolated_trace


def rbf_interpolation(x, y, positions, interpolation_base):

    # Radial basis function interpolation 'quintic': r**5 where r is the distance from the next point
    # Smoothing is set to length of the input data

    rbf_x = Rbf(positions, x, smooth=len(positions), function='quintic')
    rbf_y = Rbf(positions, y, smooth=len(positions), function='quintic')

    print('Interpolation')
    print(len(positions))
    print(rbf_x.norm)
    print('------------')
    # Interpolate based on the RBF model
    x_interpolated = rbf_x(interpolation_base)
    y_interpolated = rbf_y(interpolation_base)

    return x_interpolated, y_interpolated


def interp1d_interpolation(x, y, positions, interpolation_base, interpolation_kind='cubic'):

    interp1d_x = interp1d(positions, x, kind=interpolation_kind)
    interp1d_y = interp1d(positions, y, kind=interpolation_kind)

    x_interpolated = interp1d_x(interpolation_base)
    y_interpolated = interp1d_y(interpolation_base)

    return x_interpolated, y_interpolated


def pchip_interpolation(x, y, positions, interpolation_base):

    print(positions)
    print(interpolation_base)

    pchip_x = pchip_interpolate(positions, x, interpolation_base)
    pchip_y = pchip_interpolate(positions, y, interpolation_base)

    return pchip_x, pchip_y


if __name__ == '__main__':

    k = 20  # Resolution
    x = np.linspace(-5, 5, k+1)

    # ____Testing functions____
    # y = sigmoid(x**3)
    y = x ** 2
    # y = np.sin(x)
    # y = (np.sin(x**2))

    # contour = pd.read_csv(r'G:\Curvature\ProjectCurvature_18_02_2020\Analysis\EndoContours\2DS120_AARO0441_ANDREU AGULLO_21_09_2018_4CH_FULL_TRACE_ENDO_V1_D1_B.CSV', skiprows=17)
    # x = contour.iloc[0, :-1:2]
    # y = contour.iloc[0, 1:-1:2]
    print(x[-1])
    print(y[-1])
    xy = list(zip(x, y))  # list of points in 2D space

    plt.scatter(x, y)
    plt.show()
    ifunc = rbf_interpolation
    curv1 = GradientCurvature(trace=xy, interpolation_function=ifunc)
    start = time.time()
    curv1.calculate_curvature()
    end = time.time()
    print('Gradient curvature execution time: {}'.format(end-start))

    curv2 = Curvature(trace=xy, interpolation_function=ifunc)
    start = time.time()
    curv2.calculate_curvature()
    end = time.time()
    print('Menger curvature execution time: {}'.format(end - start))

    print(k)
    print('Menger')
    print('Maximum curvature: {}'.format(max(curv2.curvature)))
    print('Minimum curvature: {}'.format(min(curv2.curvature)))

    print('Gradient')
    print('Maximum curvature: {}'.format(np.max(curv1.curvature)))
    print('Minimum curvature: {}'.format(np.min(curv1.curvature)))

    plt.subplot(414)
    plt.plot(range(2, len(curv2.curvature)+2), curv2.curvature, 'd-', label='Menger curvature')
    plt.plot(curv1.curvature, 'g.-', label='Gradient curvature')
    plt.legend()
    plt.show()
