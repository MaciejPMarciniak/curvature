import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import time


class Curvature:
    """
    Class for computing curvature of ordered list of points on a plane
    """
    def __init__(self, line):

        self.line = np.array(line)
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
        # WARNING FOR NEGATIVE CURVATURE #
        # if menger_curvature < 0.0:
        #     warnings.warn('Negative curvature found with points: '
        #                   '\na: {}\t b: {}\t c: {}'.format(a, b, c))
        return -menger_curvature

    def calculate_curvature(self, gap=0):

        self.curvature = np.zeros(len(self.line) - 2)
        for local_, point in enumerate(range(len(self.curvature)-gap*2)):
            triplet = self.line[point:point+3+gap*2:gap+1, :]
            self.curvature[local_] = self._get_menger_curvature(*triplet)
        return self.curvature

    def plot_curvature(self):

        fig, _ = plt.subplots(figsize=(8, 7))
        _.plot(self.line[1:-1, 0], np.abs(self.curvature), 'r-', lw=2)
        _.set_title('Corresponding Menger\'s curvature'.format(len(self.curvature)))
        plt.show()
        fig.savefig(os.path.join('images', 'Curvature.png'))
        return _


class GradientCurvature:

    def __init__(self, trace, plot_derivatives=False):
        self.trace = trace
        self.plot_derivatives = plot_derivatives
        self.curvature = None

    def _get_gradients(self):
        self.x_trace = [x[0] for x in self.trace]
        self.y_trace = [y[1] for y in self.trace]

        x_prime = np.gradient(self.x_trace)
        y_prime = np.gradient(self.y_trace)
        x_bis = np.gradient(x_prime)
        y_bis = np.gradient(y_prime)

        if self.plot_derivatives:
            plt.subplot(211)
            plt.plot(x_prime)
            plt.plot(y_prime)
            plt.subplot(212)
            plt.plot(x_bis)
            plt.plot(y_bis)

        return x_prime, y_prime, x_bis, y_bis

    def calculate_curvature(self):
        x_prime, y_prime, x_bis, y_bis = self._get_gradients()
        curvature = x_prime * y_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2)) - \
            y_prime * x_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2))  # Numerical trick to get accurate values
        self.curvature = curvature
        return curvature


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


if __name__ == '__main__':

    k = 10000  # Resolution
    x = np.linspace(-5, 5, k+1)

    # ____Testing functions____
    # y = sigmoid(x)
    y = x ** 2
    # y = np.sin(x)
    # y = np.sqrt(x)

    xy = list(zip(x, y))  # list of points in 2D space

    curv1 = GradientCurvature(xy)
    start = time.time()
    curv1.calculate_curvature()
    end = time.time()
    print('Gradient curvature execution time: {}'.format(end-start))

    curv2 = Curvature(line=xy)
    start = time.time()
    curv2.calculate_curvature(gap=0)
    end = time.time()
    print('Menger curvature execution time: {}'.format(end - start))

    curv2.curvature = np.hstack((curv2.curvature[-1], curv2.curvature[:-1], ))
    curv2.plot_curvature()

    print(k)
    print('Menger')
    print('Maximum curvature: {}'.format(max(curv2.curvature)))
    print('Minimum curvature: {}'.format(min(curv2.curvature)))

    print('Gradient')
    print('Maximum curvature: {}'.format(np.max(curv1.curvature)))
    print('Minimum curvature: {}'.format(np.min(curv1.curvature)))

    plt.plot(-curv2.curvature, 'd-', label='Menger curvature')
    plt.plot(curv1.curvature, 'g.-', label='Gradient curvature')
    plt.legend()
    plt.show()
