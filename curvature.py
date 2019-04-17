import numpy as np
import os
import warnings
import matplotlib.pyplot as plt


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
            exit('At least two points are at the same position')

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
        _.plot(self.line[1:-1, 0], self.curvature, 'r-', lw=2)
        _.set_title('Corresponding Menger\'s curvature'.format(len(self.curvature)))
        fig.savefig(os.path.join('images', 'Curvature.png'))
        return _


if __name__ == '__main__':

    x = np.linspace(-5, 5, 1001)
    y = (x ** 2)
    xy = list(zip(x, y))  # list of points in 2D space

    curv = Curvature(line=xy)
    curv.calculate_curvature(gap=0)

    print('Curvature values (first 10 points): {}'.format(curv.curvature[:10]))
    print('Curvature values (10 middle points): {}'.format(curv.curvature[int(len(x)/2-5):int(len(x)/2+5)]))
    print('Maximum curvature: {}'.format(max(curv.curvature)))
    print('Minimum curvature: {}'.format(min(curv.curvature)))

    curv.plot_curvature()
