import numpy as np
import warnings
import matplotlib.pyplot as plt


class Curvature:
    """
    Class for computing curvature of ordered list of points on a plane
    """
    def __init__(self, line):

        self.line = np.array(line)
        self.length = len(line)
        self.curvature = np.zeros(self.length-2)
        self.apices_per_frame = []

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
        return menger_curvature

    def calculate_curvature(self, gap=0):

        for local_, point in enumerate(range(len(self.curvature)-gap*2)):
            triplet = self.line[point:point+3+gap*2:gap+1, :]
            self.curvature[local_] = self._get_menger_curvature(*triplet)
        return self.curvature

    def plot_curvature(self):

        fig, _ = plt.subplots(figsize=(8, 7))
        _.plot(self.curvature, 'r-', lw=5)
        _.set_title('Curvature of {} points'.format(len(self.curvature)))
        fig.savefig('Curvature')
        return _


if __name__ == '__main__':

    x = np.linspace(-5, 5, 1001)
    y = (x ** 2)
    xy = list(zip(x, y))

    curv = Curvature(line=xy)
    curv.calculate_curvature()
    print(max(curv.curvature))
    curv.plot_curvature()
    curv.calculate_curvature(gap=1)
    print(max(curv.curvature))
    curv.plot_curvature()
    curv.calculate_curvature(gap=3)
    print(max(curv.curvature))
    curv.plot_curvature()
