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

    @staticmethod
    def get_twice_triangle_area(a, b, c):

        if np.all(a == b) or np.all(b == c) or np.all(c == a):
            exit('At least two points are at the same position')

        twice_triangle_area = (b[0] - a[0])*(c[1] - a[1]) - (b[1]-a[1]) * (c[0]-a[0])

        if twice_triangle_area == 0:
            warnings.warn('Collinear consecutive points found: '
                          '\na: {}\t b: {}\t c: {}'.format(a, b, c))

        return twice_triangle_area

    def get_menger_curvature(self, a, b, c):

        menger_curvature = (2 * self.get_twice_triangle_area(a, b, c) /
                            (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)))
        if menger_curvature < 0.0:
            warnings.warn('Negative curvature found with points: '
                          '\na: {}\t b: {}\t c: {}'.format(a, b, c))
        return menger_curvature

    def calculate_curvature(self):

        for local_, point in enumerate(range(len(self.curvature))):
            triplet = self.line[point:point+3, :]
            self.curvature[local_] = self.get_menger_curvature(triplet[0], triplet[1], triplet[2])

        return self.curvature


if __name__ == '__main__':

    x = np.linspace(-5, 5, 1001)
    y = 3 * (x ** 2)
    xy = list(zip(x, y))

    curv = Curvature(line=xy)
    curv.calculate_curvature()
    print(max(curv.curvature))

    plt.plot(x, y)
    plt.plot(x[1:-1], curv.curvature)
    plt.show()

