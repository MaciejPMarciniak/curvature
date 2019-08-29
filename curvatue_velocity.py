import numpy as np
import matplotlib.pyplot as plt
import time
from curvature import Curvature

# Proof of agreement between Menger's and Cauchy's definitions of curvature.
# Perhaps vectorized implementation# could be faster?
# Agreement: abs(c_M - c_C) < 1e-12


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_line_equation(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def get_parameters_of_normal(m, xn, yn):
    k = -1/m
    b = yn - k * xn
    return k, b


def get_normal_intersection_point(k1, k2, b1, b2):
    xp = (b2 - b1) / (k1 - k2)
    yp1 = k1 * xp + b1
    yp2 = k2 * xp + b2
    assert np.abs(yp1 - yp2) < 0.0001, 'Wrong intersection calculation'
    return xp, yp1


def get_cauchy_curvature(xo, xp, yo, yp):
    r = np.linalg.norm([xo-xp, yo-yp])
    if xp < xo:
        r = -r
    c = 1/r
    return c


def calculate_cauchy_curvature_from_triplet(a, b, c):
    m1, c1 = get_line_equation([a[0], b[0]], [a[1], b[1]])
    m2, c2 = get_line_equation([b[0], c[0]], [b[1], c[1]])

    k1, b1 = get_parameters_of_normal(m1, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    k2, b2 = get_parameters_of_normal(m2, (c[0] + b[0]) / 2, (c[1] + b[1]) / 2)

    xi, yi = get_normal_intersection_point(k1, k2, b1, b2)

    curvature = get_cauchy_curvature(xi, b[0], yi, b[1])
    return curvature


a = np.linspace(-5, 5, 300)
b = sigmoid(a)
line = [(i, j) for i, j in zip(a, b)]

curv = Curvature(line=line)

start = time.time()
curv.calculate_curvature()
end = time.time()
print(end - start)

cauchy_curvature = []
start = time.time()
for i in range(1, 299):
    cauchy_curvature.append(calculate_cauchy_curvature_from_triplet(*line[i-1:i+2]))
end = time.time()
print(end - start)

xc = range(len(curv.curvature))
plt.plot(xc, curv.curvature)
plt.plot(xc, cauchy_curvature, 'r.')
plt.axhline(np.max(curv.curvature), linestyle='--')
plt.xlabel("$x$")
plt.ylabel("$Curvature value$")
plt.title("Curvature of the sigmoid function")
plt.show()
