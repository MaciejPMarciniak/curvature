# Curvature measurement
In mathematics, the Menger curvature of a triple of points in n-dimensional Euclidean space is the reciprocal of the 
radius of the circle that passes through the three points.  Intuitively, curvature is the amount by which a geometric 
object such as a *surface* deviates from being a **flat plane**, or a *curve* from being a **straight line**.

Application to left ventricle trace

# Motivation
HTN
BSH

# Screenshots
Abstract

# How2use
### class Curvature (curvature.py)
```
import numpy as np
from curvature import Curvature

x = np.linspace(-5, 5, 1001)
y = (x ** 2)
xy = list(zip(x, y))  # list of points in 2D space

curv = Curvature(line=xy)
curv.calculate_curvature(gap=0)

print(curv.curvature[:10])
print(max(curv.curvature))
print(min(curv.curvature))

curv.plot_curvature()
```
Code example

### class Cohort (ventricle.py)
### class PlottingDistributions (plotting.py)
### class PlottingCurvature (plotting.py)

# Credits
Abstract

# License