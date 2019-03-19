# Curvature measurement
In mathematics, the Menger curvature of a triple of points in n-dimensional Euclidean space is the reciprocal of the 
radius of the circle that passes through the three points.  Intuitively, curvature is the amount by which a geometric 
object deviates from being a:
* **flat plane** in case of the *surface*, 
* **straight line** in case of the *curve*.


In this project, the curvature measurement is applied to human left ventricle traces, in order to determine the 
occurrence of the left ventricular basal septal hypertrophy from 2 dimensional echocardiography (2D echo). The described
methods can be applied to any traces of the left ventricle. 

# Motivation
HTN

BSH

# Screenshots
Abstract

# How2use
### class Curvature (curvature.py)
class Curvature(line=[(x1, y1), (x2, y2), (x3, y3) ... (xn, yn)] 

**Input argument** 

List of tuples. Each tuple contains 2 values, i.e. X and Y position on the 2D plane. 

**Output** 

Numpy array. Menger's curvature value for each tuple in the input list, except first and the last one. 

**Methods:** 

*Curvature.calculate_curvature(gap=0)* 

Optional parameter gap sets the number of points away from the processed point, based on which the curvature is calculated. I.e. 

* if gap = 0, three consecutive points are used 
* if gap = 1, for point number 2, points 0 and 4 are used, for point number 3, points 1 and 5 are used and so on. 
* if gap = 2, for point number 3, points 0 and 6 are used, for point number 4, points 1 and 7 are used and so on. 

It has been included as a smoothing option, with the trade-off on information loss. 

*Curvature.plot_curvature()* 

Plots the curvature values as a line plot. 

 

Example
```Curvautre usage example
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
### class Cohort (ventricle.py)
Cohort use description
```Cohort usage for multiple curvature calculations
import os
from ventricle import Cohort

source = os.path.join('/home/mat/Python/data/curvature')
target_path = os.path.join('/home/mat/Python/data/curvature/')

cohort = Cohort(source_path=source, view=_view, output_path=target_path)

cohort.get_extemes(32)
cohort.plot_curvatures('asf')
cohort.plot_curvatures(coloring_scheme='curvature')
cohort.save_curvatures()
cohort.plot_distributions(plot_data=True, table_name='_all_cases_with_labels.csv')
cohort.print_names_and_ids(to_file=True)
```
### class PlottingDistributions (plotting.py)
### class PlottingCurvature (plotting.py)

# Credits
Abstract

# License