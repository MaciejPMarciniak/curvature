# Curvature measurement
In mathematics, the Menger curvature of three points in n-dimensional Euclidean space is the reciprocal of the 
radius of the circle that passes through the three points.  Intuitively, curvature is the amount by which a geometric 
object deviates from being:
* a **flat plane** in case of the *surface*, 
* a **straight line** in case of the *curve*.


In this project, the curvature measurement is applied to human left ventricle traces, in order to determine the 
occurrence of the left ventricular basal septal hypertrophy from 2 dimensional echocardiography (2D echo). The described
methods can be applied to any traces of the left ventricle. 

# Motivation
 Localized basal septal hypertrophy (BSH) is a known marker of a more advanced impact of afterload on cardiac function 
 in hypertension. There is variability in criteria used for defining BSH, mainly based on ratios of multiple septal wall
 thickness measurements with high inter-observer variability. The aim is to investigate septal curvature as a novel, 
 semiautomated method for better recognition of patients with BSH. 

# Screenshots
Examples of differences in septal curvature among 3 patients: healthy, hypertensive and hypertensive with septal bulge:
![curvature examples](images/Curvature_healthy_htn_bsh.png "Curvature differences among patient groups")

# How2use
### curvature.Curvature

**Call**
```text
Curvature(line=[(x1, y1), (x2, y2), (x3, y3) ... (xn, yn)] 
```

**Input**

Input argument *line* is a list of tuples. Each tuple contains 2 values, i.e. X and Y position on the 2D plane. 

**Output** 

Numpy array. Menger's curvature value for each tuple in the input list, except first and the last one. 

---
**Methods** 
```text
Curvature.calculate_curvature(gap=0)
```

Calculates the curvature of the line. It is defined as the reciprocal of the radius of a circle intersecting three 
points in 2D space.
Optional parameter *gap* sets the number of points away from the processed point, based on which the curvature is calculated:
* if gap = 0, three consecutive points are used,
* if gap = 1, for point number 2, points 0 and 4 are used, for point number 3, points 1 and 5 are used and so on, 
* if gap = 2, for point number 3, points 0 and 6 are used, for point number 4, points 1 and 7 are used and so on. 

It has been included as a smoothing option, with the trade-off on information loss.

---
```text
Curvature.plot_curvature()
```

Plots the curvature values as a line plot. 

---
**Example (quadratic parabola)**
```python
import numpy as np
from curvature import Curvature

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
```
Results:
```text
Curvature values (first 10 points): [0.00198212 0.00199397 0.00200591 0.00201794 0.00203007 0.0020423
 0.00205463 0.00206705 0.00207958 0.00209221]
 
Curvature values (10 middle points): [1.98075815 1.98905163 1.99501104 1.99860098 1.99980002 1.99860098
 1.99501104 1.98905163 1.98075815 1.97017947]
 
Maximum curvature: 1.9998000199980006
Minimum curvature: 0.0019821241706415283
```
![parabola](images/Parabola.png "Parabola, 1001 points")
![curvature](images/Curvature.png "Menger's curvature")

### class Cohort (bsh.py)
The aim of this class is to calculate the curvature of the trace of left ventricle and find indices that are useful for
classification of the basal septal hypertrophy in hypertensive patients. The indices include *maximum* and *minimum* 
curvature, the *changes in curvature over the cycle* and *interactions* between them. The derived indices are useful for the 
statistical analysis, and show potential to unveil the basal septal hypertrophy setting in a robust, unbiased way.

**Call**

class Cohort(source_path='path_to_data', view='4C', output_path='path_to_store_results', 
output='name_of_output_file.csv') 

**Input**

*source_path*: path to the .csv files containing the myocardial trace obtained with speckle tracing in EchoPAC. 
 

*view*: the view in which the image was taken; 4-chamber ('4C'), 3-chamber ('3C'), or 2-chamber ('2C') 

 
*output_path*: path to a folder where the results of computation and plots will be stored 

 
**Output** 

1 Tables:

* File names in the input directory with corresponding IDs of cases,

* Curvature values of the trace changing in the cardiac cycle,

* All derived indices calculated in available views,

* Simple statistical values of the derived indices,

* Lists of most prevalent cases, in terms of the derived indices. 

2 Plots:

* Inidividual plots of the trace and the curvature throughout the cardiac cycle,  

* Distributions of the derived indices in the population. 

---
**Methods**
```text
Cohort.print_names_and_ids(to_file=False, views=('4C', '3C', '2C')) 
```

Creates (or prints) the table with names of the files and corresponding IDs. Useful when the clinician provides tables with different IDs, unrelated to one another. 

*to_file* controls whether the table is saved to file, or is printed in the console. 

*views* list is used to choose the relevant views to print out. The function prints the names and IDs for all views by default. 

_Example:_ 
```text
Cohort.print_names_and_ids(views=['4C'])
```
|  |  |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|
| AAAC0130_4C | BBB0460_4C | CCC0043_4C |  DD_4C | EE_4C | ...
| AAAC0130 | BBB0460 | CCC0043 | X 7323260121 | aiouey11022017 | ...

--- 
```text
Cohort.save_curvatures() 
```

Saves the curvature of individual trace over 1 cycle. Rows denote the frames and columns are the separate points of the trace. 

_Example:_ 
```text
Cohort.save_curvatures() 
```

In *~/output_curvature/curvatures/ABCDE0123.csv*:

Frames/trace points | 0 | 1 | 2 | 3 | 4 | ...
 :---:|:---:|:---:| :---: | :---: | :---: | :---:
  0 | -0.0004368951 | -0.0008411005 | 9.81975201759697E-05 | -0.0023831521 | -0.0045366323 | ...
  1 | 0.0004937481 | -0.0003834384 | -0.0003401089 | -0.0018914279 | -0.0039957284 | ... 
  2 | 0.0005044135 | -0.001319833 | 0.0011134577 | -0.0028624835 | -0.0044718255 | ...
  3 | 0.0005097837 | 0.0006038951 | -0.0013404811 | -0.0013120823 | -0.0049921892 | ...
  4 | 9.36306817606534E-05 | -0.0008662325 | 0.0001828868 | -0.0033066163 | -0.0034790443 | ... 
 ... | ... | ... | ... | ... | ... | ...

--- 
```text
Cohort.get_statistics() 
```

Builds a table with means and standard deviations of the derived indices for different labelled cohorts. It is useful for quick hypothesis testing. 

Example: table with statistics

--- 
```text
Cohort.get_extemes(n=30) 
```

Creates a table with IDs of cases with most prevalent indices and interactions. It is useful for the analyst to decide on which indices are relevant for the classification. 

*n* is the number of cases to print for each index.

Example: table with IDs.  

--- 
```text
Cohort.plot_curvatures(coloring_scheme='curvature', plot_mean=False) 
```

Plots the with traces in a given view and the curvature of each points in the trace in each frame. The traces can be coloured according to the value of the curvature, or the frame number. This function also creates heatmaps showing the curvature of the trace in the given view changing in time.  

Example: plot of traces and the heatmap 

--- 
```text
Cohort.plot_distributions(plot_data=False, plot_master=False, table_name=None) 
```

Plots the distributions of the derived indices, for univariate and bivariate exploratory data analysis. 

Examples: Univariate and bivariate plots 


**Full example**

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

# Credits
Abstract

# License