from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

w = h = 360

n = 6
np.random.seed(0)
pts = np.array([[172, 47], [117, 192], [323, 251], [195, 359], [9, 211], [277, 242]])

vor = Voronoi(pts)

print(type(vor))
# <class 'scipy.spatial.qhull.Voronoi'>

fig = voronoi_plot_2d(vor)
fig.savefig("src/test/scipy_matplotlib_voronoi.png")

