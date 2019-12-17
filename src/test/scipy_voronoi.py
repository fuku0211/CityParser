from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

w = h = 360

n = 6
np.random.seed(0)
pts = np.array([[172, 47], [117, 192], [323, 251], [195, 359], [9, 211], [277, 242]])


vor = Voronoi(pts)

fig = voronoi_plot_2d(vor)

for i in range(pts.shape[0]):
    fig.axes[0].text(pts[i, 0], pts[i, 1], f"{i}")
for i in range(vor.vertices.shape[0]):
    fig.axes[0].text(vor.vertices[i, 0], vor.vertices[i, 1], f"{i}")
fig.axes[0].set_aspect("equal")
fig.savefig("src/test/scipy_matplotlib_voronoi.png")

