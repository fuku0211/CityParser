import numpy as np
# import plotly.plotly as py
from plotly.graph_objs import *
from plotly import tools as tls
from plotly.offline import plot

pts = np.loadtxt("src/test/binary/data.txt")
x, y, z = zip(*pts)


points = Scatter3d(
    mode="markers", name="", x=x, y=y, z=z, marker=Marker(size=2, color="#458B00")
)

ALPHAHULL = 5
simplexes = Mesh3d(
    alphahull=ALPHAHULL,
    name="",
    x=x,
    y=y,
    z=z,
    opacity=1,
    color='green',  # set the color of simplexes in alpha shape
)
# simplexes = Mesh3d(
#     alphahull=10.0,
#     name="",
#     x=x,
#     y=y,
#     z=z,
#     color="90EE90",  # set the color of simplexes in alpha shape
#     opacity=0.15,
# )

x_style = dict(
    zeroline=False,
    range=[min(x), max(x)],
    tickvals=np.linspace(min(x), max(x), 5)[1:].round(1),
)
y_style = dict(
    zeroline=False,
    range=[min(y), max(y)],
    tickvals=np.linspace(min(y), max(y), 4)[1:].round(1),
)
z_style = dict(
    zeroline=False, range=[min(z), max(z)], tickvals=np.linspace(min(z), max(z), 5).round(1)
)

alpha = 1 / ALPHAHULL
layout = Layout(
    title=f"Alpha shape of a set of 3D points. Alpha={alpha}",
    width=1000,
    height=1000,
    scene=Scene(xaxis=x_style, yaxis=y_style, zaxis=z_style),
)

fig = Figure(data=Data([points, simplexes]), layout=layout)
# py.sign_in("empet", "smtbajoo93")
plot(fig)

