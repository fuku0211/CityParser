import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm, trange
from operator import itemgetter
import numpy as np
import pyproj

site_path = Path("data", "hdf5", "test")
gps_path = site_path / Path("gps.hdf5")

transformer = pyproj.Transformer.from_proj(6668, 6677)


def _parse_gps_data(gpsdata):
    # TODO: 標高=楕円体高であってるかわからない
    lat, lon, dire, ht = map(float, itemgetter(3, 5, 8, 33)(gpsdata))
    lat /= 100
    lon /= 100
    if lat < 0 or lon < 0:
        x, y = None, None
    else:
        y, x = transformer.transform(lat, lon)
    return (x, y, dire, ht)

n
def rotation(x, t):
    t = np.deg2rad(t)
    a = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    ax = np.dot(a, x)
    return ax


fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
coord_x = []
coord_y = []
arrow_x = []
arrow_y = []
heights = []
with h5py.File(str(gps_path), "r") as fg:
    frame_count = len(fg["20191111_121148"].keys())
    for f in trange(frame_count):
        c_x, c_y, dire, ht = _parse_gps_data(fg["20191111_121148"][str(f)])
        if c_x is None and c_y is None:
            continue
        coord_x.append(c_x)
        coord_y.append(c_y)
        heights.append(ht)

        dire = 90 - dire if dire <= 90 else 360 - (dire - 90)
        deg = np.deg2rad(dire)
        ax1.text(c_x, c_y, str(f))

        vec = rotation(np.array([0.2, 0]), dire)
        arrow_x.append(vec[0])
        arrow_y.append(vec[1])

ax1.scatter(coord_x, coord_y, s=1.5)
ax1.scatter(arrow_x, arrow_y, s=1.5, c='r')
ax1.quiver(coord_x, coord_y, arrow_x, arrow_y, units='xy', width=0.01)
ax1.set_xlim([min(coord_x), max(coord_x)])
ax1.set_ylim([min(coord_y), max(coord_y)])

ax2.plot(heights)
ax1.set_aspect('equal')

plt.show()