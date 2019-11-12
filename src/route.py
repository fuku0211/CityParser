import argparse
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm, trange
from operator import itemgetter
import numpy as np
import pyproj
import random


def _parse_gps_data(gpsdata):
    # TODO: 標高=楕円体高であってるかわからない
    lat, lon, dire, ht = map(float, itemgetter(3, 5, 8, 33)(gpsdata))
    lat /= 100
    lon /= 100
    if lat < 0 or lon < 0:
        x, y = None, None
    else:
        y, x = TRANSFORMER.transform(lat, lon)
    return (x, y, dire, ht)


def rotation(x, t):
    t = np.deg2rad(t)
    a = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ax = np.dot(a, x)
    return ax


def generate_random_color():
    return [random.randint(0, 1) for _ in range(3)]


class SizeObj():
    def __init__(self):
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0

    def update(self, coord_x, coord_y):
        self.min_x = min(coord_x)
        self.min_y = min(coord_y)
        self.max_x = max(coord_x)
        self.max_y = max(coord_y)


def visualize_route(args):
    site_path = Path("data", "hdf5", args.site)
    gps_path = site_path / Path("gps.hdf5")

    fig = plt.figure()
    figsize = SizeObj()
    ax = fig.add_subplot(1, 1, 1)
    with h5py.File(str(gps_path), "r") as fg:
        for date in args.date:
            coord_x = []
            coord_y = []

            frame_count = len(fg[date].keys())
            for f in trange(frame_count):
                c_x, c_y, dire, ht = _parse_gps_data(fg[date][str(f)])
                if c_x is None and c_y is None:
                    continue
                coord_x.append(c_x)
                coord_y.append(c_y)
                dire = 90 - dire if dire <= 90 else 360 - (dire - 90)
                ax.text(c_x, c_y, str(f), fontsize=10)
            ax.scatter(coord_x, coord_y, s=10, c=generate_random_color(), label=date)
            figsize.update(coord_x, coord_y)

    ax.set_xlim([figsize.min_x, figsize.max_x])
    ax.set_ylim([figsize.min_y, figsize.max_y])
    ax.legend(loc='upper right')
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)
    parent_parser.add_argument("-d", "--date", required=True, nargs="*")

    # 各コマンドの設定
    vis_parser = subparsers.add_parser("vis", parents=[parent_parser])
    vis_parser.set_defaults(handler=visualize_route)

    args = parser.parse_args()

    TRANSFORMER = pyproj.Transformer.from_proj(6668, 6677)
    if hasattr(args, "handler"):
        args.handler(args)
