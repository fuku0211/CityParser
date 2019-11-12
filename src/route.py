import argparse
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm, trange
from operator import itemgetter
import numpy as np
import pyproj
import random
from geometry.shapes import Shape
from matplotlib.collections import PolyCollection


def _parse_gps_data(gpsdata):
    # TODO: 標高=楕円体高であってるかわからない
    lat, lon, dire, ht = map(float, itemgetter(3, 5, 8, 33)(gpsdata))
    # 欠損値に対する処理
    if lat < 0 or lon < 0:
        x, y = None, None
    # dddmm.mmmm表記になっているのを(度数+分数/60)でddd.dddd表記にする
    # http://lifelog.main.jp/wordpress/?p=146
    else:
        dd_lat, mm_lat = divmod(lat / 100, 1)
        dd_lon, mm_lon = divmod(lon / 100, 1)
        lat = dd_lat + mm_lat * 100 / 60
        lon = dd_lon + mm_lon * 100 / 60
        y, x = TRANSFORMER.transform(lat, lon)
    return (x, y, dire, ht)


def rotation(x, t):
    t = np.deg2rad(t)
    a = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ax = np.dot(a, x)
    return ax


def generate_random_color(n):
    color = [random.randint(0, 1) for _ in range(3)]
    return [color for i in range(n)]


class Mapbbox:
    def __init__(self):
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None

    def update(self, coord_x, coord_y):
        min_x = min(coord_x)
        min_y = min(coord_y)
        max_x = max(coord_x)
        max_y = max(coord_y)
        try:
            if min_x < self.min_x:
                self.min_x = min_x
            if min_y < self.min_y:
                self.min_y = min_y
            if max_x > self.max_x:
                self.max_x = max_x
            if max_y > self.max_y:
                self.max_y = max_y
        except TypeError:
            self.min_x = min_x
            self.min_y = min_y
            self.max_x = max_x
            self.max_y = max_y

    def offsetted(self, rate=0.1):
        offset_x = (self.max_x - self.min_x) * rate
        offset_y = (self.max_y - self.min_y) * rate
        return (
            self.min_x - offset_x,
            self.min_y - offset_y,
            self.max_x + offset_x,
            self.max_y + offset_y,
        )


def visualize_route(args):
    date_path = Path("data", "hdf5", args.site)
    shape_path = Path("data", "shp", args.site)
    gps_path = date_path / Path("gps.hdf5")
    site_shps = Shape(shape_path)

    fig = plt.figure()
    bbox = Mapbbox()
    ax = fig.add_subplot(1, 1, 1)
    # 敷地地図を描画
    print("drawing shapes")
    # for poly_categ in [site_shps.bldg]:
    #     coll = PolyCollection(poly_categ)
    #     ax.add_collection(coll)
    for line_categ in [site_shps.road]:
        for points in tqdm(line_categ):
            points = points.T
            ax.plot(points[0, :], points[1, :], color="g", lw=1)
            bbox.update(points[0, :], points[1, :])

    # 移動ルートを描画
    print("drawing routes")
    with h5py.File(str(gps_path), "r") as fg:
        for date in args.date:
            coord_x = []
            coord_y = []

            frame_count = len(fg[date].keys())
            for f in trange(frame_count, desc=f"{date}"):
                c_x, c_y, dire, ht = _parse_gps_data(fg[date][str(f)])
                if c_x is None and c_y is None:
                    continue
                coord_x.append(c_x)
                coord_y.append(c_y)
                dire = 90 - dire if dire <= 90 else 360 - (dire - 90)
                if args.num:
                    ax.text(c_x, c_y, str(f), fontsize=10)
            ax.scatter(
                coord_x,
                coord_y,
                s=10,
                c=generate_random_color(len(coord_x)),
                label=date,
            )
            bbox.update(coord_x, coord_y)

    bbox_outer = bbox.offsetted()
    ax.set_xlim([bbox_outer[0], bbox_outer[2]])
    ax.set_ylim([bbox_outer[1], bbox_outer[3]])
    ax.legend(loc="upper right")
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
    vis_parser.add_argument("--num", action="store_true")
    vis_parser.set_defaults(handler=visualize_route)

    args = parser.parse_args()

    TRANSFORMER = pyproj.Transformer.from_crs("EPSG:6668", "EPSG:30169")

    if hasattr(args, "handler"):
        args.handler(args)