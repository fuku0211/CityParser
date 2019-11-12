import argparse
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm, trange
import random
from geometry.shapes import Shape
from matplotlib.collections import PolyCollection
from utils.tool import parse_gps_data


def _create_random_color(n):
    """0から1で表現されたrgbをn個もつリストを返却

    Args:
        n (int): 要素数

    Returns:
        list[list]: rgbの数
    """
    color = [random.randint(0, 1) for _ in range(3)]
    return [color for i in range(n)]


class Mapbbox:
    """地図の表示領域を制御するクラス

    Returns:
        Mapbbox
    """

    def __init__(self):
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None

    def update(self, coord_x, coord_y):
        """点の座標をもとにboundingboxを更新する

        Args:
            coord_x (list[float]): x座標のリスト
            coord_y (list[float]): y座標のリスト
        """
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
        except TypeError:  # 初回のエラーに対する処理
            self.min_x = min_x
            self.min_y = min_y
            self.max_x = max_x
            self.max_y = max_y

    def offsetted(self, rate=0.1):
        """余白を作って表示する

        Args:
            rate (float, optional): 幅全体に対する余白の割合. Defaults to 0.1.

        Returns:
            (tuple(float)): (min_x, min_y, max_x, max_y)
        """
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
    for poly_categ in [site_shps.bldg]:
        coll = PolyCollection(poly_categ)
        ax.add_collection(coll)
    for line_categ in [site_shps.road]:
        for points in tqdm(line_categ):
            points = points.T
            ax.plot(points[0, :], points[1, :], color="g", lw=1)
            bbox.update(points[0, :], points[1, :])

    # 移動ルートを描画
    print("drawing routes")
    with h5py.File(str(gps_path), "r") as fg:
        for date in args.date:
            # gpsデータを解析して座標値をリストに格納する
            coord_x = []
            coord_y = []
            frame_count = len(fg[date].keys())
            for f in trange(frame_count, desc=f"{date}"):
                c_x, c_y, dire, ht = parse_gps_data(fg[date][str(f)])
                if c_x is None and c_y is None: # 欠損値に対する処理
                    continue
                coord_x.append(c_x)
                coord_y.append(c_y)
                if args.num:
                    ax.text(c_x, c_y, str(f), fontsize=10)
            # 各点を描画
            ax.scatter(
                coord_x,
                coord_y,
                s=10,
                c=_create_random_color(len(coord_x)),
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
    if hasattr(args, "handler"):
        args.handler(args)
