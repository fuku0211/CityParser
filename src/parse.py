import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

from utils.color_output import output_with_color
from geometry.shapes import Shape

def parse_convex_hull_from_pcd(args):
    """点群を凸包を用いて解析し、そのヒストグラムを出力する

    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    output_with_color("parse volume")
    site_path = Path("data", "pts", args.site)
    routes = [i.name for i in site_path.iterdir()]
    if args.date is not None:
        routes = [args.date]

    volumes = []  # 凸包の体積
    areas = []  # 凸包の表面積
    error = 0
    for route in tqdm(routes):
        file_path = site_path / Path(route, "clustering.pts")
        with open(file_path, "rb") as f:
            lines = f.readlines()
        lines = lines[1:]  # ヘッドはスルー

        # 点群座標を抽出してnp.ndarrayに変換する
        xyz = [i.split()[:3] for i in lines]
        xyz = [list(map(float, i)) for i in xyz]
        xyz = np.asarray(xyz)

        # 各点がどのクラスタに属しているか示すインデックスのリスト
        cluster_idx = [int(line.split()[3].decode()) for line in lines]
        cluster_idx = np.asarray(cluster_idx)

        for idx in range(max(cluster_idx) + 1):
            # idx番目の点群クラスタの座標
            cluster_pts = xyz[np.where(cluster_idx == idx)[0], :]
            try:
                convexhull = ConvexHull(cluster_pts)
                volumes.append(convexhull.volume)
                areas.append(convexhull.area)
            except: # hullが小さすぎてエラーを吐く場合
                error += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(volumes, bins=50, range=(0, 3))
    ax1.set_title("volumes")
    ax2.hist(areas, bins=50, range=(0, 10))
    ax2.set_title("areas")
    plt.show()


def parse_voronoi(args):
    shp_dir = Path("data", "shp", args.site)
    json_dir = Path("data", "json", args.site)
    a = Shape(shp_dir, json_dir)
    print()

if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # 共通の引数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)
    parent_parser.add_argument("-d", "--date", required=True)

    # 各コマンドの設定
    volume_parser = subparsers.add_parser("volume", parents=[parent_parser])
    volume_parser.set_defaults(handler=parse_convex_hull_from_pcd)

    voronoi_parser = subparsers.add_parser("voronoi", parents=[parent_parser])
    voronoi_parser.set_defaults(handler=parse_voronoi)
    args = parser.parse_args()

    if hasattr(args, "handler"):
        args.handler(args)
