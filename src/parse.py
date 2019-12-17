import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

from geometry.shapes import ShapeFileCache, Site
from utils.color_output import output_with_color
from utils.tool import random_colors
from matplotlib.collections import PolyCollection


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
            except:  # hullが小さすぎてエラーを吐く場合
                error += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(volumes, bins=1000)
    # ax2.hist(areas, bins=50, range=(0, 10))
    fig_dir = Path("data", "png", args.site)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / Path("volume.png"), facecolor="azure", bbox_inches='tight', pad_inches=0)
    plt.show()


def parse_voronoi(args):
    shp_dir = Path("data", "shp", args.site)
    json_dir = Path("data", "json", args.site)
    with open(json_dir / Path("config.json"), "r") as f:
        file = json.load(f)
        black_list = file["blacklist"]
    shps = ShapeFileCache(shp_dir, json_dir)
    site = Site(shps)
    site.run_voronoi_building_land(args.interval, black_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    coll = PolyCollection([i.boundary for i in site.blocks], facecolor=(0.9,0.9,0.9))
    ax.add_collection(coll)

    for block in site.blocks:
        colors = random_colors(len(block.buildings))
        for idx, building in enumerate(block.buildings):
            if building.lands is None:
                continue
            coll = PolyCollection([i for i in building.lands], facecolor=colors[idx])
            ax.add_collection(coll)

        coll = PolyCollection([i.boundary for i in block.buildings], facecolor=(0.6, 0.6, 0.6))
        ax.add_collection(coll)

    ax.set_xlim([min(site.boundary[:, 0]), max(site.boundary[:, 0])])
    ax.set_ylim([min(site.boundary[:, 1]), max(site.boundary[:, 1])])
    # ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.show()
    print()


if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # 共通の引数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)
    parent_parser.add_argument("-d", "--date")

    # 各コマンドの設定
    volume_parser = subparsers.add_parser("volume", parents=[parent_parser])
    volume_parser.set_defaults(handler=parse_convex_hull_from_pcd)

    voronoi_parser = subparsers.add_parser("voronoi", parents=[parent_parser])
    voronoi_parser.add_argument("-i", "--interval", default=1)
    voronoi_parser.set_defaults(handler=parse_voronoi)
    args = parser.parse_args()

    if hasattr(args, "handler"):
        args.handler(args)
