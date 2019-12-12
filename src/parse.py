import argparse
import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import (ConvexHull, Delaunay, Voronoi, delaunay_plot_2d,
                           voronoi_plot_2d)
from shapely.geometry import Polygon
from tqdm import tqdm

from geometry.shapes import Shape
from utils.color_output import output_with_color


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
    with open(json_dir / Path("config.json"), "r") as f:
        file = json.load(f)
        black_list = file["blacklist"]
    shps = Shape(shp_dir, json_dir)

    # 各街区のリストに、その街区内に建つ建物のPolygonを格納する
    all_buildings = []
    output_with_color("searching buildings in block", "g")
    for block in tqdm([Polygon(i) for i in shps.block]):
        block_buildings = []
        for building in [Polygon(i) for i in shps.bldg]:
            if block.contains(building):
                block_buildings.append(building)
        all_buildings.append(block_buildings)

    # 無視する街区をリストから排除
    blocks = []
    boundarys = []
    for i, code in enumerate(shps.gcode):
        if code not in black_list:
            blocks.append(all_buildings[i])
            boundarys.append(shps.block[i])

    output_with_color("voronoi tesselation")
    mother_pts = []
    for idx, block in enumerate(tqdm(blocks)):
        # 建物のない街区はスキップ
        if len(block) == 0:
            continue

        # 街区の外周線上にボロノイ母点を配置する
        block_pts = _divide_boundary(boundarys[idx], args.interval)

        # 街区内の建物の外周線上にボロノイ母点を配置する
        block_building_pts = []
        for building in block:
            building_bndry = np.asarray(building.exterior.coords)
            block_building_pts.append(_divide_boundary(building_bndry, args.interval))
        building_pts = np.concatenate(block_building_pts)

        # 街区内のすべてのボロノイ母点
        block_mother_pts = np.concatenate([block_pts, building_pts])

        vor = Voronoi(block_mother_pts)
        fig = voronoi_plot_2d(vor)
        fig.axes[0].set_aspect("equal")
        plt.show()
        # fig.savefig("test.png")
        print()

    mother_pts = np.concatenate(mother_pts)
    x, y = mother_pts.T

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y)
    plt.show()
    print()


def _divide_boundary(boundary, interval):
    div_pts = []
    for i in range(boundary.shape[0] - 1):
        edge_pts = boundary[i : i + 2, :]
        dist = np.linalg.norm(edge_pts[0, :] - edge_pts[1, :])
        # コマンドラインの引数をもとに分割数を決定する
        div = dist / interval
        div = ceil(div) if div > 1 else 1
        x = np.linspace(edge_pts[0, 0], edge_pts[1, 0], div)
        y = np.linspace(edge_pts[0, 1], edge_pts[1, 1], div)
        # 結合用にndarrayに変換
        x = x.reshape(1, x.size)
        y = y.reshape(1, y.size)
        div_pts.append(np.concatenate([x, y]).T)
    return np.concatenate(div_pts)


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
    voronoi_parser.add_argument("-i", "--interval", default=0.3)
    voronoi_parser.set_defaults(handler=parse_voronoi)
    args = parser.parse_args()

    if hasattr(args, "handler"):
        args.handler(args)
