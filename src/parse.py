import argparse
import json
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from tqdm import tqdm

from geometry.shapes import Plant, ShapeFileCache, Site, plot_lands_in_site
from utils.color_output import output_with_color
from utils.tool import random_colors, select_pcd_setting


def _load_plants_from_pts(file_path):
    """敷地内の植物のリストを返却する
    """
    with open(file_path, "rb") as f:
        lines = f.readlines()
    lines = lines[1:]  # ヘッドはスルー

    output_with_color("loading file")
    xyz = []  # 点群座標
    cluster_idx = []  # 各点がどのクラスタに属しているか示すインデックスのリスト
    for line in tqdm(lines):
        tmp = line.split()
        xyz.append(list(map(float, tmp[:3])))
        cluster_idx.append(int(tmp[3].decode()))
    xyz = np.asarray(xyz)
    cluster_idx = np.asarray(cluster_idx)

    output_with_color("creating plants")
    plants = []
    for idx in tqdm(range(max(cluster_idx) + 1)):
        # idx番目の点群クラスタの座標
        cluster_pts = xyz[np.where(cluster_idx == idx)[0], :]
        try:
            convex_hull = ConvexHull(cluster_pts)
            plants.append(Plant(convex_hull))
        except:  # hullが小さすぎてエラーを吐く場合
            pass
    return plants


def parse_plant_volumes_from_pcd(args):
    """点群を凸包を用いて解析し、そのヒストグラムを出力する

    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    site_path = Path("data", "pts", args.site)
    file_name = select_pcd_setting(site_path)

    file_path = site_path / Path(file_name)
    plants = _load_plants_from_pts(file_path)
    volumes = [i.volume for i in plants]
    areas = [i.area for i in plants]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(volumes, bins=100, range=(0, 150))
    # ax2.hist(areas, bins=50, range=(0, 10))
    fig_dir = Path("data", "png", args.site)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        fig_dir / Path("volume.png"), bbox_inches="tight", pad_inches=0,
    )
    plt.show()


def parse_voronoi(args):
    # shapeファイルとコンフィグの読み込み
    shp_dir = Path("data", "shp", args.site)
    json_dir = Path("data", "json", args.site)
    pts_dir = Path("data", "pts", args.site)

    # 植生のロード
    file_name = select_pcd_setting(pts_dir)
    plants = _load_plants_from_pts(pts_dir / Path(file_name))
    centroids = [Point(i.centroid[0], i.centroid[1]) for i in plants]

    # shapeファイルのロード
    shps = ShapeFileCache(shp_dir, json_dir)
    site = Site(shps)

    # ボロノイ分割の実行
    site.run_voronoi_building_land(args.interval, shps.black_list)
    plot_lands_in_site(site, shps, plants)

    # 建物の土地あたりの植生体積量を計算する
    result = []
    for block in tqdm(site.blocks, desc="all"):
        for building in tqdm(block.buildings, desc=f"gcode:{block.gcode}", leave=False):
            land_poly = Polygon(building.land.boundary)
            volumes = []
            for i, plant in enumerate(tqdm(plants, leave=False)):
                # XY平面に投影して植生の重心が土地内にある場合建物の植生として扱う
                if land_poly.contains(centroids[i]):
                    volumes.append(plant.volume)
            if len(volumes) == 0:  # 土地内に植生がない場合
                continue
            result.append(sum(volumes) / building.land.area)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(result, bins=100, range=(0, 150))
    fig_dir = Path("data", "png", args.site)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        fig_dir / Path(f"volume_land_i={args.interval}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()


if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # 共通の引数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)

    # 各コマンドの設定
    volume_parser = subparsers.add_parser("volume", parents=[parent_parser])
    volume_parser.set_defaults(handler=parse_plant_volumes_from_pcd)

    voronoi_parser = subparsers.add_parser("voronoi", parents=[parent_parser])
    voronoi_parser.add_argument("-i", "--interval", default=1)
    voronoi_parser.set_defaults(handler=parse_voronoi)
    args = parser.parse_args()

    if hasattr(args, "handler"):
        args.handler(args)
