from contextlib import ExitStack
from itertools import product
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapefile
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import cascaded_union, split
from tqdm import tqdm

from utils.color_output import output_with_color
from utils.tool import get_key_from_value, plot_voronoi_with_label, random_colors
import json


class ShapeFileCache:
    """shapeファイルを読み込んだ情報を格納

    Attributes
    ----------
    shp_dir : Path
        shapeフォルダのディレクトリ
    json_dir : Path
        jsonフォルダのディレクトリ
    bldg : list
        建物の輪郭線座標
    road : list
        道路の輪郭線座標
    side : list
        歩道の輪郭線座標
    """

    def __init__(self, shp_dir, json_dir):
        """

        Parameters
        ----------
        shp_dir : Path
            shapeフォルダのディレクトリ
        json_dir : Path
            jsonフォルダのディレクトリ
        """
        self.shp_dir = shp_dir
        self.json_dir = json_dir

        output_with_color("loading config file", "g")
        self.black_list = None
        self.acode = None
        self.ccode = None
        self._load_config_file()

        output_with_color("loading shape file", "g")
        self.site = None
        self.block = None
        self.gcode = None
        self._load_site_and_block()

        self.bldg = self._load_parts("tatemono.shp")
        self.road = self._load_parts("road.shp")
        self.side = self._load_parts("hodou.shp")

    def _load_config_file(self):
        file_path = self.json_dir / Path("config.json")
        with open(file_path) as f:
            conf_dict = json.load(f)
        self.black_list = conf_dict["blacklist"]
        self.acode = conf_dict["acode"]
        self.ccode = conf_dict["ccode"]

    def _load_parts(self, file_name):
        """shapeファイルからラインの座標を読み込む

        Parameters
        ----------
        file_name : str
            ファイル名

        Returns
        -------
        list
            ラインの座標値
        """
        file_path = self.shp_dir / Path(file_name)

        shp_points_all = []
        with shapefile.Reader(str(file_path)) as src:
            shps = src.shapes()
        site_poly = Polygon(self.site)

        # 敷地内のshapeのみを抽出する
        if shps[0].shapeType == 5:  # Polygon
            for shp in tqdm(shps, desc=f"{file_name}"):
                if Polygon(shp.points).intersects(site_poly):
                    shp_points_all.append(np.asarray(shp.points))
        else:
            for shp in tqdm(shps, desc=f"{file_name}"):
                if LineString(shp.points).intersects(site_poly):
                    shp_points_all.append(np.asarray(shp.points))

        return shp_points_all

    def _load_site_and_block(self):
        file_path = self.shp_dir / Path("gaiku.shp")
        with shapefile.Reader(str(file_path)) as src:
            blocks = src.shapes()
            rcds = src.records()
        blocks_in_site = []
        gcodes_in_site = []
        for i, rcd in enumerate(rcds):
            if rcd["acode"] == str(self.acode) and rcd["ccode"] == str(self.ccode):
                blocks_in_site.append(blocks[i].points)
                gcodes_in_site.append(rcd["gcode"])
        site = cascaded_union([Polygon(i) for i in blocks_in_site])
        self.site = np.asarray(site.exterior.coords)
        self.block = blocks_in_site
        self.gcode = gcodes_in_site


class MapObject(object):
    def __init__(self, boundary):
        self.boundary = boundary

    def divide_boundary(self, interval):
        div_pts = []
        for i in range(self.boundary.shape[0] - 1):
            edge_pts = self.boundary[i : i + 2, :]
            dist = np.linalg.norm(edge_pts[0, :] - edge_pts[1, :])
            # コマンドラインの引数をもとに分割数を決定する
            div = dist / interval
            div = ceil(div) if div > 3 else 3
            x = np.linspace(edge_pts[0, 0], edge_pts[1, 0], div)
            y = np.linspace(edge_pts[0, 1], edge_pts[1, 1], div)
            # 結合用にndarrayに変換
            x = x.reshape(1, x.size)
            y = y.reshape(1, y.size)
            line_div_pts = np.concatenate([x, y]).T
            div_pts.append(line_div_pts[1:, :])  # 角の分割点が重ならないように
        return np.concatenate(div_pts)


class Land:
    def __init__(self, parts):
        self.parts = parts
        self.boundary = self._get_land_boundary()
        self.area = self._get_land_area()

    def _get_land_boundary(self):
        union = cascaded_union([Polygon(i) for i in self.parts])
        return union.exterior.coords

    def _get_land_area(self):
        return Polygon(self.boundary).area


class Building(MapObject):
    def __init__(self, boundary):
        super(Building, self).__init__(boundary)
        self.land = None

    def set_lands(self, candidates):
        parts = []
        building = Polygon(self.boundary)
        for voronoi in candidates:
            if building.intersects(Polygon(voronoi)):  # 交差判定
                parts.append(voronoi)
        self.land = Land(parts)


class Block(MapObject):
    def __init__(self, boundary, gcode):
        super(Block, self).__init__(boundary)
        self.gcode = gcode
        self.buildings = None

    def set_buildings(self, candidates):
        block_buildings = []
        block = Polygon(self.boundary)
        for candidate in candidates:
            if block.contains(Polygon(candidate)):
                block_buildings.append(Building(candidate))
        self.buildings = block_buildings


class Site(MapObject):
    def __init__(self, shp_cache):
        super(Site, self).__init__(shp_cache.site)
        self.blocks = self._create_blocks(shp_cache)

    def _create_blocks(self, shp_cache):
        output_with_color("setting building and blocks", "g")
        blocks = []
        for i, boundary in enumerate(tqdm(shp_cache.block)):
            block = Block(boundary, shp_cache.gcode[i])
            block.set_buildings(shp_cache.bldg)
            blocks.append(block)
        return blocks

    @staticmethod
    def _create_inf_vector(start_pt, pt_a, pt_b):
        """ボロノイ分割で無限遠方向に伸びる辺のベクトルを求める

        Parameters
        ----------
        start_pt : array
            辺の始点
        pt_a : array
            ボロノイ母点
        pt_b : array
            ボロノイ母点

        Returns
        -------
        array
        """
        mid_pt = (pt_a + pt_b) / 2
        vec = start_pt - mid_pt
        return vec / np.linalg.norm(vec)

    @staticmethod
    def _get_overlaps_block_and_voronoi(block_poly, vor_poly):
        try:
            overlap = block_poly.intersection(vor_poly)
            if isinstance(overlap, MultiPolygon):
                overlap_poly = list(overlap)
            else:
                overlap_poly = [overlap]
        except:
            print("overlap error")
            overlap_poly = []
        return overlap_poly

    def run_voronoi_building_land(self, interval, black_list):
        output_with_color("voronoi tesselation")
        for idx, block in enumerate(tqdm(self.blocks)):
            # 建物のない街区はスキップ
            if len(block.buildings) == 0:
                continue

            # ブラックリスト入りしている街区はスキップ
            if block.gcode in black_list:
                continue

            # 街区の外周線上にボロノイ母点を配置する
            block_div_pts = block.divide_boundary(interval)

            # 街区内の建物の外周線上にボロノイ母点を配置する
            bldg_div_pts = [i.divide_boundary(interval) for i in block.buildings]
            block_mother_pts = np.concatenate([block_div_pts, np.vstack(bldg_div_pts)])

            # 重複した点はボロノイ分割でエラーを起こすため削除
            block_mother_pts = block_mother_pts.astype("float32")
            block_mother_pts = np.unique(block_mother_pts, axis=0)

            vor = Voronoi(block_mother_pts)
            voronois = []
            for reg_idx, region in enumerate(vor.regions):
                # ボロノイが無効or無限遠に伸びる辺を持つ時
                if len(region) == 0 or -1 in region:
                    continue
                else:
                    voronois.append(np.asarray([vor.vertices[i] for i in region]))

            for building in block.buildings:
                building.set_lands(voronois)

    # def run_voronoi_building_land(self, interval, black_list, extend=50):
    #     output_with_color("voronoi tesselation")
    #     for idx, block in enumerate(tqdm(self.blocks)):
    #         # 建物のない街区はスキップ
    #         if len(block.buildings) == 0:
    #             continue

    #         # ブラックリストをスキップ
    #         if block.gcode in black_list:
    #             continue

    #         # 街区内の建物の外周線上にボロノイ母点を配置する
    #         block_poly = Polygon(block.boundary)
    #         block_bldg_pts = []
    #         for building in block.buildings:
    #             bldg_div_pts = building.divide_boundary(interval)
    #             # 建物の外周線上の点が街区外にある場合削除する
    #             in_block = []
    #             for i, pts in enumerate(bldg_div_pts.tolist()):
    #                 if block_poly.contains(Point(pts[0], pts[1])):
    #                     in_block.append(i)
    #             bldg_div_pts = bldg_div_pts[in_block, :]
    #             block_bldg_pts.append(bldg_div_pts)

    #         block_mother_pts = np.concatenate(block_bldg_pts)

    #         # 重複した点はボロノイ分割でエラーを起こすため削除
    #         block_mother_pts = block_mother_pts.astype("float32")
    #         block_mother_pts = np.unique(block_mother_pts, axis=0)

    #         # ボロノイ作成
    #         try:
    #             vor = Voronoi(block_mother_pts)
    #         except:
    #             # 母点の数が少ないとエラー
    #             exit()
    #         # plot_voronoi_with_label(vor)
    #         voronoi_polys = []  # 街区内のすべてのボロノイ
    #         for reg_idx, region in enumerate(vor.regions):
    #             # 無効な場合はスキップ
    #             if len(region) == 0:
    #                 continue

    #             # ボロノイが無限遠に伸びる辺を持つ時
    #             elif -1 in region:
    #                 # regionに対応するボロノイ母点を取り出す
    #                 mother_idx = np.where(vor.point_region == reg_idx)[0][0]
    #                 mother_pt = vor.points[mother_idx, :]

    #                 # FIXME: nazo
    #                 if len(region) <= 2:
    #                     continue

    #                 idx_inf = region.index(-1)
    #                 idx_a = -1 if idx_inf == 0 else idx_inf - 1
    #                 idx_b = 0 if idx_inf == len(region) - 1 else idx_inf + 1
    #                 # 2つの辺を構成する店のインデックス
    #                 edge_a = [region[idx_inf], region[idx_a]]
    #                 edge_b = [region[idx_inf], region[idx_b]]
    #                 # それぞれの辺を共有しているボロノイ母点のインデックスのペア
    #                 key_a = get_key_from_value(vor.ridge_dict, edge_a)
    #                 key_b = get_key_from_value(vor.ridge_dict, edge_b)

    #                 # FIXME: nazo
    #                 if len(key_a) == 1:
    #                     key_a = key_a[0]
    #                 else:
    #                     key_a = [i for i in key_a if mother_idx in i][0]
    #                 if len(key_b) == 1:
    #                     key_b = key_b[0]
    #                 else:
    #                     key_b = [i for i in key_b if mother_idx in i][0]
    #                 # 辺のベクトルを求める
    #                 vec_a = self._create_inf_vector(
    #                     vor.vertices[region[idx_a]],
    #                     vor.points[key_a[0], :],
    #                     vor.points[key_a[1], :],
    #                 )
    #                 vec_b = self._create_inf_vector(
    #                     vor.vertices[region[idx_b]],
    #                     vor.points[key_b[0], :],
    #                     vor.points[key_b[1], :],
    #                 )
    #                 # 内積が正かつ母点を含むポリゴンを生成する点が辺上の点になる
    #                 # TODO: ベクトルあたりが怪しい
    #                 patterns = product([vec_a, vec_a * -1], [vec_b, vec_b * -1])
    #                 patterns = [i for i in patterns if np.dot(i[0], i[1]) > 0]
    #                 poly = None
    #                 for pattern in patterns:
    #                     # 指定した位置に点を配置
    #                     replace_a = vor.vertices[region[idx_a]] + pattern[0] * extend
    #                     replace_b = vor.vertices[region[idx_b]] + pattern[1] * extend
    #                     pts = []
    #                     for i in region:
    #                         if i == -1:
    #                             pts.append(replace_a)
    #                             pts.append(replace_b)
    #                         else:
    #                             pts.append(vor.vertices[i, :])

    #                     p = Polygon(np.asarray(pts))
    #                     if p.is_valid is True and p.contains(Point(mother_pt[0], mother_pt[1])):
    #                         poly = p

    #                 # 街区とボロノイの共通部分を取りだす
    #                 if poly is None:
    #                     print()
    #                 overlaps = self._get_overlaps_block_and_voronoi(block_poly, poly)
    #                 if len(overlaps) != 0:
    #                     for p in overlaps:
    #                         voronoi_polys.append(p.exterior.coords)

    #             else:
    #                 # 街区とボロノイの共通部分を取りだす
    #                 verts = vor.vertices[region]
    #                 poly = Polygon(verts)
    #                 overlaps = self._get_overlaps_block_and_voronoi(block_poly, poly)
    #                 if len(overlaps) != 0:
    #                     for p in overlaps:
    #                         voronoi_polys.append(p.exterior.coords)

    #         # 各ボロノイを対応する建物に
    #         for building in block.buildings:
    #             building.set_lands(voronoi_polys)


class Plant:
    def __init__(self, convex_hull):
        self.convex_hull = convex_hull
        self.volume = self.convex_hull.volume
        self.area = self.convex_hull.area
        self.centroid = self._set_centroid()

    def _set_centroid(self):
        x = np.average(self.convex_hull.points[:, 0])
        y = np.average(self.convex_hull.points[:, 1])
        z = np.average(self.convex_hull.points[:, 2])
        return np.array([x, y, z])


def plot_lands_in_site(
    site,
    shp_cache,
    plants,
    c_block=(0.9, 0.9, 0.9),
    c_bldg=(0.6, 0.6, 0.6),
    c_line=(0, 0, 0),
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coll = PolyCollection([i.boundary for i in site.blocks], c_block)
    ax.add_collection(coll)

    for block in site.blocks:
        colors = random_colors(len(block.buildings))
        # 各建物の土地を描画
        for i, bldg in enumerate(block.buildings):
            coll = PolyCollection([bldg.land.boundary], facecolor=colors[i])
            ax.add_collection(coll)

    # 道路と歩道を描画
    for line_categ in [shp_cache.road, shp_cache.side]:
        for l in line_categ:
            ax.plot(l.T[0, :], l.T[1, :], color=c_line, lw=0.2, zorder=1)

    # 街区内の建物を描画
    for block in site.blocks:
        coll = PolyCollection([i.boundary for i in block.buildings], facecolor=c_bldg)
        ax.add_collection(coll)

    tmp = [(i.centroid[0], i.centroid[1]) for i in plants]
    x = np.asarray(tmp).T[0]
    y = np.asarray(tmp).T[1]
    ax.scatter(x, y, s=10)

    ax.set_xlim([min(site.boundary[:, 0]), max(site.boundary[:, 0])])
    ax.set_ylim([min(site.boundary[:, 1]), max(site.boundary[:, 1])])
    ax.set_aspect("equal")
    plt.show()
