from contextlib import ExitStack
from itertools import product
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapefile
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union, split
from tqdm import tqdm

from utils.color_output import output_with_color
from utils.tool import get_key_from_value, plot_voronoi_with_label


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
        output_with_color("loading shape file", "g")
        self.site = self._load_site()
        self.block = None
        self.gcode = None
        self._load_blocks_and_gcode()

        self.bldg = self._load_parts("tatemono.shp")
        self.road = self._load_parts("road.shp")
        self.side = self._load_parts("hodou.shp")

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

    def _load_site(self):
        file_path = self.shp_dir / Path("site.shp")
        with shapefile.Reader(str(file_path)) as src:
            site = src.shapes()[0]
        return np.asarray(site.points)

    def _load_blocks_and_gcode(self):
        file_path = self.shp_dir / Path("gaiku.shp")
        with shapefile.Reader(str(file_path)) as src:
            blocks = src.shapes()
            rcds = src.records()
        self.block = [np.asarray(i.points) for i in blocks]
        self.gcode = [i["gcode"] for i in rcds]


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
            div = ceil(div) if div > 1 else 1
            x = np.linspace(edge_pts[0, 0], edge_pts[1, 0], div)
            y = np.linspace(edge_pts[0, 1], edge_pts[1, 1], div)
            # 結合用にndarrayに変換
            x = x.reshape(1, x.size)
            y = y.reshape(1, y.size)
            line_div_pts = np.concatenate([x, y]).T
            div_pts.append(line_div_pts[1:, :])  # 角の分割点が重ならないように
        return np.concatenate(div_pts)


class Land(MapObject):
    def __init__(self, boundary):
        super(Land, self).__init__(boundary)


class Building(MapObject):
    def __init__(self, boundary):
        super(Building, self).__init__(boundary)
        self.lands = None

    def set_lands(self, candidates):
        lands = []
        building = Polygon(self.boundary)
        for voronoi_poly in candidates:
            if building.intersects(Polygon(voronoi_poly)):  # 交差判定
                lands.append(voronoi_poly)
        self.lands = lands

    def calc_land_area(self):
        pass


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

    def run_voronoi_building_land(self, interval, black_list, extend=50):
        output_with_color("voronoi tesselation")
        for idx, block in enumerate(tqdm(self.blocks)):
            # 建物のない街区はスキップ
            if len(block.buildings) == 0:
                continue

            # ブラックリストをスキップ
            if block.gcode in black_list:
                continue

            # 街区内の建物の外周線上にボロノイ母点を配置する
            block_poly = Polygon(block.boundary)
            block_bldg_pts = []
            for building in block.buildings:
                bldg_div_pts = building.divide_boundary(interval)
                # 建物の外周線上の点が街区外にある場合削除する
                in_block = []
                for i, pts in enumerate(bldg_div_pts.tolist()):
                    if block_poly.contains(Point(pts[0], pts[1])):
                        in_block.append(i)
                bldg_div_pts = bldg_div_pts[in_block, :]
                block_bldg_pts.append(bldg_div_pts)

            block_mother_pts = np.concatenate(block_bldg_pts)

            # 重複した点はボロノイ分割でエラーを起こすため削除
            block_mother_pts = block_mother_pts.astype("float32")
            block_mother_pts = np.unique(block_mother_pts, axis=0)

            # ボロノイ作成
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection

            try:
                vor = Voronoi(block_mother_pts)
            except:
                # 母点の数が少ないとエラー
                exit()
            # plot_voronoi_with_label(vor)
            voronoi_polys = []  # 街区内のすべてのボロノイ
            for reg_idx, region in enumerate(vor.regions):
                # 無効な場合はスキップ
                if len(region) == 0:
                    continue

                # ボロノイが無限遠に伸びる辺を持つ時
                elif -1 in region:
                    # regionに対応するボロノイ母点を取り出す
                    mother_idx = np.where(vor.point_region == reg_idx)[0][0]
                    mother_pt = vor.points[mother_idx, :]

                    # FIXME: nazo
                    if len(region) <= 2:
                        continue

                    idx_inf = region.index(-1)
                    idx_a = -1 if idx_inf == 0 else idx_inf - 1
                    idx_b = 0 if idx_inf == len(region) - 1 else idx_inf + 1
                    # 2つの辺を構成する店のインデックス
                    edge_a = [region[idx_inf], region[idx_a]]
                    edge_b = [region[idx_inf], region[idx_b]]
                    # それぞれの辺を共有しているボロノイ母点のインデックスのペア
                    key_a = get_key_from_value(vor.ridge_dict, edge_a)
                    key_b = get_key_from_value(vor.ridge_dict, edge_b)

                    # FIXME: nazo
                    if len(key_a) == 1:
                        key_a = key_a[0]
                    else:
                        key_a = [i for i in key_a if mother_idx in i][0]
                    if len(key_b) == 1:
                        key_b = key_b[0]
                    else:
                        key_b = [i for i in key_b if mother_idx in i][0]
                    # 辺のベクトルを求める
                    vec_a = self._create_inf_vector(
                        vor.vertices[region[idx_a]],
                        vor.points[key_a[0], :],
                        vor.points[key_a[1], :],
                    )
                    vec_b = self._create_inf_vector(
                        vor.vertices[region[idx_b]],
                        vor.points[key_b[0], :],
                        vor.points[key_b[1], :],
                    )
                    # 内積が正かつ母点を含むポリゴンを生成する点が辺上の点になる
                    # TODO: ベクトルあたりが怪しい
                    patterns = product([vec_a, vec_a * -1], [vec_b, vec_b * -1])
                    patterns = [i for i in patterns if np.dot(i[0], i[1]) > 0]
                    poly = None
                    for pattern in patterns:
                        # 指定した位置に点を配置
                        replace_a = vor.vertices[region[idx_a]] + pattern[0] * extend
                        replace_b = vor.vertices[region[idx_b]] + pattern[1] * extend
                        pts = []
                        for i in region:
                            if i == -1:
                                pts.append(replace_a)
                                pts.append(replace_b)
                            else:
                                pts.append(vor.vertices[i, :])

                        p = Polygon(np.asarray(pts))
                        if p.is_valid is True and p.contains(Point(mother_pt[0], mother_pt[1])):
                            poly = p

                    # 街区とボロノイの共通部分を取りだす
                    if poly is None:
                        print()
                    overlaps = self._get_overlaps_block_and_voronoi(block_poly, poly)
                    if len(overlaps) != 0:
                        for p in overlaps:
                            voronoi_polys.append(p.exterior.coords)


                else:
                    # 街区とボロノイの共通部分を取りだす
                    verts = vor.vertices[region]
                    poly = Polygon(verts)
                    overlaps = self._get_overlaps_block_and_voronoi(block_poly, poly)
                    if len(overlaps) != 0:
                        for p in overlaps:
                            voronoi_polys.append(p.exterior.coords)

            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)

            # coll = PolyCollection([block.boundary], facecolor=(0.9,0.9,0.9))
            # ax.add_collection(coll)
            # coll = PolyCollection(voronoi_polys)
            # ax.add_collection(coll)

            # tmp = block.boundary
            # fig.axes[0].set_xlim([min(tmp[:, 0]), max(tmp[:, 0])])
            # fig.axes[0].set_ylim([min(tmp[:, 1]), max(tmp[:, 1])])
            # fig.axes[0].set_aspect("equal")
            # plt.show()

            # 各ボロノイを対応する建物に
            for building in block.buildings:
                building.set_lands(voronoi_polys)
