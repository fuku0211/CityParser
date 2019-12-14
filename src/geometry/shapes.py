from contextlib import ExitStack
from math import ceil
from pathlib import Path

import numpy as np
import shapefile
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Polygon
from shapely.ops import cascaded_union
from tqdm import tqdm

from utils.color_output import output_with_color


class ShapeCollection():
    """shapeファイルを読み込んだ情報を格納

    Attributes
    ----------
    path : Path
        shapeファイルのパス
    bldg : list
        建物の輪郭線座標
    road : list
        道路の輪郭線座標
    side : list
        歩道の輪郭線座標
    site : list
        敷地の外周線座標
    block : list
        各街区の外周線座標
    gcode : list
        街区コード
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
        # 敷地と街区を読み込む
        self.site = None
        self.block = None
        self.gcode = None
        self._load_site_and_blocks_with_gcode()

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

    def _load_site_and_blocks_with_gcode(self):
        """[summary]
        """
        shp_path = self.shp_dir / Path("gaiku.shp")
        # shapeファイルを開く
        with ExitStack() as stack:
            fp = stack.enter_context(shapefile.Reader(str(shp_path)))
            blocks_parts = [i.points for i in fp.shapes()]
            records = fp.records()

        # gcodeが同じポリゴンをまとめる
        gcodes = [i["gcode"] for i in records]
        gcodes_set = list(set(gcodes))
        blocks_joined = []
        for code in gcodes_set:
            block_idx = [i for i, x in enumerate(gcodes) if x == code]
            marge_shp = [Polygon(blocks_parts[i]) for i in block_idx]
            if len(marge_shp) == 1:
                marged_block = marge_shp[0]
            else:
                marged_block = cascaded_union(marge_shp)
            block_coords = np.asarray(marged_block.exterior.coords)
            blocks_joined.append(block_coords)

        site = cascaded_union([Polygon(i) for i in blocks_joined])
        self.site = np.asarray(site.exterior.coords)
        self.block = blocks_joined
        self.gcode = gcodes_set


class Site():
    def __init__(self, shps, black_list):
        self._black_list = black_list
        self.block_buildings = self._get_block_buildings_from_shp(shps)
        self.block_boundaries = self._get_block_boundaries_from_shp(shps)

    def _get_block_buildings_from_shp(self, shps):
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
        for i, code in enumerate(shps.gcode):
            if code not in self._black_list:
                blocks.append(all_buildings[i])
        return blocks

    def _get_block_boundaries_from_shp(self, shps):
        # 無視する街区をリストから排除
        boundaries = []
        for i, code in enumerate(shps.gcode):
            if code not in self._black_list:
                boundaries.append(shps.block[i])
        return boundaries

    @classmethod
    def _divide_boundary(cls, boundary, interval):
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
            line_div_pts = np.concatenate([x, y]).T
            div_pts.append(line_div_pts[1:, :])  # 角の分割点が重ならないように
        return np.concatenate(div_pts)

    def get_block_voronoi(self, interval):
        output_with_color("voronoi tesselation")
        all_voronoi = []
        for idx, block in enumerate(tqdm(self.block_buildings)):
            # 建物のない街区はスキップ
            if len(block) == 0:
                all_voronoi.append([])
                continue

            # 街区の外周線上にボロノイ母点を配置する
            block_pts = self._divide_boundary(self.block_boundaries[idx], interval)

            # 街区内の建物の外周線上にボロノイ母点を配置する
            block_bldg_pts = []
            for building in block:
                building_bndry = np.asarray(building.exterior.coords)
                block_bldg_pts.append(self._divide_boundary(building_bndry, interval))
            building_pts = np.concatenate(block_bldg_pts)

            # 街区内のすべてのボロノイ母点
            block_mother_pts = np.concatenate([block_pts, building_pts])
            # 重複した点はボロノイ分割でエラーを起こすため削除
            block_mother_pts = block_mother_pts.astype("float32")
            block_mother_pts = np.unique(block_mother_pts, axis=0)

            # ボロノイ作成
            vor = Voronoi(block_mother_pts)
            voronoi_polys = []  # 街区内のすべてのボロノイ
            for region in vor.regions:
                # 街路外周線上の点をボロノイ母点としている場合スキップ
                if -1 in region or len(region) == 0:
                    continue
                verts = vor.vertices[region]
                voronoi_polys.append(Polygon(verts))

            # 建物に対応するリストに建物と交差するボロノイを格納する
            block_voronoi = []
            for building in self.block_buildings[idx]:
                vor_in_building = []
                for voronoi_poly in voronoi_polys:
                    if building.intersects(voronoi_poly): # 交差判定
                        vor_in_building.append(voronoi_poly)
                block_voronoi.append(vor_in_building)

            all_voronoi.append(block_voronoi)
        return all_voronoi
