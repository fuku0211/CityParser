import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import shapefile
from shapely.geometry import Polygon, LineString
from shapely.ops import cascaded_union
from tqdm import tqdm

from utils.color_output import output_with_color


class Shape:
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
            if shps[0].shapeType == 5:
                for shp in tqdm(shps, desc=f"{file_name}"):
                    if Polygon(shp.points).intersects(Polygon(self.site)):
                        shp_points_all.append(np.asarray(shp.points))
            else:
                for shp in tqdm(shps, desc=f"{file_name}"):
                    if LineString(shp.points).intersects(Polygon(self.site)):
                        shp_points_all.append(np.asarray(shp.points))

        return shp_points_all

    def _load_site_and_blocks_with_gcode(self):
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
