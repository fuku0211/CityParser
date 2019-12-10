import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import shapefile
from shapely.geometry import Polygon
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
        self.bldg = self._load_parts("tatemono.shp")
        self.road = self._load_parts("road.shp")
        self.side = self._load_parts("hodou.shp")
        self.site = self._load_site_boundary()

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
            for shp in tqdm(shps, desc=f"{file_name}"):
                # 線の内部に線がある場合は外の線と中の線を区別する
                if len(shp.parts) > 1:
                    parts_idx = shp.parts.append(len(shp.points) - 1)
                    for i in range(0, len(shp.points) - 1, 2):
                        shp_points_all.append(
                            shp.points[parts_idx[i] : parts_idx[i + 1]]
                        )
                else:
                    shp_points_all.append(np.asarray(shp.points, dtype="float32"))
        return shp_points_all

    def _load_site_boundary(self):
        shp_path = self.shp_dir / Path("gaiku.shp")
        json_path = self.json_dir / Path("config.json")
        # shapeファイルとjsonファイルを開く
        with ExitStack() as stack:
            fp = stack.enter_context(shapefile.Reader(str(shp_path)))
            fj = stack.enter_context(open(json_path, "r"))
            shps = fp.shapes()
            rcds = fp.records()
            configs = json.load(fj)

        # 敷地内の街区とそのレコードを取り出す
        shps_in_site = []
        rcds_in_site = []
        for i, r in enumerate(rcds):
            if r["acode"] == configs["acode"] and r["ccode"] == configs["ccode"]:
                shp = Polygon(shps[i].points)
                shps_in_site.append(shp)
                rcds_in_site.append(rcds[i])
        site = cascaded_union(shps_in_site)
        return np.asarray(site.exterior.coords)

    def _load_block(self):
        pass
