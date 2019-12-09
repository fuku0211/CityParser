import shapefile
from pathlib import Path
from tqdm import tqdm
import numpy as np
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
    """
    def __init__(self, shp_path):
        """

        Parameters
        ----------
        shp_path : Path
            shapeファイルのパス
        """
        self.path = shp_path
        output_with_color("loading shape file", "g")
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
        file_path = self.path / Path(file_name)

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
