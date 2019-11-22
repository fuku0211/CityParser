import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import LineString, Polygon
from tqdm import tqdm, trange

from geometry.shapes import Shape
from utils.tool import parse_gps_data
from utils.color_output import output_with_color


class Mapbbox:
    """地図の表示領域を制御するクラス

    Returns:
        Mapbbox
    """

    def __init__(self):
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    def update(self, coords_x, coords_y):
        """引数の点がbounding boxの外側にあるとき、その点が含まれるようにbounding boxの境界の値を調整する

        Args:
            coords_x (list[float]): x座標のリスト
            coords_y (list[float]): y座標のリスト
        """

        min_x = min(coords_x)
        min_y = min(coords_y)
        max_x = max(coords_x)
        max_y = max(coords_y)
        # bboxの境界の外側に
        try:
            if min_x < self.min_x:
                self.min_x = min_x
            if min_y < self.min_y:
                self.min_y = min_y
            if max_x > self.max_x:
                self.max_x = max_x
            if max_y > self.max_y:
                self.max_y = max_y

        # 初回はNoneを比較することになりエラーが発生する
        # 対策として引数をそのままプロパティに代入する
        except TypeError:
            self.min_x = min_x
            self.min_y = min_y
            self.max_x = max_x
            self.max_y = max_y

        finally:
            self.polygon = Polygon(
                [
                    (self.min_x, self.min_y),
                    (self.max_x, self.min_y),
                    (self.max_x, self.max_y),
                    (self.min_x, self.max_y),
                ]
            )

    def apply_margin(self, rate=0.1):
        """bboxの境界を余白を含めた値に変更する

        Args:
            rate (float, optional): 幅全体に対する余白の割合. Defaults to 0.1.

        Returns:
            (tuple(float)): (min_x, min_y, max_x, max_y)
        """
        offset_x = (self.max_x - self.min_x) * rate
        offset_y = (self.max_y - self.min_y) * rate
        self.min_x -= offset_x
        self.min_y -= offset_y
        self.max_x += offset_x
        self.max_y += offset_y

    def contain(self, geometry):
        return self.polygon.contains(geometry)


def visualize_route(args, text_step=10):
    date_path = Path("data", "hdf5", args.site)
    shape_path = Path("data", "shp", args.site)
    gps_path = date_path / Path("gps.hdf5")

    fig = plt.figure()
    bbox = Mapbbox()
    ax = fig.add_subplot(1, 1, 1)

    # 移動ルートを描画
    output_with_color("drawing routes", "g")
    with h5py.File(str(gps_path), "r") as fg:
        for date in args.date:
            # gpsデータを解析して座標値をリストに格納する
            route_coords_x = []
            route_coords_y = []
            frame_count = len(fg[date].keys())
            for f in trange(args.start, frame_count, 1, desc=f"{date}"):
                c_x, c_y, dire, ht = parse_gps_data(fg[date][str(f)])
                if c_x is None and c_y is None:  # 欠損値に対する処理
                    continue
                route_coords_x.append(c_x)
                route_coords_y.append(c_y)
                if args.num and f % text_step == 0:  # フレーム番号を一定間隔で表示する
                    ax.text(c_x, c_y, str(f), fontsize=10)
            # 各点を描画
            ax.scatter(
                route_coords_x, route_coords_y, s=10, label=date, zorder=2
            )
            bbox.update(route_coords_x, route_coords_y)

    # 敷地地図を描画
    site_shps = Shape(shape_path)
    output_with_color("drawing shapes", "g")
    for poly_categ in [site_shps.bldg]:
        # 地図のbbox内に存在するもののみを取り出して描画
        poly_inbbox = []
        for p in tqdm(poly_categ):
            if p.shape[0] == 2:  # 国土数値情報の建物情報に混じったただの直線を無視
                continue
            if bbox.contain(Polygon(p)):
                poly_inbbox.append(p)

        if args.road is False:
            coll = PolyCollection(poly_inbbox, facecolor=(0.9, 0.9, 0.9))
            ax.add_collection(coll)

    for line_categ in [site_shps.road, site_shps.side]:
        # 地図のbbox内に存在するもののみを取り出して描画
        for l in tqdm(line_categ):
            if bbox.contain(LineString(l)):
                ax.plot(l.T[0, :], l.T[1, :], color=(0, 0, 0), lw=0.2, zorder=1)

    bbox.apply_margin()
    ax.set_xlim([bbox.min_x, bbox.max_x])
    ax.set_ylim([bbox.min_y, bbox.max_y])
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.show()


class Route:
    def __init__(self, json_path, hdf5_path, seg=False):
        self.json_path = json_path
        self.depth_path = hdf5_path / Path("depth.hdf5")
        self.color_path = hdf5_path / Path("color.hdf5")
        self.gps_path = hdf5_path / Path("gps.hdf5")
        self.seg_path = hdf5_path / Path("seg.hdf5") if seg else None

    def extract_routes_from_config(self, dates):
        """configファイルを元に、あるルートから必要な部分だけを抽出する

        Args:
            dates (list[str]): 処理対象の日時
        """
        with ExitStack() as stack:
            fj = stack.enter_context(open(self.json_path, "r"))
            fd = stack.enter_context(h5py.File(self.depth_path, "a"))
            fc = stack.enter_context(h5py.File(self.color_path, "a"))
            fg = stack.enter_context(h5py.File(self.gps_path, "a"))
            fs = None
            if self.seg_path is not None:
                fs = stack.enter_context(h5py.File(self.seg_path, "a"))

            routes_dict = json.load(fj)

            # コマンドライン引数に入力したルートがjson上で設定されていない場合処理を終了する
            if set(dates) <= set(routes_dict.keys()):
                pass
            else:
                print("Route config error")
                error = set(dates) - set(routes_dict.keys())
                print(f"You should create config file for {error}")
                exit()

            print("the number of routes")
            for route, sections in routes_dict.items():
                print(f"    - {route} -> {len(sections)} routes")

            # segはない場合があるためその時は無視する
            if self.seg_path is None:
                key_files = [fd, fc, fg]
            else:
                key_files = [fd, fc, fg, fs]

            for route, sections in tqdm(routes_dict.items(), desc="whole"):
                for section_name, sections in tqdm(sections.items(), desc=f"{route}"):
                    group_name = route + f"_{section_name}"
                    # すでにファイル内に名前が重複したグループがあった場合削除する
                    for file in key_files:
                        if group_name in list(file.keys()):
                            del file[group_name]

                    # 元のグループから情報を取得して抽出先のグループに書き込む
                    for file in key_files:
                        out_group = file.create_group(group_name)  # 抽出先のグループ
                        categ_name = file.filename.split("\\")[-1]
                        desc_text = f"{categ_name}:{section_name}"
                        for section in sections:
                            st = section[0]
                            end = section[1]
                            for f in tqdm(range(st, end), desc=desc_text, leave=False):
                                out_group.create_dataset(
                                    str(f), data=file[route][str(f)], compression="gzip"
                                )


def extract_routes(args):
    """configファイルをもとにルートを抽出する

    """
    json_path = Path("data", "json", args.site, "config.json")
    hdf5_path = Path("data", "hdf5", args.site)

    if args.with_seg:
        routes = Route(json_path, hdf5_path, seg=True)
    else:
        routes = Route(json_path, hdf5_path)
    routes.extract_routes_from_config(args.date)


if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)
    parent_parser.add_argument("-d", "--date", required=True, nargs="+")

    # 各コマンドの設定
    vis_parser = subparsers.add_parser("vis", parents=[parent_parser])
    vis_parser.add_argument("--num", action="store_true")
    vis_parser.add_argument("--start", type=int)
    vis_parser.add_argument("--road", action="store_true")
    vis_parser.set_defaults(handler=visualize_route)

    split_parser = subparsers.add_parser("split", parents=[parent_parser])
    split_parser.add_argument("--with_seg", action="store_true")
    split_parser.set_defaults(handler=extract_routes)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
