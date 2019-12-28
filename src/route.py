import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import gpxpy.gpx
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from tqdm import tqdm

from geometry.shapes import ShapeFileCache
from utils.color_output import output_with_color
from gps.convert import parse_lat_lon_from_gps, parse_x_y_from_gps


class Mapbbox:
    """地図の表示領域を保持する

    Attributes
    ----------
    min_x : float
        xの最小値
    min_y : float
        yの最小値
    max_x : float
        xの最大値
    max_y : float
        yの最大値
    polygon : Polygon
        オブジェクトが地図領域内にあるか判定するために使う
    """

    def __init__(self):
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    def update(self, coords_x, coords_y):
        """引数の点がbounding boxの外側にある時、その点が含まれるようにbounding boxの境界の値を調整する

        Parameters
        ----------
        coords_x : list
            x座標のリスト
        coords_y : list
            y座標のリスト
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

        Parameters
        ----------
        rate : float, optional
            幅全体に対する余白の割合, by default 0.1
        """
        offset_x = (self.max_x - self.min_x) * rate
        offset_y = (self.max_y - self.min_y) * rate
        self.min_x -= offset_x
        self.min_y -= offset_y
        self.max_x += offset_x
        self.max_y += offset_y

    def contain(self, geometry):
        """地図内にオブジェクトが含まれるか判定する

        Parameters
        ----------
        geometry : Polygon
            判定するオブジェクト

        Returns
        -------
        bool
            含む場合True
        """
        return self.polygon.contains(geometry)


def visualize_route(args):
    """歩いたルートを描画する

    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    text_step : int, optional
        フレーム数の表示の間隔, by default 10
    """
    date_path = Path("data", "hdf5", args.site)
    gps_path = date_path / Path("gps.hdf5")
    shape_path = Path("data", "shp", args.site)
    json_path = Path("data", "json", args.site)

    fig = plt.figure()
    bbox = Mapbbox()
    ax = fig.add_subplot(1, 1, 1)

    # 敷地地図を描画
    site_shps = ShapeFileCache(shape_path, json_path)

    # 敷地が入るように描画範囲を調整
    site_coords_x = site_shps.site.T[0, :]
    site_coords_y = site_shps.site.T[1, :]
    bbox.update(site_coords_x, site_coords_y)
    bbox.apply_margin()
    # 敷地の描画
    for i, code in enumerate(site_shps.gcode):
        if code in site_shps.black_list:
            ax.add_collection(
                PolyCollection([site_shps.block[i]], facecolor=(0.95, 0.8, 0.8))
            )
        else:
            ax.add_collection(
                PolyCollection([site_shps.block[i]], facecolor=(0.8, 0.95, 0.8))
            )

    # ポリラインの描画
    output_with_color("drawing polylines", "g")
    for line_categ in [site_shps.road, site_shps.side]:
        for l in tqdm(line_categ):
            ax.plot(l.T[0, :], l.T[1, :], color=(0, 0, 0), lw=0.2, zorder=1)

    # ポリゴンの描画
    output_with_color("drawing polygons", "g")
    for poly_categ in [site_shps.bldg]:
        # 地図のbbox内に存在するもののみを取り出して描画
        poly_inbbox = []
        for p in tqdm(poly_categ):
            if p.shape[0] == 2:  # 国土数値情報の建物情報に混じったただの直線を無視
                continue
            poly_inbbox.append(p)

        if args.road is False:
            coll = PolyCollection(poly_inbbox, facecolor=(0.3, 0.3, 0.3))
            ax.add_collection(coll)

    # 移動ルートを描画
    output_with_color("drawing routes", "g")
    with h5py.File(str(gps_path), "r") as fg:
        if args.all is True:  # 分割後のルートを処理する場合
            routes = [i for i in fg.keys() if args.date[0] + "_" in i]
        else:
            routes = args.date

        for route in tqdm(routes, desc="all"):
            # gpsデータを解析して座標値をリストに格納する
            route_coords_x = []
            route_coords_y = []
            frame_keys = list(map(int, fg[route].keys()))
            frame_keys.sort()
            if args.start is not None:  # 開始地点指定時は必要な部分のリストを取り出す
                frame_keys = [i for i in frame_keys if i >= args.start]
            for i, f in enumerate(tqdm(frame_keys, desc=f"{route}", leave=False)):
                if args.end is not None and f >= args.end:
                    break
                if f % args.step == 0:  # 指定した間隔の点のとき
                    c_x, c_y, dire, ht = parse_x_y_from_gps(fg[route][str(f)])
                    if c_x is None and c_y is None:  # 欠損値に対する処理
                        continue
                    route_coords_x.append(c_x)
                    route_coords_y.append(c_y)
                    if args.num:  # フレーム番号を表示する
                        # ax.text(c_x, c_y, str(f), fontsize=10)
                        ax.text(c_x, c_y, str(ht), fontsize=10)

            # 各点を描画
            ax.scatter(route_coords_x, route_coords_y, s=10, label=route, zorder=2)

    ax.set_xlim([bbox.min_x, bbox.max_x])
    ax.set_ylim([bbox.min_y, bbox.max_y])
    # ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.show()


class Route:
    """ルート処理を格納する

    Attributes
    ----------
    json_path : Path
        section.jsonのパス
    depth_path : Path
        depth.hdf5のパス
    color_path : Path
        color.hdf5のパス
    gps_path : Path
        gps.hdf5のパス
    seg_path : Path
        seg.hdf5のパス
    """

    def __init__(self, json_path, hdf5_path, seg=False):
        """

        Parameters
        ----------
        json_path : section.jsonのパス
            section.jsonのパス
        hdf5_path : Path
            hdf5フォルダ内の敷地フォルダのディレクトリ
        seg : bool, optional
            セグメンテーションファイルも対象にするか, by default False
        """
        self.json_path = json_path
        self.depth_path = hdf5_path / Path("depth.hdf5")
        self.color_path = hdf5_path / Path("color.hdf5")
        self.gps_path = hdf5_path / Path("gps.hdf5")
        self.seg_path = hdf5_path / Path("seg.hdf5") if seg else None

    def extract_sections_from_json(self, dates):
        """sectionファイルをもとにあるルートから必要な区間だけを抽出してhdf5ファイルに書き込む

        Parameters
        ----------
        dates : list
            処理対象の日時
        """
        # TODO: １ルートのみに対応
        with ExitStack() as stack:
            fj = stack.enter_context(open(self.json_path, "r"))
            fd = stack.enter_context(h5py.File(self.depth_path, "a"))
            fc = stack.enter_context(h5py.File(self.color_path, "a"))
            fg = stack.enter_context(h5py.File(self.gps_path, "a"))
            fs = None
            if self.seg_path is not None:
                fs = stack.enter_context(h5py.File(self.seg_path, "a"))

            sections_dict = json.load(fj)

            # segはない場合があるためその時は無視する
            if self.seg_path is None:
                srcs = [fd, fc, fg]
            else:
                srcs = [fd, fc, fg, fs]

            # コマンドライン引数に入力したルートがjson上で設定されていない場合処理を終了する
            if set(dates) <= set(sections_dict.keys()):
                pass
            else:
                print("Route config error")
                error = set(dates) - set(sections_dict.keys())
                print(f"You should create config file for {error}")
                exit()

            print("the number of routes")
            for section, sections in sections_dict.items():
                print(f"    - {section} -> {len(sections)} routes")

            for date, sections in tqdm(sections_dict.items(), desc="whole"):
                for sec_idx, section in enumerate(tqdm(sections, desc=f"{date}")):
                    sec_suffix = str(sec_idx).zfill(3)
                    group_name = date + f"_{sec_suffix}"
                    # 元のグループから情報を取得して抽出先のグループに書き込む
                    for src in srcs:
                        src_name = src.filename.split("\\")[-1]
                        # すでにファイル内に名前が重複したグループがあった場合削除する
                        if group_name in list(src.keys()):
                            del src[group_name]

                        out_group = src.create_group(group_name)  # 抽出先のグループ
                        desc_text = f"{src_name}:{sec_suffix}"
                        st = section[0]
                        end = section[1]
                        for f in tqdm(range(st, end), desc=desc_text, leave=False):
                            out_group.create_dataset(
                                str(f), data=src[date][str(f)], compression="gzip",
                            )


def extract_routes(args):
    """configファイルをもとにルートを抽出する

    """
    json_path = Path("data", "json", args.site, "section.json")
    hdf5_path = Path("data", "hdf5", args.site)

    if args.with_seg:
        routes = Route(json_path, hdf5_path, seg=True)
    else:
        routes = Route(json_path, hdf5_path)
    routes.extract_sections_from_json(args.date)


def create_gpx_from_hdf5(args):
    gpx = gpxpy.gpx.GPX()

    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack(name="test")
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment_a = gpxpy.gpx.GPXTrackSegment()
    gpx_segment_b = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment_a)
    gpx_track.segments.append(gpx_segment_b)

    gpx_path = Path("data", "gpx", args.site)
    gpx_path.mkdir(parents=True, exist_ok=True)

    for date in tqdm(args.date, desc="all"):
        px_file_w = open(str(gpx_path / Path(f"{date}.gpx")), "w")

        # Create points:
        hdf5_path = Path("data", "hdf5", args.site, "gps.hdf5")
        with h5py.File(str(hdf5_path), "r") as f:
            group_route = f[date]
            for i in tqdm(range(len(group_route.keys())), desc=f"{date}"):
                lat, lon = parse_lat_lon_from_gps(group_route[str(i)])
                gpx_segment_a.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

        px_file_w.write(gpx.to_xml(version="1.1"))
        px_file_w.close()


if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True, help="処理する敷地の名前")
    parent_parser.add_argument(
        "-d", "--date", required=True, nargs="+", help="処理する撮影日時"
    )

    # 各コマンドの設定
    vis_parser = subparsers.add_parser(
        "vis", parents=[parent_parser], help="歩行ルートを描画する"
    )
    vis_parser.add_argument("--start", type=int, default=0, help="経路を表示開始する番号")
    vis_parser.add_argument("--end", type=int, help="経路を表示終了する番号")
    vis_parser.add_argument("--num", action="store_true", help="移動経路に番号を表示する")
    vis_parser.add_argument("--road", action="store_true", help="道路だけを地図に表示する")
    vis_parser.add_argument("--all", action="store_true", help="分割後のルートをすべて扱う")
    vis_parser.add_argument("--step", type=int, default=10, help="表示する点の間隔")
    vis_parser.set_defaults(handler=visualize_route)

    # splitコマンドの動作
    split_parser = subparsers.add_parser(
        "split", parents=[parent_parser], help="全体の歩行ルートから必要な部分を抽出する"
    )
    split_parser.add_argument(
        "--with_seg", action="store_true", help="seg.hdf5も一緒に処理する"
    )
    split_parser.set_defaults(handler=extract_routes)

    # gpxコマンドの動作
    gpx_parser = subparsers.add_parser(
        "gpx", parents=[parent_parser], help="gpxファイルを作成する"
    )
    gpx_parser.set_defaults(handler=create_gpx_from_hdf5)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
