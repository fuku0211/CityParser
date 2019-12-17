import argparse
import math
from contextlib import ExitStack
from pathlib import Path
from time import time

import h5py
import numpy as np
import open3d as o3d
import pyproj
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.dbscan import dbscan
from tqdm import tqdm, trange

from geometry.capture import CameraIntrinsic
from utils.tool import (
    array_to_3dim,
    calc_angle_between_axis,
    parse_x_y_from_gps,
    random_colors,
)
from utils.color_output import output_with_color


class Frame:
    """1フレームに関する情報を保持する.

    Attributes
    ----------
    depth : ndarray
        各ピクセルにおけるカメラから物体までの距離
    color : ndarray
        録画のRGB
    gps_from : ndarray
        このフレームのGPS
    gps_to : ndarray
        1つ先のフレームのGPS
    seg : ndarray
        セグメンテーション結果
    """
    def __init__(self, depth, color, gps_from, gps_to, seg=None):
        """1フレームに関する情報を保持する.

        Parameters
        ----------
        depth : ndarray
            各ピクセルにおけるカメラから物体までの距離
        color : ndarray
            録画のRGB
        gps_from : ndarray
            このフレームのGPS
        gps_to : ndarray
            1つ先のフレームのGPS
        seg : ndarray, optional
            セグメンテーション結果, by default None
        """
        self.depth = depth
        self.color = color
        self.gps_from = gps_from
        self.gps_to = gps_to
        self.seg = seg


def create_pcd(args):
    """点群を作成する
    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数

    Notes
    ----------
    録画のRGB情報とDepth情報(+セグメンテーション結果)とGPS情報それぞれを保存したhdf5ファイルが必要.

    """
    hdf5_path = Path("data", "hdf5", args.site)
    color_path = hdf5_path / Path("color.hdf5")
    depth_path = hdf5_path / Path("depth.hdf5")
    gps_path = hdf5_path / Path("gps.hdf5")
    if args.with_seg:
        seg_path = hdf5_path / Path("seg.hdf5")

    with ExitStack() as stack:
        # 保存するデータに対応するhdf5ファイルを開く
        fc = stack.enter_context(h5py.File(str(color_path), "r"))
        fd = stack.enter_context(h5py.File(str(depth_path), "r"))
        fg = stack.enter_context(h5py.File(str(gps_path), "r"))
        if args.with_seg:
            fs = stack.enter_context(h5py.File(str(seg_path), "r"))

        if args.all is True:  # 分割処理した後のでーたを処理する場合
            routes = [i for i in fg.keys() if args.date[0] + "_" in i]
        else:
            routes = args.date

        output_with_color("creating point cloud")
        for route in tqdm(routes, desc="whole"):
            points = []
            colors = []
            color_group = fc[route]
            depth_group = fd[route]
            gps_group = fg[route]
            if args.with_seg:
                seg_group = fs[route]

            frame_keys = list(map(int, fg[route].keys()))
            frame_keys.sort()
            iter_idx = iter(range(len(frame_keys)))
            for f in tqdm(iter_idx, desc=route, leave=False):
                # 進行方向を求めるために2つの連続したフレームのGPS情報を解析する
                try:
                    gps_from = parse_x_y_from_gps(gps_group[str(frame_keys[f])])
                    gps_to = parse_x_y_from_gps(gps_group[str(frame_keys[f + 1])])
                except IndexError:  # 最後のフレームは方向を決められないので削除
                    break

                # gpsデータが取得できていなかった場合スキップ
                if None in gps_from[0:2] + gps_to[0:2]:
                    continue

                # sectionの切り替わりの位置だった場合スキップ
                if frame_keys[f + 1] - frame_keys[f] != 1:
                    continue

                # gps座標が同じ場合進行方向がわからないので、違う位置を指すまでイテレータを進める
                new_idx = 0
                if gps_from[0:2] == gps_to[0:2]:
                    skip = 1
                    while True:
                        new_idx = f + 1 + skip
                        try:
                            gps_to = parse_x_y_from_gps(gps_group[str(frame_keys[new_idx])])
                        except IndexError:
                            break
                        if gps_from[0:2] != gps_to[0:2]:
                            break
                        skip += 1
                        next(iter_idx)

                # イテレータを進める間に最後のフレームに到達した場合は終了
                if new_idx == len(frame_keys):
                    break

                # frameオブジェクトを作成する
                color_frame = array_to_3dim(color_group[str(frame_keys[f])])
                depth_frame = array_to_3dim(depth_group[str(frame_keys[f])])
                frame = Frame(depth_frame, color_frame, gps_from, gps_to)

                if args.with_seg:
                    frame.seg = array_to_3dim(seg_group[str(frame_keys[f])])

                # 点群の座標と色を取得する
                if args.front:
                    point, color = _create_pcd_from_frame(
                        frame, front=True, voxel=args.voxel
                    )
                else:
                    point, color = _create_pcd_from_frame(frame, voxel=args.voxel)
                points.append(point)
                colors.append(color)

                # TEST: テスト用点群生成
                # if 100 <= f <= 110:
                #     pcd = o3d.geometry.PointCloud()
                #     point[:, 0] += 14900
                #     point[:, 1] += 39000
                #     pcd.points = o3d.utility.Vector3dVector(point)
                #     pcd.colors = o3d.utility.Vector3dVector(color)
                #     pcd.estimate_normals(fast_normal_computation=False)
                #     file_path = Path("src", "test", "binary", f"{f}.pcd")
                #     o3d.io.write_point_cloud(str(file_path), pcd)
                #     if f == 110:
                #         exit()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0))
            pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
            # ルートのフォルダに結果を保存する
            route_path = Path("data", "pts", args.site, route)
            route_path.mkdir(parents=True, exist_ok=True)
            if args.with_seg:
                file_path = route_path / Path("segmentation.pts")
            else:
                file_path = route_path / Path("original.pts")
            o3d.io.write_point_cloud(str(file_path), pcd)


def _create_pcd_from_frame(frame, front=False, voxel=0.1, vis=False, rev=12):
    """1フレームをもとに点群を作成する

    Parameters
    ----------
    frame : Frame
        Frameオブジェクト
    front : bool, optional
        正面で撮影したかどうか, by default False
    voxel : int or float, optional
        ダウンサンプリングに用いるボクセルのサイズ, by default 0.1
    vis : bool, optional
        作成した点群を確認するかどうか, by default False
    rev : int, optional
        カメラの傾きを補正する(度数), by default 12

    Returns
    -------
    (pts, colors) : (ndarray, ndarray)
        点群の座標と色
    """
    x, y, _, ht = frame.gps_from
    x_to, y_to, _, _ = frame.gps_to

    # depth情報とrgbから点群を作る
    # segは緑の部分が1、それ以外が0の配列
    # 掛け算をすることで緑以外の部分は0となり、点群を生成する際に無視される
    if frame.seg is None:
        depth = o3d.geometry.Image(frame.depth)
    else:
        depth = o3d.geometry.Image(frame.depth * frame.seg)

    color = o3d.geometry.Image(frame.color)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=10, convert_rgb_to_intensity=False, depth_scale=1000
    )

    # x軸方向が北として平面座標上に点群を配置する
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, CameraIntrinsic().o3d())
    # FIXME: frame.colorをいじるとエラー
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[:, [2, 1, 0]])
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    colors = np.asarray(pcd.colors)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 上下反転

    # direは北から時計回りで表現された進行方向の角度
    # -90で点群の正面を北(x軸方向)に合わせている
    dire = calc_angle_between_axis(np.array([x_to - x, y_to - y], dtype="float64"))
    if dire == np.nan:
        print("nan")
        exit()

    if front:
        pcd = pcd.rotate([0, math.radians(-90 - math.degrees(dire)), 0], center=False)
    else:
        # pcd = pcd.rotate([0, 0, math.pi])
        pcd = pcd.rotate(
            [0, -math.pi - dire, math.pi / 2 + math.radians(rev)], center=False
        )

    pts = np.asarray(pcd.points)
    pts = pts[:, [2, 0, 1]]  # open3dはy-up,pyplotはz-upの座標なのでyとzを入れ替えておく
    pts += np.array([x, y, ht])

    # vis = True
    # TEST: 点群確認用
    if vis:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([pcd, mesh_frame])

    return pts, colors


def cluster_pcd(args):
    """点群をクラスタリングする

    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    output_with_color("start clustering")
    site_path = Path("data", "pts", args.site)
    dates = [i.name for i in site_path.iterdir()]

    for route in tqdm(dates):
        # 点群のロード
        load_path = site_path / Path(route, "segmentation.pts")
        load_pcd = o3d.io.read_point_cloud(str(load_path))

        labels = list(load_pcd.cluster_dbscan(0.7, 150))
        pts = np.asarray(load_pcd.points)

        result_pcd = o3d.geometry.PointCloud()
        # クラスタリングでノイズ判定された点を削除する
        noiseless = [pts[i, :] for i in range(pts.shape[0]) if labels[i] != -1]
        result_pcd.points = o3d.utility.Vector3dVector(noiseless)
        # 同クラスタには同じ色を割り当てる
        palette = random_colors(max(labels) + 1)
        colors = [palette[labels[i]] for i in range(pts.shape[0]) if labels[i] != -1]
        result_pcd.colors = o3d.utility.Vector3dVector(colors)

        save_path = site_path / Path(route, "clustering.pts")
        o3d.io.write_point_cloud(str(save_path), result_pcd)

        # ptsファイルの3列目にクラスタのインデックスを書き込む
        label_noiseless = [i for i in labels if i != -1]
        with open(save_path, "rb") as f:
            lines = f.readlines()
            lines[0] = "x y z cluster r g b\r\r\n".encode("utf-8")
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                split_line = line.split()
                split_line[3] = f"{label_noiseless[i - 1]}".encode("utf-8")
                split_line[6] += "\r\r\n".encode("utf-8")
                lines[i] = b" ".join(split_line)
        with open(save_path, "wb") as f:
            f.writelines(lines)

        # start = time()
        # # クラスタリング手法によって分岐
        # if args.method == "dbscan":
        #     # Create DBSCAN algorithm.
        #     dbscan_instance = dbscan(elem_pcd, 0.3, 3)
        #     # Start processing by DBSCAN.
        #     dbscan_instance.process()
        #     # Obtain results of clustering.
        #     clusters = dbscan_instance.get_clusters()
        #     noise = dbscan_instance.get_noise()

        # elif args.method == "kmeans":
        #     # Prepare initial centers using K-Means++ method.
        #     initial_centers = kmeans_plusplus_initializer(elem_pcd, 8).initialize()
        #     # Create instance of K-Means algorithm with prepared centers.
        #     kmeans_instance = kmeans(elem_pcd, initial_centers)
        #     # Run cluster analysis and obtain results.
        #     kmeans_instance.process()
        #     clusters = kmeans_instance.get_clusters()
        # print(f"file : {date}")
        # print(f"clustering time : {time() - start}")

        # # 各クラスタに色を設定する
        # c = random_colors(len(clusters))
        # vis = []
        # for idx, cluster in enumerate(clusters):
        #     p = o3d.geometry.PointCloud()
        #     p.points = o3d.utility.Vector3dVector([elem_pcd[i, :] for i in cluster])
        #     p.paint_uniform_color(c[idx])
        #     vis.append(p)

        # # 同じ日付のフォルダが既に存在していた場合、ファイルを削除する
        # date_folder = site_path / Path(date)
        # if date_folder.exists():
        #     [p.unlink() for p in date_folder.iterdir()]
        # date_folder.mkdir(exist_ok=True)

        # for i, p in enumerate(vis):
        #     cluster_file = date_folder / Path(str(i) + ".pts")
        #     o3d.io.write_point_cloud(str(cluster_file), vis[i])


if __name__ == "__main__":
    # 緯度経度を平面直角座標に変換するためのコード
    transformer = pyproj.Transformer.from_proj(6668, 6677)

    # コマンドライン設定
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # 共通の引数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)

    # createコマンドの動作
    create_parser = subparsers.add_parser("create", parents=[parent_parser])
    create_parser.add_argument("-d", "--date", nargs="*", required=True)
    create_parser.add_argument("--voxel", type=float, default=0.1)
    create_parser.add_argument("--with_seg", action="store_true")
    create_parser.add_argument("--front", action="store_true")
    create_parser.add_argument("--all", action="store_true")
    create_parser.set_defaults(handler=create_pcd)

    # clusterコマンドの動作
    clus_parser = subparsers.add_parser("cluster", parents=[parent_parser])
    clus_parser.add_argument(
        "-m", "--method", choices=["dbscan", "xmeans"], required=True
    )
    clus_parser.set_defaults(handler=cluster_pcd)

    args = parser.parse_args()

    if hasattr(args, "handler"):
        args.handler(args)
