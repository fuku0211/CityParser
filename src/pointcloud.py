import argparse
import math
from contextlib import ExitStack
from pathlib import Path
from pprint import pprint

import h5py
import numpy as np
import open3d as o3d
from inputimeout import TimeoutOccurred, inputimeout
from tqdm import tqdm

from geometry.capture import CameraIntrinsic
from utils.color_output import output_with_color
from utils.tool import (
    array_to_3dim,
    calc_angle_between_axis,
    random_colors,
)
from gps.convert import parse_x_y_from_gps

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

        routes = [i for i in list(fg.keys()) if len(i.split("_")) == 3]

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
            pbar = tqdm(total=len(frame_keys), desc=route, leave=False)
            for f in iter_idx:
                # 進行方向を求めるために2つの連続したフレームのGPS情報を解析する
                try:
                    gps_from = parse_x_y_from_gps(gps_group[str(frame_keys[f])])
                    gps_to = parse_x_y_from_gps(gps_group[str(frame_keys[f + 1])])
                except IndexError:  # 最後のフレームは方向を決められないので削除
                    pbar.update(1)
                    break

                # gpsデータが取得できていなかった場合スキップ
                if None in gps_from[0:2] + gps_to[0:2]:
                    pbar.update(1)
                    continue

                # gps座標が同じ場合進行方向がわからないので、違う位置を指すまでイテレータを進める
                new_idx = 0
                if gps_from[0:2] == gps_to[0:2]:
                    skip = 1
                    while True:
                        new_idx = f + 1 + skip
                        try:
                            gps_to = parse_x_y_from_gps(
                                gps_group[str(frame_keys[new_idx])]
                            )
                        except IndexError:  # イテレータをすすめる間に最後のフレームに達した場合
                            break
                        if gps_from[0:2] != gps_to[0:2]:
                            break
                        skip += 1
                        pbar.update(1)
                        next(iter_idx)

                # イテレータを進める間に最後のフレームに到達した場合は終了
                if new_idx == len(frame_keys):
                    pbar.update(1)
                    break

                # frameオブジェクトを作成する
                color_frame = array_to_3dim(color_group[str(frame_keys[f])])
                depth_frame = array_to_3dim(depth_group[str(frame_keys[f])])
                frame = Frame(depth_frame, color_frame, gps_from, gps_to)

                if args.with_seg:
                    frame.seg = array_to_3dim(seg_group[str(frame_keys[f])])

                # 点群の座標と色を取得する
                point, color = _create_pcd_from_frame(
                    frame, trunc=args.trunc, voxel=args.voxel, rev=args.rev
                )
                points.append(point)
                colors.append(color)
                pbar.update(1)

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
                file_path = route_path / Path(
                    f"segmentation_t={args.trunc}_v={args.voxel}.pts"
                )
            else:
                file_path = route_path / Path(
                    f"original_t={args.trunc}_v={args.voxel}.pts"
                )
            o3d.io.write_point_cloud(str(file_path), pcd)
            pbar.close()


def _create_pcd_from_frame(frame, trunc, voxel, rev=0, vis=False):
    """1フレームをもとに点群を作成する

    Parameters
    ----------
    frame : Frame
        Frameオブジェクト
    trunc : int or float
        デプス情報を切り捨てる距離
    voxel : int or float
        ダウンサンプリングに用いるボクセルのサイズ
    rev : int, optional
        カメラの傾きを補正する(度数) by default 0
    vis : bool, optional
        作成した点群を確認するかどうか, by default False

    Returns
    -------
    pts, colors : ndarray, ndarray
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
        color,
        depth,
        depth_trunc=trunc,
        convert_rgb_to_intensity=False,
        depth_scale=1000,
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


def _load_setting(file_path):
    setting = file_path.stem.split("_")[1:]
    for s in setting:
        if "t=" in s:
            trunc = int(s.split("=")[-1])
        elif "v=" in s:
            voxel = float(s.split("=")[-1])
    return trunc, voxel


def cluster_pcd(args):
    """点群をクラスタリングする

    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    site_path = Path("data", "pts", args.site)
    routes = [i.name for i in site_path.iterdir() if i.is_dir()]
    # 点群設定の一覧を出力する
    route_path = site_path / routes[0]
    file_names = [i.name for i in route_path.glob("segmentation_*.pts")]
    pprint(list(zip(range(len(file_names)), file_names)))

    # 入力した数値を元に設定を選択する
    try:
        file_name = file_names[
            int(inputimeout(prompt="select setting by index >"), timeout=3)
        ]
    except TimeoutOccurred:
        print("timeout")
        file_name = file_names[args.setting]
    print(f"selected >{file_name}")
    coords = []

    # 点群のロード
    output_with_color("loading pointcloud", "g")
    for route in tqdm(routes):
        load_path = site_path / Path(route, file_name)
        load_pcd = o3d.io.read_point_cloud(str(load_path))
        coords.append(np.asarray(load_pcd.points))
    coords = np.concatenate(coords)

    # クラスタリング
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(coords)
    labels = list(
        all_pcd.cluster_dbscan(args.radius, args.min_pts, print_progress=True)
    )

    # クラスタリングでノイズ判定された点を削除する
    noiseless = [coords[i, :] for i in range(coords.shape[0]) if labels[i] != -1]
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(noiseless)
    # 同クラスタには同じ色を割り当てる
    palette = random_colors(max(labels) + 1)
    colors = [palette[labels[i]] for i in range(coords.shape[0]) if labels[i] != -1]
    result_pcd.colors = o3d.utility.Vector3dVector(colors)

    setting = file_name.split(".")[0].split("_")[1:]
    save_path = site_path / Path(
        f"clustering_{setting[0]}_{setting[1]}_r={args.radius}_m={args.min_pts}.pts",
    )
    output_with_color("writing cluster")
    o3d.io.write_point_cloud(str(save_path), result_pcd)

    output_with_color("writing cluster index")
    # ptsファイルの3列目にクラスタのインデックスを書き込む
    label_noiseless = [i for i in labels if i != -1]
    with open(save_path, "rb") as f:
        lines = f.readlines()
        lines[0] = "x y z cluster r g b\r\r\n".encode("utf-8")
        for i, line in enumerate(tqdm(lines)):
            if i == 0:
                continue
            split_line = line.split()
            split_line[3] = f"{label_noiseless[i - 1]}".encode("utf-8")
            split_line[6] += "\r\r\n".encode("utf-8")
            lines[i] = b" ".join(split_line)
    with open(save_path, "wb") as f:
        f.writelines(lines)


if __name__ == "__main__":
    # コマンドライン設定
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # 共通の引数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)

    # createコマンドの動作
    create_parser = subparsers.add_parser("create", parents=[parent_parser])
    create_parser.add_argument("--voxel", type=float, default=0.1)
    create_parser.add_argument("--trunc", type=int, default=10)
    create_parser.add_argument("--rev", type=int, default=0)
    create_parser.add_argument("--with_seg", action="store_true")
    create_parser.set_defaults(handler=create_pcd)

    # clusterコマンドの動作
    clus_parser = subparsers.add_parser("cluster", parents=[parent_parser])
    clus_parser.add_argument("--radius", type=float, default=0.7)
    clus_parser.add_argument("--min_pts", type=int, default=150)
    clus_parser.add_argument("--setting", type=int)
    clus_parser.set_defaults(handler=cluster_pcd)

    args = parser.parse_args()

    if hasattr(args, "handler"):
        args.handler(args)
