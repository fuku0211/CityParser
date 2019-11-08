import argparse
import math
from contextlib import ExitStack
from operator import itemgetter
from pathlib import Path
from time import time

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numba
import numpy as np
import open3d as o3d
import pyproj
import shapefile
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.samples.definitions import FCPS_SAMPLES
from tqdm import tqdm, trange
from utils.tool import array_to_3dim, calc_gap_between_yaxis_and_vector, random_colors

# 緯度経度を平面直角座標に変換するためのコード
transformer = pyproj.Transformer.from_proj(6668, 6677)


def _parse_gps_data(gpsdata):
    # TODO: 標高=楕円体高であってるかわからない
    lat, lon, dire, ht = map(float, itemgetter(3, 5, 8, 33)(gpsdata))
    lat /= 100
    lon /= 100
    y, x = transformer.transform(lat, lon)
    return (x, y, dire, ht)


def _get_pts_from_hdf5(depth, color, gps, seg=None, front=True, voxel=0.03):

    x, y, dire, ht = gps
    if seg is None:
        depth = o3d.geometry.Image(depth)
    else:
        # segは緑の部分が1、それ以外が0の配列
        # 掛け算をすることで緑以外の部分は0となり、点群を生成する際に無視される
        depth = o3d.geometry.Image(depth * seg)
    color = o3d.geometry.Image(color_frame)
    # depth情報とrgbから点群を作る
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=5, convert_rgb_to_intensity=False, depth_scale=1000
    )

    # x軸方向が北として平面座標上に点群を配置する
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    colors = np.asarray(pcd.colors)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 上下反転
    # direは北から時計回りで表現された進行方向の角度
    # -90で点群の正面を北(x軸)に合わせている
    pcd = pcd.rotate([0, math.radians(-90 - dire), 0], center=False)
    # TODO: ちゃんと動作するか確認
    if not front:
        pcd = pcd.rotate([0, math.radians(-90), 0], center=False)  # 上を向くようにy軸で回転

    pts = np.asarray(pcd.points)
    pts = pts[:, [2, 0, 1]]  # open3dはy-up,pyplotはz-upの座標なのでyとzを入れ替えておく
    pts += np.array([x, y, 0])

    # TEST: 点群確認用
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1, origin=[0, 0, 0]
    # )
    # o3d.visualization.draw_geometries([pcd, mesh_frame])

    return (pts, colors)


if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["create"])
    parser.add_argument("-s", "--site")
    parser.add_argument("-f", "--date_front", nargs="*")
    parser.add_argument("-u", "--date_up", nargs="*")
    parser.add_argument("--color_only", action="store_false")
    args = parser.parse_args()

    site_path = Path("data", "hdf5", args.site)
    color_path = site_path / Path("color.hdf5")
    depth_path = site_path / Path("depth.hdf5")
    gps_path = site_path / Path("gps.hdf5")
    seg_path = site_path / Path("seg.hdf5")

    if args.mode == "create":
        # カメラのパラメータを設定 (解像度、焦点距離、光学中心)
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280,
            720,
            925.41943359375,
            924.6876220703125,
            638.5858764648438,
            368.45904541015625,
        )

        points = []
        colors = []
        with ExitStack() as stack:
            # 保存するデータに対応するhdf5ファイルを開く
            fc = stack.enter_context(h5py.File(str(color_path), "r"))
            fd = stack.enter_context(h5py.File(str(depth_path), "r"))
            # fs = stack.enter_context(h5py.File(str(seg_path), "r"))
            fg = stack.enter_context(h5py.File(str(gps_path), "r"))

            for route in tqdm(args.date_front, desc="whole"):
                color_group = fc[route]
                depth_group = fd[route]
                # seg_group = fs[route]
                gps_group = fg[route]

                frame_count = len(color_group.keys())
                for f in trange(0, frame_count, 2, desc=f"route : {route}"):
                    color_frame = array_to_3dim(color_group[str(f)])
                    depth_frame = array_to_3dim(depth_group[str(f)])
                    # seg_frame = array_to_3dim(seg_group[str(f)].value)
                    # t = time()
                    gps_data = _parse_gps_data(gps_group[str(f)])
                    # print(timw3q0e() - t)
                    seg_frame = None
                    if args.color_only:
                        point, color = _get_pts_from_hdf5(
                            depth_frame, color_frame, gps_data
                        )
                    else:
                        point, color = _get_pts_from_hdf5(
                            depth_frame, color_frame, gps_data, seg=seg_frame
                        )
                    points.append(point)
                    colors.append(color)

        # 8.49
        t = time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
        print(time()-t)

        # # Create DBSCAN algorithm.
        # dbscan_instance = dbscan(down_all, 0.3, 3)
        # # Start processing by DBSCAN.
        # dbscan_instance.process()
        # # Obtain results of clustering.
        # clusters = dbscan_instance.get_clusters()
        # noise = dbscan_instance.get_noise()

        # # # Prepare initial centers using K-Means++ method.
        # # initial_centers = kmeans_plusplus_initializer(down_all, 8).initialize()
        # # # Create instance of K-Means algorithm with prepared centers.
        # # kmeans_instance = kmeans(down_all, initial_centers)
        # # # Run cluster analysis and obtain results.
        # # kmeans_instance.process()
        # # clusters = kmeans_instance.get_clusters()

        # # 体積分布
        # # fig = plt.figure()
        # # ax = fig.add_subplot(1,1,1)
        # # mu, sigma = 100, 15
        # # x = mu + sigma * np.random.randn(10000)
        # # a = np.array([len(i)/1000 for i in clusters])
        # # ax.hist(a, bins=60)
        # # plt.show()

        # # 各クラスタに色を設定する
        # c = random_colors(len(clusters))
        # vis = []
        # for idx, cluster in enumerate(clusters):
        #     p = o3d.geometry.PointCloud()
        #     p.points = o3d.utility.Vector3dVector([down_all[i, :] for i in cluster])
        #     p.paint_uniform_color(c[idx])
        #     # a = p.colors
        #     vis.append(p)

        # # o3d.visualization.draw_geometries(vis)
        # for i, v in enumerate(vis):
        #     o3d.io.write_point_cloud(f"./recap/{i}.pts", v)

        #9.27
        o3d.io.write_point_cloud("./test.pts", pcd)
