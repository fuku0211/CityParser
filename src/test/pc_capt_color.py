import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

if __name__=="__main__":
    align = rs.align(rs.stream.color)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(depth_scale)
    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

    while True:
        frames = pipeline.wait_for_frames()
        # 画角調整(d415はいらない？)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('color image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        # print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)


        if cv2.waitKey(1) != -1:
            print('finish')
            break

    depth_frame = aligned_frames.get_depth_frame()

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)

    depth_frame = spatial.process(depth_frame)

    # depth=0と測定された点(欠損)
    d = np.asarray(depth_frame.get_data())
    print(np.count_nonzero(d == 0))
    depth = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))

    # colorとdepthから色付き点群を作成するとき
    color = o3d.geometry.Image(color_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_trunc = 7, convert_rgb_to_intensity = False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    # depthのみで点群を作成する時
    # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, pinhole_camera_intrinsic, depth_trunc=1000)

    # 左右上下反転？
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    print(np.asarray(pcd.points).shape)

    pipeline.stop()
    o3d.io.write_point_cloud('./pc_color.pcd', pcd)
    o3d.visualization.draw_geometries([pcd])