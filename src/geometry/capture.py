import pyrealsense2 as rs
import numpy as np
import open3d as o3d

WIDTH = 1280
HEIGHT = 720
FPS = 30


class CameraIntrinsic:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.fx = 925.41943359375
        self.fy = 924.6876220703125
        self.cx = 638.5858764648438
        self.cy = 368.45904541015625

    def o3d(self):
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )


class RealsenseCapture:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.fps = FPS
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )

    def start(self):
        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)

        # 画角調整に必要
        self.align = rs.align(rs.stream.color)

        # 深度マップにSpatial Filterをかける設定
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        self.spatial.set_option(rs.option.holes_fill, 3)

    def read(self, is_array=True):
        # フレームからRGBとDepthを取り出す
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.color_frame = aligned_frames.get_color_frame()  # RGB
        self.depth_frame = aligned_frames.get_depth_frame()  # Depth

        if not self.color_frame or not self.depth_frame:
            return (None, None)
        elif is_array:
            color_image = np.array(self.color_frame.get_data())
            # Depthの方にはフィルタで補完を効かせる
            filtered_depth = self.spatial.process(self.depth_frame)
            depth_image = np.array(filtered_depth.get_data())
            return (color_image, depth_image)
        else:
            return (self.color_frame, self.depth_frame)

    def release(self):
        # ストリームを中止
        self.pipeline.stop()
