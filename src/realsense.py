import argparse
import concurrent.futures
import datetime
from contextlib import ExitStack
from operator import itemgetter
from pathlib import Path
from time import sleep, time

import cv2
import h5py
import numpy as np
import serial
from PIL import Image, ImageDraw, ImageFont

from geometry.capture import RealsenseCapture
from utils.tool import array_to_3dim


# ストリーム
def stream_realsense():
    cap = RealsenseCapture()
    cap.start()
    while True:
        frames = cap.read()
        color_frame = frames[0]
        depth_frame = frames[1]

        # ヒートマップに変換
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET
        )

        # レンダリング
        images = np.hstack((color_frame, depth_colormap))
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)
        key = cv2.waitKey(1) & 0xFF

        # qキーを押したら終了
        if key == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            exit()


# 録画した動画を再生する
def play_record(site, date):
    color_path = Path("data", "hdf5", site, "color.hdf5")
    depth_path = Path("data", "hdf5", site, "depth.hdf5")

    with h5py.File(str(color_path), "a") as fc, h5py.File(str(depth_path), "a") as fd:
        color_group = fc[date]
        depth_group = fd[date]

        for i in range(len(color_group)):
            # ベクトル化したデータをもとの配列の形に戻す
            color_frame = array_to_3dim(color_group[str(i)])
            depth_frame = array_to_3dim(depth_group[str(i)])

            # デプスマップをヒートマップに変換
            frame_heatmap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET
            )

            # 画像を横に並べる
            images = np.hstack((color_frame, frame_heatmap))
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

            # 画像の左上にフレーム数表示
            images = Image.fromarray(images)
            draw = ImageDraw.Draw(images)
            fnt = ImageFont.truetype("arial.ttf", 32)
            draw.text((10, 10), str(i), font=fnt, fill=(255, 0, 0, 255))

            # リサイズしてcv2用に色変換する
            images = images.resize((int(images.width / 2), int(images.height / 2)))
            images = np.asarray(images)
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", images)

            key = cv2.waitKey(1) & 0xFF
            # qキーを押したら終了
            if key == ord("q"):
                print(i)
                exit()


def _extract_gps_data(ser):
    global gps_data
    global gps_frag

    frag_rmc, frag_vtg, frag_gga = [False, False, False]
    i = 0
    while gps_frag:
        log = ser.readline()
        if b"$GNRMC" in log:
            rmc = log.decode().split(",")
            frag_rmc = True

        elif b"$GNVTG" in log:
            vtg = log.decode().split(",")
            frag_vtg = True

        elif b"$GNGGA" in log:
            gga = log.decode().split(",")
            frag_gga = True

        def _fill_empty(data):
            if data == "":
                return "-1"
            else:
                return data

        if all([frag_rmc, frag_vtg, frag_gga]):
            result = rmc + vtg + gga
            result = list(map(_fill_empty, result))
            gps_data = np.array(result, dtype=h5py.special_dtype(vlen=str))
            # print(f"raw : {gps_data}")
            frag_rmc, frag_vtg, frag_gga = [False, False, False]

            i += 1


def _create_zfill_time(time_now):
    """ソートしやすいようゼロ埋めした日時を返却する

    Parameters
    ----------
    time_now : datetime.datetime
        現在日時

    Returns
    -------
    string
        yymmdd_hhmmss
    """
    y = time_now.year
    mo = str(time_now.month).zfill(2)
    d = str(time_now.day).zfill(2)
    h = str(time_now.hour).zfill(2)
    mi = str(time_now.minute).zfill(2)
    se = str(time_now.second).zfill(2)
    return f"{y}{mo}{d}_{h}{mi}{se}"


if __name__ == "__main__":
    # コマンドライン用
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["stream", "record", "play"],
        default="stream",
        help="execution mode",
    )
    parser.add_argument("--site")
    parser.add_argument("--date")
    args = parser.parse_args()

    # ストリーム再生
    if args.mode == "stream":
        stream_realsense()

    # 録画を再生
    elif args.mode == "play":
        play_record(args.site, args.date)

    # 録画
    gps_data = None
    gps_frag = True
    if args.mode == "record":
        with serial.Serial("COM6", 230400) as ser:  # GPSのポート指定
            # 録画と並行してgpsの情報取得をする
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            executor.submit(_extract_gps_data, ser)

            site_path = Path("data", "hdf5", args.site)
            color_path = site_path / Path("color.hdf5")
            depth_path = site_path / Path("depth.hdf5")
            gps_path = site_path / Path("gps.hdf5")

            t_record = time()
            t_now = datetime.datetime.now()
            cap = RealsenseCapture()
            cap.start()

            with ExitStack() as stack:
                # 保存するデータに対応するhdf5ファイルを開く
                fc = stack.enter_context(h5py.File(str(color_path), "a"))
                fd = stack.enter_context(h5py.File(str(depth_path), "a"))
                fg = stack.enter_context(h5py.File(str(gps_path), "a"))

                color_group = fc.create_group(_create_zfill_time(t_now))
                depth_group = fd.create_group(_create_zfill_time(t_now))
                gps_group = fg.create_group(_create_zfill_time(t_now))

                try:
                    # 各フレームに対する操作
                    frame = 0
                    while True:
                        if gps_data is not None:
                            # start = time()
                            sleep(0.2)  # これを挟まないと処理速度が足りずgpsの時間がずれていく
                            frames = cap.read()
                            print(itemgetter(1, 3, 5)(gps_data))
                            gps_group.create_dataset(
                                str(frame), data=gps_data, compression="gzip"
                            )
                            color_group.create_dataset(
                                str(frame), data=frames[0].ravel(), compression="gzip"
                            )
                            depth_group.create_dataset(
                                str(frame), data=frames[1].ravel(), compression="gzip"
                            )
                            frame += 1
                            if frame == 28800:
                                raise KeyboardInterrupt
                            # print(time() - start)

                # # ctrl+c でKeyboardInterruptを呼べるのを利用して終了時の動作を指定する
                except KeyboardInterrupt:
                    print("record time : {}".format(str(time() - t_record)))
                    gps_frag = False
                    executor.shutdown(False)
                    exit()
