import argparse
from pathlib import Path
from time import sleep

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from geometry.model import create_model, deeplabv3
from utils.tool import HEIGHT, WIDTH, array_to_3dim, get_gpu_info
import h5py


def __create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

    Returns:
    A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """

    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = __create_cityscapes_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def inference_from_hdf5(args):
    site_path = Path("data", "hdf5", args.site)
    color_path = site_path / Path("color.hdf5")
    seg_path = site_path / Path("seg.hdf5")

    model = create_model(args)
    with h5py.File(str(color_path), "r") as fc, h5py.File(str(seg_path), "a") as fs:
        if args.all is True: # 分割後のルートを処理する場合
            routes = [i for i in fc.keys() if args.date[0] + "_" in i]
        else:
            routes = args.date

        for route in tqdm(routes, desc="all"):
            color_group = fc[route]
            if route in fs.keys():
                del fs[route]
            seg_group = fs.create_group(route)

            for i in tqdm(color_group.keys(), desc=f"route : {route}", leave=False):
                # フレームを読み込んでセグメンテーションできるようにリサイズする
                img = array_to_3dim(color_group[str(i)])
                img = Image.fromarray(img)
                img = img.resize((2048, 1024))
                # セグメンテーションして元のサイズに戻す
                result = deeplabv3(img, model).astype("uint8")
                result = result == 8
                result = result * 255
                result = np.dstack([result, result, result]).astype("uint8")
                result = Image.fromarray(result).resize((WIDTH, HEIGHT))
                result = np.asarray(result)
                tag = result[:, :, 0] + result[:, :, 1] + result[:, :, 2]
                element = tag != 0
                seg_group.create_dataset(
                    str(i), data=element.ravel(), compression="gzip"
                )


def replay_segment(args):
    site_path = Path("data", "hdf5", args.site)
    color_path = site_path / Path("color.hdf5")
    seg_path = site_path / Path("seg.hdf5")

    with h5py.File(str(color_path), "r") as fc, h5py.File(str(seg_path), "r") as fs:
        color_group = fc[args.date]
        seg_group = fs[args.date]
        for i in color_group.keys():
            seg_frame = np.rot90(array_to_3dim(seg_group[str(i)]))
            color_frame = np.rot90(array_to_3dim(color_group[str(i)]))
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

            # boolをintにして*8することで植物の部分が8になるようにする
            seg_frame = seg_frame * 8
            seg_frame = label_to_color_image(seg_frame)
            seg_frame = Image.fromarray(seg_frame)
            color_frame = Image.fromarray(color_frame)
            mix_frame = Image.blend(color_frame, seg_frame, 0.7)
            out_frame = np.hstack((np.asarray(color_frame), np.asarray(mix_frame)))
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("RealSense", out_frame)
            key = cv2.waitKey(1) & 0xFF

            sleep(0.1)

            # qキーを押したら終了
            if key == ord("q"):
                cv2.destroyAllWindows()
                cv2.imwrite("data/frame.jpg", out_frame)
                exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-s", "--site", required=True)

    inf_parser = subparsers.add_parser("inference", parents=[parent_parser])
    inf_parser.add_argument("-d", "--date", required=True, nargs="*")
    inf_parser.add_argument("--model", type=int, default=0)
    inf_parser.add_argument("--all", action="store_true")
    inf_parser.set_defaults(handler=inference_from_hdf5)

    play_parser = subparsers.add_parser("replay", parents=[parent_parser])
    play_parser.add_argument("-d", "--date", required=True)
    play_parser.set_defaults(handler=replay_segment)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
