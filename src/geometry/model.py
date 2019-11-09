# DeepLab Demo
# This demo will demostrate the steps to run deeplab semantic segmentation model on sample input images.

# @title Imports

import os
import tarfile
import tempfile
from six.moves import urllib

import numpy as np


def create_model():
    import tensorflow as tf

    class DeepLabModel(object):
        """Class to load deeplab model and run inference."""

        INPUT_TENSOR_NAME = "ImageTensor:0"
        OUTPUT_TENSOR_NAME = "SemanticPredictions:0"
        FROZEN_GRAPH_NAME = "frozen_inference_graph"

        def __init__(self, tarball_path):
            """Creates and loads pretrained deeplab model."""
            self.graph = tf.Graph()

            graph_def = None
            # Extract frozen graph from tar archive.
            tar_file = tarfile.open(tarball_path)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.GraphDef.FromString(file_handle.read())
                    break

            tar_file.close()

            if graph_def is None:
                raise RuntimeError("Cannot find inference graph in tar archive.")

            with self.graph.as_default():
                tf.import_graph_def(graph_def, name="")

            self.sess = tf.Session(graph=self.graph)

        def run(self, image):
            """Runs inference on a single image.

            Args:
            image: A PIL.Image object, raw input image.

            Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
            """

            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]},
            )
            seg_map = batch_seg_map[0]
            return seg_map

    MODEL_NAME = "xception71_dpc_cityscapes_trainfine"

    _DOWNLOAD_URL_PREFIX = "http://download.tensorflow.org/models/"
    _MODEL_URLS = {
        "mobilenetv2_coco_voctrainaug": "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz",
        "mobilenetv2_coco_voctrainval": "deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz",
        "xception_coco_voctrainaug": "deeplabv3_pascal_train_aug_2018_01_04.tar.gz",
        "xception_coco_voctrainval": "deeplabv3_pascal_trainval_2018_01_04.tar.gz",
        "mobilenetv2_coco_cityscapes_trainfine": "deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz",
        "xception65_cityscapes_trainfine": "deeplabv3_cityscapes_train_2018_02_06.tar.gz",
        "xception71_dpc_cityscapes_trainfine": "deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz",
        "xception71_dpc_cityscapes_trainval": "deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz",
        "xception65_ade20k_train": "deeplabv3_xception_ade20k_train_2018_05_29.tar.gz",
    }
    _TARBALL_NAME = "deeplab_model.tar.gz"

    model_dir = tempfile.mkdtemp()
    tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    print("downloading model, this might take a while...")
    urllib.request.urlretrieve(
        _DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path
    )
    print("download completed! loading DeepLab model...")

    MODEL = DeepLabModel(download_path)
    print("model loaded successfully!")

    return MODEL


def deeplabv3(image_rgb, model):
    return model.run(image_rgb)
