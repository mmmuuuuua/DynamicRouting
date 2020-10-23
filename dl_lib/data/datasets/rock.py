# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import glob
import json
import logging
import multiprocessing as mp
import os
from itertools import chain

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image

from dl_lib.structures import BoxMode
from dl_lib.utils.comm import get_world_size
from dl_lib.utils.file_io import PathManager
from dl_lib.utils.logger import setup_logger

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


def load_rock_semantic(_image_dir, _cat_dir, _splits_file):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """

    ret = []

    with open(os.path.join(_splits_file), "r") as f:
        lines = f.read().splitlines()

    for ii, line in enumerate(lines):
        _image = os.path.join(_image_dir, line)
        _cat = os.path.join(_cat_dir, line + ".png")

        # assert os.path.isfile(_image)
        # assert os.path.isfile(_cat)
        # assert os.path.isfile(_mask)
        # im_ids.append(line.rstrip('\n'))
        # images.append(_image)
        # categories.append(_cat)
        # masks.append(_mask)

        ret.append({
            "file_name": _image,
            "sem_seg_file_name": _cat,
            "height": 512,
            "width": 512,
        })

    return ret


if __name__ == "__main__":
    """
    Test the cityscapes dataset loader.

    Usage:
        python -m dl_lib.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("gt_dir")
    parser.add_argument("--type",
                        choices=["instance", "semantic"],
                        default="instance")
    args = parser.parse_args()
