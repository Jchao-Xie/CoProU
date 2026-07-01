import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np

from vggt_training.data.dataset_util import *
from vggt_training.data.base_dataset import BaseDataset

from vggt_training.data.datasets.nusc_splits import train as train_split, val as val_split

from ..coprou_transforms import ratio_crop, random_horizontal_flip, resize_images
from models.vggt.utils.load_fn import load_and_preprocess_images


class nuScenesDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        nuscenes_DIR: str = "storage/nuscenes_original_size",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_rate: int = 1,
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            nuscenes_DIR (str): Directory path to nuscenes data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_rate = expand_rate
        self.nuscenes_DIR = nuscenes_DIR
        self.min_num_images = min_num_images
        self.split = split

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"nuscenes_DIR is {self.nuscenes_DIR}")

        # Load or generate sequence list
        txt_path = osp.join(self.nuscenes_DIR, "sequence_list.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # Generate sequence list and save to txt            
            sequence_list = glob.glob(osp.join(self.nuscenes_DIR, "*_0"))            
            sequence_list = [file_path.split(self.nuscenes_DIR)[-1].lstrip('/') for file_path in sequence_list]
            sequence_list = sorted(sequence_list)

            # Save to txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(sequence_list))


        if self.training:
            sequence_list = [sequence_name for sequence_name in sequence_list if sequence_name.split("_")[0] in train_split]
        else:
            sequence_list = [sequence_name for sequence_name in sequence_list if sequence_name.split("_")[0] in val_split]
        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)
        
         # --- Precompute sorted .jpg files per sequence ---
        self.seq_to_images = {}
        for seq_name in self.sequence_list:
            seq_dir = osp.join(self.nuscenes_DIR, seq_name)
            if not osp.isdir(seq_dir):
                raise RuntimeError(f"Missing directory: {seq_dir}")

            jpg_files = [
                osp.join(seq_dir, f)
                for f in os.listdir(seq_dir)
                if f.lower().endswith(".jpg")
            ]
            # Sort lexicographically (or numerically if frame indices)
            jpg_files = sorted(jpg_files)

            self.seq_to_images[seq_name] = jpg_files

        logging.info(f"Loaded {len(self.seq_to_images)} sequences with pre-cached image lists.")

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: nuScenes Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: nuScenes Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        camera_id = int(seq_name[-1])

        # Load camera parameters
        try:
            camera_parameters = np.loadtxt(
                osp.join(self.nuscenes_DIR, seq_name, "poses.txt"), 
                delimiter=" ", 
                skiprows=1
            )
            intrinsic = np.loadtxt(
                osp.join(self.nuscenes_DIR, seq_name, "intrinsics.txt"), 
            )
        except Exception as e:
            logging.error(f"Error loading camera parameters for {seq_name}: {e}")
            raise

        num_images = len(camera_parameters)

        ids = self.get_sequence_ids(num_images, img_per_seq, self.expand_rate)
        images = [
            cv2.cvtColor(
                cv2.imread(self.seq_to_images[seq_name][i], cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            for i in ids
        ]

        if self.training:
            target_image_shape = self.get_target_shape(aspect_ratio)

            images, intrinsic = random_horizontal_flip(images, intrinsic)
            images, intrinsic = ratio_crop(images, intrinsic, aspect_ratio)
            images, intrinsic = resize_images(images, intrinsic, target_image_shape)
        else:
            in_h, in_w, _ = images[0].shape
            aspect_ratio = in_h / in_w
            target_image_shape = self.get_target_shape(aspect_ratio)
            images, intrinsic = resize_images(images, intrinsic, target_image_shape)

        set_name = "nuscenes"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "intrinsics": intrinsic,
        }
        return batch