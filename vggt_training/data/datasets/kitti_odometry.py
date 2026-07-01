import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np

from vggt_training.data.dataset_util import *
from vggt_training.data.base_dataset import BaseDataset

from ..coprou_transforms import ratio_crop, random_horizontal_flip, resize_images
from models.vggt.utils.load_fn import load_and_preprocess_images


class KittiOdometryDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        KittiOdometry_DIR: str = "storage/kitti_odometry",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_rate: int = 1,
    ):
        """
        Initialize the KittiOdometryDataset.

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
        self.val_split = ['09', '10', '01']
        
        self.expand_rate = expand_rate
        self.KittiOdometry_DIR = KittiOdometry_DIR
        self.min_num_images = min_num_images
        self.split = split

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"KittiOdometry_DIR is {self.KittiOdometry_DIR}")

        # Load or generate sequence list

        # Generate sequence list and save to txt            
        kitti_segments, sequence_list = self.build_kitti_segments(self.KittiOdometry_DIR, max_len=200)
        sequence_list = sorted(sequence_list)

        # # Save to txt file
        # with open(txt_path, 'w') as f:
        #     f.write('\n'.join(sequence_list))


        if self.training:
            sequence_list = [sequence_name for sequence_name in sequence_list if sequence_name.split("_")[0] not in self.val_split]
        else:
            sequence_list = [sequence_name for sequence_name in sequence_list if sequence_name.split("_")[0] in self.val_split]
        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)
        
         # --- Precompute sorted .jpg files per sequence ---
        self.seq_to_images = kitti_segments


        logging.info(f"Loaded {len(self.sequence_list)} sequences with pre-cached image lists.")

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: nuScenes Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: nuScenes Data dataset length: {len(self)}")
        
    def chunk(self, items, max_len=200):
        return [items[i:i+max_len] for i in range(0, len(items), max_len)]

    def build_kitti_segments(self, kitti_root, seq_ids=range(22), cam_subdir="image_2", ext="png", max_len=200):
        segments = {}
        sequence_list = []
        for sid in seq_ids:
            seq = f"{sid:02d}"
            img_dir = osp.join(kitti_root, seq, cam_subdir)
            frames = sorted(glob.glob(osp.join(img_dir, f"*.{ext}")))
            if not frames:
                continue

            chunks = self.chunk(frames, max_len=max_len)
            for k, ch in enumerate(chunks):
                if len(ch) > 48:
                    segments[f"{seq}_{k:02d}"] = {
                        "sequence": seq,
                        "chunk_idx": k,
                        "frames": ch,                # list of file paths, len <= 200
                    }
                    sequence_list.append(f"{seq}_{k:02d}")
        return segments, sequence_list
    
    def load_kitti_cam2_intrinsic(self, kitti_root, seq_name):
        calib_path = osp.join(
            kitti_root,
            seq_name.split('_')[0],
            "calib.txt"
        )

        with open(calib_path, "r") as f:
            for line in f:
                if line.startswith("P2:"):
                    P2 = np.array(
                        [float(x) for x in line.strip().split()[1:]],
                        dtype=np.float32
                    ).reshape(3, 4)
                    break
            else:
                raise ValueError("P2 not found in calib.txt")

        # Intrinsic matrix K
        K = P2[:3, :3]
        return K, P2

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

        # Load camera parameters
        try:
            intrinsic, _ = self.load_kitti_cam2_intrinsic(self.KittiOdometry_DIR, seq_name.split('_')[0])
        except Exception as e:
            logging.error(f"Error loading camera parameters for {seq_name}: {e}")
            raise

        num_images = len(self.seq_to_images[seq_name]["frames"])

        ids = self.get_sequence_ids(num_images, img_per_seq, self.expand_rate)
        images = [
            cv2.cvtColor(
                cv2.imread(self.seq_to_images[seq_name]["frames"][i], cv2.IMREAD_COLOR),
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

        set_name = "KittiOdometry"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "intrinsics": intrinsic,
        }
        return batch