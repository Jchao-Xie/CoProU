### Modified from https://github.com/fzhiheng/RoGS/blob/main/datasets/nusc.py


import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool

import cv2
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from data.base import BaseDataset

import argparse
import yaml
import addict


def loda_depth(depth_file):
    loaded_data = np.load(depth_file)
    depth_img = sp.csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    return depth_img


def worldpoint2camera(points: np.ndarray, WH, cam2world, cam_intrinsic, min_dist: float = 1.0):
    """
    1. transform world points to camera points
    Args:
        points: (N, 3)
        image:  (H, W, 3)
        cam2world: (4, 4)
        cam_intrinsic: (3, 3)
        min_dist: float

    Returns:
        uv: (2, N)
        depths: (N, )
        mask: (N, )

    """
    width, height = WH
    world2cam = np.linalg.inv(cam2world)  # (4, 4)
    points_cam = world2cam[:3, :3] @ points.T + world2cam[:3, 3:4]  # (3, N)
    depths = points_cam[2, :]  # (N, )
    points_uv1 = view_points(points_cam, np.array(cam_intrinsic), normalize=True)  # (3, N)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points_uv1[0, :] > 1)
    mask = np.logical_and(mask, points_uv1[0, :] < width - 1)
    mask = np.logical_and(mask, points_uv1[1, :] > 1)
    mask = np.logical_and(mask, points_uv1[1, :] < height - 1)

    uv = points_uv1[:, mask][:2, :]
    uv = np.round(uv).astype(np.uint16)
    depths = depths[mask]
    return uv, depths, mask


class NuscDataset(BaseDataset):
    def __init__(self, configs, nusc= None, use_label=False, use_depth=False):
        self.nusc = nusc
        self.version = configs["version"]
        super().__init__()
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.base_dir = configs["base_dir"]
        self.image_dir = configs["image_dir"]
        self.camera_names = configs["camera_names"]
        self.min_distance = configs["min_distance"]
        clip_list = configs["clip_list"]
        self.chassis2world_unique = []
        self.raw_wh = dict()


        self.use_depth = use_depth
        self.lidar_times_all = []
        self.lidar_filenames_all = []
        self.lidar2world_all = []
        self.scene_name_all = []

        # road_pointcloud = dict()
        for scene_name in tqdm(clip_list, desc="Loading data clips"):
            self.scene_name = scene_name
            records = [samp for samp in self.nusc.sample if self.nusc.get("scene", samp["scene_token"])["name"] in scene_name]
            records.sort(key=lambda x: (x['timestamp']))

            print(f"Loading image from scene {scene_name}")
            cam_info, chassis_info = self.load_cameras(records)

            self.raw_wh[scene_name] = cam_info["wh"]
            self.camera2world_all.extend(cam_info["poses"])
            self.camera_times_all.extend(cam_info["times"])
            self.cameras_K_all.extend(cam_info["intrinsics"])
            self.cameras_idx_all.extend(cam_info["idxs"])
            self.image_filenames_all.extend(cam_info["filenames"])
            self.scene_name_all.extend(cam_info["scene_name"])

            self.chassis2world_unique.extend(chassis_info["unique_poses"])
            self.chassis2world_all.extend(chassis_info["poses"])

            # label_filenames = [rel_camera_path.replace("/CAM", "/seg_CAM").replace(".jpg", ".png") for rel_camera_path in cam_info["filenames"]]
            # self.label_filenames_all.extend(label_filenames)

            lidar_info = self.load_lidars(records)
            self.lidar_times_all.extend(lidar_info["times"])
            self.lidar_filenames_all.extend(lidar_info["filenames"])
            self.lidar2world_all.extend(lidar_info["poses"])


        # self.file_check()
        if len(self.image_filenames_all) == 0:
            raise FileNotFoundError("No data found in the dataset")

        self.chassis2world_unique = np.array(self.chassis2world_unique)
        self.chassis2world_all = np.array(self.chassis2world_all)  # [N, 4, 4]
        self.camera2world_all = np.array(self.camera2world_all)  # [N, 4, 4]
        self.camera_times_all = np.array(self.camera_times_all)  # [N, ]

        self.lidar2world_all = np.array(self.lidar2world_all)  # [N, 4, 4]
        self.lidar_times_all = np.array(self.lidar_times_all)  # [N, ]

        self.ref_pose = self.chassis2world_unique[0]
        ref_pose_inv = np.linalg.inv(self.ref_pose)
        self.chassis2world_unique = ref_pose_inv @ self.chassis2world_unique
        self.camera2world_all = ref_pose_inv @ self.camera2world_all
        self.chassis2world_all = ref_pose_inv @ self.chassis2world_all
        self.lidar2world_all = ref_pose_inv @ self.lidar2world_all

        nerf_normalization = self.getNerfppNorm()
        self.cameras_extent = nerf_normalization["radius"]

    def __len__(self):
        return len(self.image_filenames_all)

    def __getitem__(self, idx):
        cam_idx = self.cameras_idx_all[idx]
        cam2world = self.camera2world_all[idx]
        K = self.cameras_K_all[idx]
        scene_name = self.scene_name_all[idx]
        camera_name = self.camera_names[cam_idx]
        image_path = os.path.join(self.base_dir, self.image_filenames_all[idx])
        image_name = os.path.basename(image_path).split(".")[0]
        input_image = cv2.imread(image_path)

        crop_cy = int(0)
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image[crop_cy:, :, :]  # crop the sky
        gt_image = (np.asarray(resized_image) / 255.0).astype(np.float32)  # [H, W, 3]
        gt_image = np.clip(gt_image, 0.0, 1.0)
        width, height = gt_image.shape[1], gt_image.shape[0]

        new_K = deepcopy(K)
        width_scale = self.resized_image_size[0] / origin_image_size[1]
        height_scale = self.resized_image_size[1] / origin_image_size[0]
        new_K[0, :] *= width_scale
        new_K[1, :] *= height_scale
        new_K[1][2] -= crop_cy
        R = cam2world[:3, :3]
        T = cam2world[:3, 3]

        sample = {"image": resized_image, "idx": idx, "cam_idx": cam_idx, "image_name": image_name, "R": R, "T": T, "K": new_K, "W": width, "H": height, "scene_name": scene_name}

        if self.use_depth:
            cam_time = self.camera_times_all[idx]
            lidar_idx = np.argmin(np.abs(self.lidar_times_all - cam_time))
            lidar2world = self.lidar2world_all[lidar_idx]
            lidar_path = os.path.join(self.base_dir, self.lidar_filenames_all[lidar_idx])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            points_world = lidar2world[:3, :3] @ points.T + lidar2world[:3, 3:4]  # (3, N)
            uv, depths, mask = worldpoint2camera(points_world.T, (width, height), cam2world, new_K)
            sort_idx = np.argsort(depths)[::-1]
            uv = uv[:, sort_idx]
            depths = depths[sort_idx]
            depth_image = np.zeros((height, width), dtype=np.float32)
            depth_image[uv[1], uv[0]] = depths
            sample["depth"] = depth_image

        return sample


    def load_lidars(self, records):
        lidar_times = []
        lidar_files = []
        lidar2worlds = []

        for rec in tqdm(records):
            samp = self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])
            flag = True
            while flag or not samp["is_key_frame"]:
                flag = False

                lidar_times.append(samp["timestamp"])
                lidar_files.append(samp["filename"])

                cs_record = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
                lidar2ego = np.eye(4)
                lidar2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
                lidar2ego[:3, 3] = cs_record['translation']

                poserecord = self.nusc.get('ego_pose', samp['ego_pose_token'])
                ego2global = np.eye(4)
                ego2global[:3, :3] = Quaternion(poserecord['rotation']).rotation_matrix
                ego2global[:3, 3] = poserecord['translation']

                lidar2global = ego2global @ lidar2ego
                lidar2worlds.append(lidar2global)

                if samp["next"] != "":
                    samp = self.nusc.get('sample_data', samp["next"])
                else:
                    break

        return {"times": lidar_times, "filenames": lidar_files, "poses": lidar2worlds}

    def load_cameras(self, records):
        chassis2world_unique = []
        chassis2worlds = []
        camera2worlds = []
        cameras_K = []
        cameras_idxs = []
        cameras_times = []
        image_filenames = []
        scene_name = []
        wh = dict()

        # interpolate images from 2HZ to 12 HZ  (sample + sweep)
        for rec in tqdm(records):
            chassis_flag = True
            for camera_idx, cam in enumerate(self.camera_names):
                # compute camera key frame poses
                rec_token = rec["data"][cam]
                samp = self.nusc.get("sample_data", rec_token)
                wh.setdefault(cam, (samp["width"], samp["height"]))
                flag = True
                while flag or not samp["is_key_frame"]:
                    flag = False
                    rel_camera_path = samp["filename"]
                    cameras_times.append(samp["timestamp"])
                    image_filenames.append(rel_camera_path)

                    camera2chassis = self.compute_extrinsic2chassis(samp)
                    c2w = self.compute_chassis2world(samp)
                    chassis2worlds.append(c2w)
                    if chassis_flag:
                        chassis2world_unique.append(c2w)
                    camera2world = c2w @ camera2chassis
                    camera2worlds.append(camera2world.astype(np.float32))

                    calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
                    intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
                    cameras_K.append(intrinsic.astype(np.float32))
                    
                    scene_name.append(self.scene_name)
                    cameras_idxs.append(camera_idx)
                    # not key frames
                    if samp["next"] != "":
                        samp = self.nusc.get('sample_data', samp["next"])
                    else:
                        break
                chassis_flag = False
        cam_info = {"poses": camera2worlds, "intrinsics": cameras_K, "idxs": cameras_idxs, "filenames": image_filenames, "times": cameras_times, "wh": wh, "scene_name":scene_name}
        chassis_info = {"poses": chassis2worlds, "unique_poses": chassis2world_unique}

        return cam_info, chassis_info

    def compute_chassis2world(self, samp):
        """transform sensor in world coordinate"""
        # comput current frame Homogeneous transformation matrix : from chassis 2 global
        pose_chassis2global = self.nusc.get("ego_pose", samp['ego_pose_token'])
        chassis2global = transform_matrix(pose_chassis2global['translation'],
                                          Quaternion(pose_chassis2global['rotation']),
                                          inverse=False)
        return chassis2global

    def compute_extrinsic(self, samp_a, samp_b):
        """transform from sensor_a to sensor_b"""
        sensor_a2chassis = self.compute_extrinsic2chassis(samp_a)
        sensor_b2chassis = self.compute_extrinsic2chassis(samp_b)
        sensor_a2sensor_b = np.linalg.inv(sensor_b2chassis) @ sensor_a2chassis
        return sensor_a2sensor_b

    def compute_extrinsic2chassis(self, samp):
        calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        rot = np.array(Quaternion(calibrated_sensor["rotation"]).rotation_matrix)
        tran = np.expand_dims(np.array(calibrated_sensor["translation"]), axis=0)
        sensor2chassis = np.hstack((rot, tran.T))
        sensor2chassis = np.vstack((sensor2chassis, np.array([[0, 0, 0, 1]])))  # [4, 4] camera 3D
        return sensor2chassis



def get_configs():
    parser = argparse.ArgumentParser(description='G4M config')
    parser.add_argument('--config', default="configs/local_nusc_mini.yaml", help='config yaml path')
    parser.add_argument('--start', default=0, type=int, help="Start index (inclusive)")
    parser.add_argument('--end', default=-1, type=int, help="End index (exclusive)")
    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.safe_load(file)
    configs["file"] = os.path.abspath(args.config)

    return configs

if __name__ == "__main__":
    configs = get_configs()
    configs = addict.Dict(configs)
    dataset_cfg = configs.dataset
    nusc = NuScenes(version="v1.0-{}".format(dataset_cfg["version"]), dataroot=dataset_cfg["base_dir"], verbose=True)
    dataset_cfg["clip_list"] = train_scene_names = [
                                                scene['name']
                                                for scene in nusc.scene
                                                if "night" not in scene['description'].lower()
                                            ]
    dataset = NuscDataset(dataset_cfg, nusc)
    
    curr_scene_cam = None          # e.g. "scene-0001_0"
    pose_fh = None                 # open file handle for poses.txt
    cached_K = None
    root_out = dataset_cfg["data_output_dir"]
    handles = {}            # key → open file handle

    for sample in tqdm(dataset):
        key     = f"{sample['scene_name']}_{sample['cam_idx']}"
        out_dir = os.path.join(root_out, key)
        os.makedirs(out_dir, exist_ok=True)

        # intrinsics (once)
        k_path = os.path.join(out_dir, "intrinsics.txt")
        if not os.path.exists(k_path):
            np.savetxt(k_path, sample["K"], fmt="%.8f")
        elif not np.allclose(np.loadtxt(k_path), sample["K"]):
            raise ValueError(f"K changed inside {key}")

        # save image
        img_path = os.path.join(out_dir, f"{sample['image_name']}.jpg")
        cv2.imwrite(
            img_path,
            cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )

        # append pose (open ➜ write ➜ close)
        pose_path = os.path.join(out_dir, "poses.txt")
        RT = np.hstack([sample["R"], sample["T"].reshape(3, 1)])  # 3×4
        with open(pose_path, "a") as fh:
            fh.write(" ".join(f"{x:.8f}" for x in RT.flatten()) + "\n")
    
