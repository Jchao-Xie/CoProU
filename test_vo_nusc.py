import torch

from imageio import imread, imsave
# from skimage.transform import resize as imresize
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os

from inverse_warp import pose_vec2mat
from scipy.ndimage.interpolation import zoom

from inverse_warp import *

import models
from utils import tensor2array

import cv2

from datasets.nuscenes_config.splits import val as validation_list

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=False, type=str, help="pretrained PoseNet path")
parser.add_argument("--pretrained-model", required=False, type=str, help="pretrained Model (including PoseNet and DispNet) path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

parser.add_argument("--test", action='store_true', help="using test dataset")

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
parser.add_argument("--interval", type=int, default=1, help="define the interval of target- and reference image")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(filename, args):
    img = Image.open(filename).convert("RGB")
    img = np.array(img).astype(np.float32)  # Convert to NumPy array with float32

    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        # Resize the image using Pillow
        img = Image.fromarray(img.astype(np.uint8))  # Convert back to Pillow Image
        img = img.resize((args.img_width, args.img_height), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32)  # Convert back to NumPy array
    img = np.transpose(img, (2, 0, 1))  # Rearrange dimensions
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()

    if args.pretrained_posenet:
        weights_pose = torch.load(args.pretrained_posenet)
        pose_net = models.PoseResNet().to(device)
        pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
        pose_net.eval()
    elif args.pretrained_model:
        pose_net = models.PoseResNet().to(device)
        print(f"Loading combined model from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Split PoseNet and DispNet weights from global Lightning state_dict
        pose_weights = {k.replace("pose_net.", ""): v for k, v in state_dict.items() if k.startswith("pose_net.")}

        pose_net.load_state_dict(pose_weights, strict=False)
        pose_net.eval()
    else:
        raise ValueError("You must provide either --pretrained-model or both --pretrained-posenet and --pretrained-dispnet.")

    # Find the index of "checkpoints" in the parts of the path
    parts = Path(args.pretrained_posenet if args.pretrained_posenet else args.pretrained_model).parts
    start_index = parts.index("checkpoints")

    # Build the relative path from "checkpoints" onward, and remove .pth.tar suffix
    relative_path = Path(*parts[start_index:]).with_suffix('')  # removes .tar
    relative_path = relative_path.with_suffix('')                # removes .pth

    # Define output_dir
    output_dir = Path(args.output_dir)

    # Join with output_dir
    output_dir = output_dir / relative_path if not args.test else output_dir / relative_path / Path("test")
    
    if args.interval > 1:
        output_dir = output_dir / Path("interval_" + str(args.interval))
    
    gt_dir = "/home/stud/xiji/SC-Depth_Anything/nusc_eval/gt_poses/"
    
    val_scenes = sorted([
                        os.path.join(args.dataset_dir, folder.name)
                        for folder in Path(args.dataset_dir).iterdir()
                        if folder.is_dir() and folder.name.endswith("_0") and folder.name.split("_")[0] in validation_list
                    ])[:70]
    
    test_scenes = sorted([
                        os.path.join(args.dataset_dir, folder.name)
                        for folder in Path(args.dataset_dir).iterdir()
                        if folder.is_dir() and folder.name.endswith("_0") and folder.name.split("_")[0] in validation_list
                    ])[70:]
    scenes = val_scenes if not args.test else test_scenes
    
    for scene in tqdm(scenes):
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gt_poses_path = Path(scene) / 'poses.txt'
        gt_poses = np.loadtxt(gt_poses_path)

        test_files = sorted(Path(scene).glob("*.jpg"))
        
        if args.interval > 1:
            test_files = test_files[::args.interval]
            gt_poses = gt_poses[::args.interval]

        # print('{} files to test'.format(len(test_files)))

        global_pose = np.eye(4)
        poses = [global_pose[0:3, :].reshape(1, 12)]

        n = len(test_files)
        tensor_img1 = load_tensor_image(test_files[0], args)

        for iter in range(n - 1):

            tensor_img2 = load_tensor_image(test_files[iter+1], args)

            pose = pose_net(tensor_img1, tensor_img2)

            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)

            poses.append(global_pose[0:3, :].reshape(1, 12))

            # update
            tensor_img1 = tensor_img2

        poses = np.concatenate(poses, axis=0)
        scene_path = Path(scene)
        filename = output_dir / (scene_path.stem + ".txt")
        np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
        # np.savetxt(Path(gt_dir) / (scene_path.stem + ".txt"), gt_poses, delimiter=' ', fmt='%1.8e')
        


if __name__ == '__main__':
    main()
