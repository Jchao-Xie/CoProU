import torch

from imageio import imread, imsave
# from skimage.transform import resize as imresize
from PIL import Image
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from inverse_warp import pose_vec2mat
from scipy.ndimage.interpolation import zoom

from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

from inverse_warp import *

import models
from depth_anything_v2.dpt import fine_tuning_DepthAnythingV2
from utils import tensor2array

import cv2
import imageio

import os
from loss_functions import compute_ssim_loss


from vo_visualization_nusc import depth_anything_transform, load_tensor_image, colorize_map
from loss_functions import compute_pairwise_loss
from utils import unnormalize_and_save, depth_visualization, mask_visualization, error_visualization, heapmap_visualization

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=False, type=str, help="pretrained PoseNet path")
parser.add_argument('--pretrained-dispnet', required=False, help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-model', required=False, help='path to pre-trained model')
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--tgt-img", type=str, required=True, help="target image path")
parser.add_argument("--ref-img", type=str, required=True, help="reference image path")
parser.add_argument('--dataset', type=str, choices=['kitti', 'nuscenes'], required=True, help='the dataset')

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():
    args = parser.parse_args()

    # weights_pose = torch.load(args.pretrained_posenet)
    pose_net = models.PoseResNet().to(device)
    # pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    # pose_net.eval()
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = fine_tuning_DepthAnythingV2(**model_configs[args.encoder])
    disp_net = depth_anything.to(device)
    # weights_disp = torch.load(args.pretrained_dispnet)
    # disp_net.load_state_dict(weights_disp['state_dict'], strict=False)
    # disp_net.eval()
    
    
    if args.pretrained_model:
        print(f"Loading combined model from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Split PoseNet and DispNet weights from global Lightning state_dict
        pose_weights = {k.replace("pose_net.", ""): v for k, v in state_dict.items() if k.startswith("pose_net.")}
        disp_weights = {k.replace("disp_net.", ""): v for k, v in state_dict.items() if k.startswith("disp_net.")}

        pose_net.load_state_dict(pose_weights, strict=False)
        disp_net.load_state_dict(disp_weights, strict=False)

    elif args.pretrained_posenet and args.pretrained_dispnet:
        print(f"Loading PoseNet from {args.pretrained_posenet}")
        pose_ckpt = torch.load(args.pretrained_posenet, map_location=device)
        pose_net.load_state_dict(pose_ckpt["state_dict"], strict=False)

        print(f"Loading DispNet from {args.pretrained_dispnet}")
        disp_ckpt = torch.load(args.pretrained_dispnet, map_location=device)
        disp_net.load_state_dict(disp_ckpt["state_dict"], strict=False)

    else:
        raise ValueError("You must provide either --pretrained-model or both --pretrained-posenet and --pretrained-dispnet.")

    pose_net.eval()
    disp_net.eval()
    
    tgt_img = load_tensor_image(args.tgt_img, args)
    ref_img = load_tensor_image(args.ref_img, args)

    print(f'target image: {args.tgt_img}, \nreference image: {args.ref_img}')

    img_pil = Image.open(args.tgt_img).convert("RGB")
    img_pil = img_pil.resize((args.img_width, args.img_height), Image.Resampling.LANCZOS)
    img = np.array(img_pil).astype(np.float32)

    tgt_dan_img = depth_anything_transform(img)
    tgt_dan_img = torch.tensor(tgt_dan_img, device=device).unsqueeze(0)
    
    img_pil = Image.open(args.ref_img).convert("RGB")
    img_pil = img_pil.resize((args.img_width, args.img_height), Image.Resampling.LANCZOS)
    img = np.array(img_pil).astype(np.float32)

    ref_dan_img = depth_anything_transform(img)
    ref_dan_img = torch.tensor(ref_dan_img, device=device).unsqueeze(0)

    instrinsic_path = os.path.join(os.path.dirname(args.tgt_img), 'intrinsics.txt' if args.dataset == "nuscenes" else "cam.txt") 
    intrinsic = np.genfromtxt(instrinsic_path).astype(np.float32)
    intrinsic = torch.tensor(intrinsic, device=device).unsqueeze(0)

    
    pose = pose_net(tgt_img, ref_img)
    depth_package = disp_net(tgt_dan_img, tgt_img.size()[-2:])
    tgt_depth, tgt_unty = depth_package[0]
    
    ref_depth, ref_unty = disp_net(ref_dan_img, ref_img.size()[-2:])[0]
    
    ref_img_warped, valid_mask, projected_depth, computed_depth, ref_unty_warped = inverse_warp2(ref_img, tgt_depth, ref_depth, ref_unty, pose, intrinsic, padding_mode='zeros')

    ref_unty_warped = ref_unty_warped.clamp(0, 1)
    
    diff_img = (tgt_img - ref_img_warped).abs()
    # print(f"{(diff_img * valid_mask  >= 1).sum()}")  
    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
    # print(f"{(diff_depth >= 1).sum()}") 
    
    combined_unty = torch.sqrt(tgt_unty**2 + ref_unty_warped**2)
    
    combined_unty_max_op = torch.max(tgt_unty, ref_unty_warped)

    auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)) * valid_mask
    valid_mask = auto_mask

    ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
    diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    
    error_visualization(diff_img[0], "image error only")

    diff_img = diff_img / combined_unty + torch.log(combined_unty)
    diff_depth = diff_depth

    unnormalize_and_save(ref_img_warped[0], name='ref_img_warped.png')
    unnormalize_and_save(tgt_img[0], name='tgt_img.png')
    unnormalize_and_save(ref_img[0], name='ref_img.png')
    depth_visualization((computed_depth)[0], (projected_depth)[0], (diff_depth)[0], 'depth_error')
    depth_visualization((1 / computed_depth)[0], (1 / projected_depth)[0], (diff_depth)[0], 'disp')
    depth_visualization((tgt_unty)[0], (ref_unty_warped)[0], (combined_unty)[0], 'uncerntainty')
    depth_visualization((tgt_unty)[0], (ref_unty)[0], (combined_unty)[0], 'uncerntainty_origin')
    depth_visualization((tgt_depth)[0], (ref_depth)[0], (diff_depth)[0], 'depth_origin')
    depth_visualization(tgt_img[0] * valid_mask[0], ref_img_warped[0] * valid_mask[0], diff_img[0] * valid_mask[0], 'image_error')
    depth_visualization((computed_depth)[0], (ref_depth)[0], (diff_depth)[0], 'depth')
    mask_visualization(valid_mask[0], 'valid_mask')
    heapmap_visualization((tgt_unty)[0], 'target uncertainty')
    heapmap_visualization((ref_unty)[0], 'reference uncertainty')
    heapmap_visualization((ref_unty_warped)[0], 'warped uncertainty')
    heapmap_visualization((combined_unty)[0], 'combined projected uncertainty')
    heapmap_visualization((combined_unty_max_op)[0], 'combined projected uncertainty with max')
    heapmap_visualization((computed_depth)[0], 'computed depth')
    heapmap_visualization((1 / projected_depth)[0], 'synthesized disp')
    heapmap_visualization((diff_depth)[0], 'depth error only')
    heapmap_visualization((1 / tgt_depth)[0], 'target disp')
    heapmap_visualization((1 / ref_depth)[0], 'reference disp')

if __name__ == '__main__':
    main()