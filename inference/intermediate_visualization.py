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


from utils import unnormalize_and_save, depth_visualization, mask_visualization, error_visualization, heapmap_visualization

from models.vggt.models.vggt import VGGT
from models.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.vggt.utils.geometry import unproject_depth_map_to_point_map

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pretrained-model', required=True, help='path to pre-trained CoProU-VO-MF model')
parser.add_argument("--img-height", default=294, type=int, help="Image height")
parser.add_argument("--img-width", default=518, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--tgt-img", type=str, required=True, help="target image path")
parser.add_argument("--ref-imgs", type=str, nargs="+", default=None, help="reference image path")
parser.add_argument("--sequence-length", default=5, type=int, help="The length of the sequence")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from train import coprou
from omegaconf import OmegaConf
from inference.point_cloud import get_reference_images
def model_from_pretrained(model_path):
    cfg_path = Path(model_path).parent / 'hparams.yaml'
    cfg = OmegaConf.load(cfg_path)
    model = coprou(cfg=cfg)
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Loaded model weights from {model_path}")
    
    return model, cfg

def process_poses(poses, B, S, H, W):
    pose_mat, (tgt_uncertainty, ref_uncertainty) = poses

    pose_mat = pose_mat.view(B * S, *pose_mat.shape[2:])
    tgt_uncertainty = tgt_uncertainty.view(B * S, -1, H, W)
    ref_uncertainty = ref_uncertainty.view(B * S, -1, H, W)

    return pose_mat, (tgt_uncertainty, ref_uncertainty)

@torch.no_grad()
def main():
    args = parser.parse_args()

    model, cfg = model_from_pretrained(args.pretrained_model)
    model = model.to(device)
    dtype = torch.bfloat16 if 'bf16' in cfg.optim.amp.precision else torch.float32
    is_vggt = False
    img_size, patch_size = cfg.img_size, cfg.patch_size
    
    model.eval()

    tgt_img = load_and_preprocess_images([args.tgt_img], target_size=img_size, patch_size=patch_size).to(device)
    if args.ref_imgs is None:
        ref_imgs_path = get_reference_images(args.tgt_img, args.sequence_length)
    else:
        ref_imgs_path = args.ref_imgs
    if len(ref_imgs_path) != 0:
        ref_imgs = load_and_preprocess_images(ref_imgs_path, target_size=img_size, patch_size=patch_size).to(device)
        ref_imgs = list(ref_imgs.unsqueeze(0).unbind(dim=1))
    else:
        ref_imgs = []

    print(f'target image: {args.tgt_img}')
    for reference_img in ref_imgs_path:
        print(f'reference image: {reference_img}')
        
    # Stack images (tgt + refs)
    images = torch.stack([tgt_img] + ref_imgs, dim=1)
    
    # if model.cfg.model.camera_head.type is not None:
    if len(ref_imgs) != 1:
        assert len(ref_imgs) % 2 == 0
        
        half_ref = len(ref_imgs) // 2
        
        images = torch.stack(ref_imgs[:half_ref] + [tgt_img] + ref_imgs[half_ref:], dim=1)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # print(f'target image: {args.tgt_img}')
    # for reference_img in ref_imgs_path:
    #     print(f'reference image: {reference_img}')
        
    # intrinsic
        # Collect all image paths (tgt + refs)
    all_intrinsic_paths = [Path(p) for p in [args.tgt_img] + ref_imgs_path]

    # Load intrinsics
    if 'waymo' in args.tgt_img:
        gt_intrinsics = [np.genfromtxt(img_path.parent.parent / 'intrinsics.txt') for img_path in all_intrinsic_paths]
    elif 'kitti' in args.tgt_img.lower():
        gt_intrinsics = []
        for img_path in all_intrinsic_paths:
            calib_path = img_path.parent.parent / 'calib.txt'
            calib = {}
            with open(calib_path, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()]).reshape(3, 4)
            gt_intrinsics.append(calib['P2'][:, :3])
    else:
        gt_intrinsics = [np.genfromtxt(img_path.parent / 'intrinsics.txt') for img_path in all_intrinsic_paths]
        
    # Compute scaling factors
    in_w, in_h = Image.open(args.tgt_img).convert("RGB").size
    scaled_h = int(images[0].size(2))  # H
    scaled_w = int(images[0].size(3))  # W
    x_scaling = scaled_w / in_w
    y_scaling = scaled_h / in_h
    #x_scaling, y_scaling = 1, 1

    # Resize intrinsics
    output_intrinsics = []
    for intr in gt_intrinsics:
        intr_resized = intr.copy()
        intr_resized[0] *= x_scaling  # fx and cx (x-axis)
        intr_resized[1] *= y_scaling  # fy and cy (y-axis)
        output_intrinsics.append(intr_resized)

    # Convert to tensor
    gt_intrinsics = torch.from_numpy(np.stack(output_intrinsics)).float().to(device)

    if not OmegaConf.select(cfg, "model.cnn.enabled", default=False):
        with torch.cuda.amp.autocast(dtype=dtype):
            # images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.model.aggregator(images)
    
    # Predict Cameras
    if (not OmegaConf.select(cfg, "model.cnn.enabled", default=False) 
        and OmegaConf.select(cfg, "model.camera_head.type", default=None) is None):
        if args.pretrained_model is not None:
            pose_enc = model.model.camera_head(aggregated_tokens_list)[-1] \
                if model.cfg.model.vggt.enable_camera else model.camera_head(aggregated_tokens_list)[-1]
        else:
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsics, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:], 
                                                            pose_encoding_type=model.cfg.model.camera_head.pose_encoding_type if args.pretrained_model is not None else model.camera_head.pose_encoding_type
                                                            )
        
        B, N, _, _ = extrinsics.shape
        
        bottom = torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype, device=extrinsics.device)
        bottom = bottom.view(1, 1, 1, 4).expand(B, N, 1, 4)
        extrinsics_homo = torch.cat([extrinsics, bottom], dim=2)
        
        tgt_index, ref_index = 1, 0 # hard code to get first pair
        cat_aggregated_tokens_list = [torch.cat([x[:, [tgt_index]], x[:, [ref_index]]], dim=-1)
                                    for x in aggregated_tokens_list]
        pair_images_tgt = images[:, [tgt_index]] # just for shape, not for gradient 
        tgt_unty, ref_unty = model.uncertainty_head(cat_aggregated_tokens_list, 
                                                                        pair_images_tgt, 
                                                                        ps_idx)
        tgt_unty = tgt_unty.squeeze(1).permute(0, 3, 1, 2)

        pose = extrinsics_homo[:, ref_index] @ torch.inverse(extrinsics_homo[:, tgt_index])
        
        # Predict Depth Maps
        if model.cfg.model.vggt.enable_depth:
            depth_map, depth_conf = model.model.depth_head(aggregated_tokens_list, images, ps_idx)
        else:
            depth_map = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
        tgt_depth, ref_depth = depth_map[:, tgt_index].permute(0, 3, 1, 2), depth_map[:, ref_index].permute(0, 3, 1, 2)
        
        standardized_img = (images - mean) / std
        tgt_img, ref_img = standardized_img[:, tgt_index], standardized_img[:, ref_index]
        
        if model.cfg.loss.with_gt_intrinsic:
            tgt_intrinsic, ref_intrinsic = None, None
        else:
            tgt_intrinsic, ref_intrinsic = intrinsic[:, tgt_index], intrinsic[:, ref_index]
        

        
        ref_img_warped, valid_mask, projected_depth, computed_depth, ref_unty_warped = inverse_warp(ref_img, tgt_depth, 
                                                                                                    ref_depth, ref_unty, 
                                                                                                    pose, gt_intrinsics[:1], tgt_intrinsic, 
                                                                                                    ref_intrinsic, 
                                                                                                    model.cfg.loss)
        
    else:    
        (
            poses, poses_inv, tgt_depths, ref_depths,
            tgt_intrinsics, ref_intrinsics,
            tgt_idx, ref_idx,
            pair_images_tgt, pair_images_ref
        ) = model.get_pairs_prediction(images) if not OmegaConf.select(cfg, "model.cnn.enabled", default=False) else model.get_cnn_prediction(images)
        
        B, S, _, H, W = pair_images_tgt.shape


        tgt_img_scaled = (pair_images_tgt - mean) / std
        ref_img_scaled = (pair_images_ref - mean) / std
        
        tgt_img_scaled = tgt_img_scaled.view(B * S, -1, H, W)
        ref_img_scaled = ref_img_scaled.view(B * S, -1, H, W)
        
        intrinsic_scaled = gt_intrinsics[:1].unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3)  # broadcast intrinsic
        
        reshape_or_none = lambda x: x.view(B * S, -1, H, W) if x is not None else None
        tgt_depth_scaled = [reshape_or_none(x) for x in tgt_depths]
        ref_depth_scaled = [reshape_or_none(x) for x in ref_depths]
        
        poses = process_poses(poses, B, S, H, W)
        poses_inv = process_poses(poses_inv, B, S, H, W)
        
        # we use inverse pose and uncertainty
        pose, unty = poses_inv
        tgt_unty, ref_unty = unty
        
        tgt_depth, tgt_depth_conf = ref_depth_scaled
        ref_depth, ref_depth_conf = tgt_depth_scaled
        
        tgt_img, ref_img = ref_img_scaled, tgt_img_scaled
        
        ref_img_warped, valid_mask, projected_depth, computed_depth, ref_unty_warped = inverse_warp(ref_img, tgt_depth, ref_depth, 
                                                                                                    ref_unty, pose, intrinsic_scaled, 
                                                                                                    None, None, 
                                                                                                    model.cfg.loss)



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
    
    diff_img_wo_unty = diff_img.clone()

    diff_img = diff_img / combined_unty + torch.log(combined_unty)
    diff_depth = diff_depth

    for i in range(diff_img.shape[0]):
        error_visualization(diff_img_wo_unty[i], f"image error only_{i}")
        unnormalize_and_save(ref_img_warped[i], name=f'ref_img_warped_{i}.png')
        unnormalize_and_save(tgt_img[i], name=f'tgt_img_{i}.png')
        unnormalize_and_save(ref_img[i], name=f'ref_img_{i}.png')
        depth_visualization((computed_depth)[i], (projected_depth)[i], (diff_depth)[i], f'depth_error_{i}')
        depth_visualization((1 / computed_depth)[i], (1 / projected_depth)[i], (diff_depth)[i], f'disp_{i}')
        depth_visualization((tgt_unty)[i], (ref_unty_warped)[i], (combined_unty)[i], f'uncerntainty_{i}')
        depth_visualization((tgt_unty)[i], (ref_unty)[i], (combined_unty)[i], f'uncerntainty_origin_{i}')
        depth_visualization((tgt_depth)[i], (ref_depth)[i], (diff_depth)[i], f'depth_origin_{i}')
        depth_visualization(tgt_img[i] * valid_mask[i], ref_img_warped[i] * valid_mask[i], diff_img[i] * valid_mask[i], f'image_error_{i}')
        depth_visualization((computed_depth)[i], (ref_depth)[i], (diff_depth)[i], f'depth_{i}')
        mask_visualization(valid_mask[i], f'valid_mask_{i}')
        heapmap_visualization((tgt_unty)[i], f'target_uncertainty_{i}')#, vmin=0, vmax=1)
        heapmap_visualization((ref_unty)[i], f'reference_uncertainty_{i}')#, vmin=0, vmax=1)
        heapmap_visualization((ref_unty_warped)[i], f'warped_uncertainty_{i}')#, vmin=0, vmax=1)
        heapmap_visualization((combined_unty)[i], f'combined_projected_uncertainty_{i}')#, vmin=0, vmax=1)
        heapmap_visualization((computed_depth)[i], f'computed_depth_{i}')
        heapmap_visualization((1 / projected_depth)[i], f'synthesized_disp_{i}')
        heapmap_visualization((diff_depth)[i], f'depth_error_only_{i}')
        heapmap_visualization((1 / tgt_depth)[i], f'target_disp_{i}')
        heapmap_visualization((1 / ref_depth)[i], f'reference_disp_{i}')

if __name__ == '__main__':
    main()