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


from inverse_warp import *

import os
import glob


from models.vggt.models.vggt import VGGT
from models.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.vggt.utils.geometry import unproject_depth_map_to_point_map

from models.vggt.utils.geometry import closed_form_inverse_se3


from models.model_module import coprou
from omegaconf import OmegaConf
def model_from_pretrained(model_path):
    cfg_path = Path(model_path).parent / 'hparams.yaml'
    cfg = OmegaConf.load(cfg_path)
    model = coprou(cfg=cfg)
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Loaded model weights from {model_path}")
    
    return model, cfg

def get_reference_images(tgt_img_path, sequence_length):
    parent_dir = os.path.dirname(tgt_img_path)
    all_imgs = sorted(
        glob.glob(os.path.join(parent_dir, "*.png" if 'kitti' in parent_dir.lower() else "*.jpg" )),
        key=lambda x: os.path.basename(x)
    )

    tgt_idx = all_imgs.index(tgt_img_path)
    half_len = sequence_length // 2

    # neighbors including itself
    start_idx = max(0, tgt_idx - half_len)
    end_idx = min(len(all_imgs), tgt_idx + half_len + sequence_length % 2)
    seq_imgs = all_imgs[start_idx:end_idx]

    # remove target itself from ref list
    ref_imgs = [img for img in seq_imgs if img != tgt_img_path]
    return ref_imgs

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pretrained-model', default=None, help='path to pre-trained CoProU-VO-MF model, if not provided going for VGGT')
    parser.add_argument("--tgt-img", type=str, required=True, help="target image path")
    parser.add_argument("--ref-imgs", type=str, nargs="+", default=None, help="reference image path")
    parser.add_argument("--sequence-length", default=97, type=int, help="The length of the sequence")
    parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
    parser.add_argument("--conf_thres", default=40, type=int, help="confidence threshold")
    parser.add_argument("--conf-range", default=5, type=int, help="range of uncertainty windows")
    parser.add_argument("--filter-z-percentile", default=100, type=int, help="fitering sky with distance")
    parser.add_argument("--save-dir", default="visualization_inference", type=str, help="directory to save visualizations")
    parser.add_argument("--show-cam", action='store_true', help="whether to show camera in the point cloud")
    parser.add_argument("--filter-z-percentile-per-img", default=40, type=int, help="fitering sky with distance per image, for each image, filter out points whose depth is larger than the given percentile among all points in that image")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.pretrained_model is not None:
        model, cfg = model_from_pretrained(args.pretrained_model)
        model = model.to(device)
        dtype = torch.bfloat16 if 'bf16' in cfg.optim.amp.precision else torch.float32
        is_vggt = False
        img_size, patch_size = cfg.img_size, cfg.patch_size
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        dtype = torch.bfloat16
        is_vggt = True
        img_size, patch_size = 518, 14
    
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
    scaled_h = int(tgt_img.size(2))  # H
    scaled_w = int(tgt_img.size(3))  # W
    x_scaling = scaled_w / in_w
    y_scaling = scaled_h / in_h
    # x_scaling, y_scaling = 1, 1

    # Resize intrinsics
    output_intrinsics = []
    for intr in gt_intrinsics:
        intr_resized = intr.copy()
        intr_resized[0] *= x_scaling  # fx and cx (x-axis)
        intr_resized[1] *= y_scaling  # fy and cy (y-axis)
        output_intrinsics.append(intr_resized)

    # Convert to tensor
    gt_intrinsics = torch.from_numpy(np.stack(output_intrinsics)).float().to(device)
    
    if is_vggt or not OmegaConf.select(cfg, "model.cnn.enabled", default=False) :
    
        camera_type = None if args.pretrained_model is None else model.cfg.model.camera_head.type
        
        if camera_type is None:
            with torch.amp.autocast('cuda', dtype=dtype):

                aggregated_tokens_list, ps_idx = model.model.aggregator(images) if args.pretrained_model is not None else model.aggregator(images)
                
            # Predict Depth Maps
            if args.pretrained_model is None:
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            elif model.cfg.model.vggt.enable_depth:
                depth_map, depth_conf = model.model.depth_head(aggregated_tokens_list, images, ps_idx)
            else:
                depth_map = model.depth_head(aggregated_tokens_list, images, ps_idx)
                
            # uncertainty_map = get_uncertainty(model, aggregated_tokens_list, images, ps_idx).squeeze(0)
            if args.pretrained_model is not None:
                uncertainty_map = iter_get_uncertainty(model, aggregated_tokens_list, images, ps_idx, args.conf_range).squeeze(0)
                
            # Predict Cameras
            if args.pretrained_model is not None:
                pose_enc = model.model.camera_head(aggregated_tokens_list)[-1] if model.cfg.model.vggt.enable_camera else model.camera_head(aggregated_tokens_list)[-1]
            else:
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:], 
                                                                pose_encoding_type=model.cfg.model.camera_head.pose_encoding_type if args.pretrained_model is not None else model.camera_head.pose_encoding_type
                                                                )

        # from eval.tools import visualize_pointcloud_with_gradio
        # vis_depth = depth_map[:, :1]
        # visualize_pointcloud_with_gradio(
        #         depth=vis_depth.squeeze(1).permute(0, 3, 1, 2),
        #         K=None,
        #         inv_K=torch.linalg.inv(gt_intrinsics[:1]),
        #         # camTcam=camTcam,
        #         image=images[:,0],   # optional: original image as color
        #         save_path="pointcloud_scene.glb"
        #     )
            images = images.squeeze(0) 
            extrinsic = extrinsic.squeeze(0)
            depth_map = depth_map.squeeze(0)
            
            if args.pretrained_model is None:
                depth_conf = depth_conf.squeeze(0)
                intrinsic = intrinsic.squeeze(0)
            
            
            N = extrinsic.shape[0]
            bottom = torch.tensor([0, 0, 0, 1], device=extrinsic.device, dtype=extrinsic.dtype)
            bottom = bottom.view(1, 1, 4).expand(N, -1, -1)  # (N, 1, 4)
            extrinsic_h = torch.cat([extrinsic, bottom], dim=1)  # (N, 4, 4)
            base_extrinsic = extrinsic_h[0]
            inv_base_extrinsic = closed_form_inverse_se3(base_extrinsic.unsqueeze(0))
            extrinsic = extrinsic_h @ inv_base_extrinsic
            extrinsic = extrinsic[:, :3, :]
            
        else:
            
            poses, poses_inv, aggregated_tokens_list, images, ps_idx, depth_map = get_reloc3r_poses(model=model, tgt_img=tgt_img, ref_imgs=ref_imgs, dtype=dtype)
                
            uncertainty_map = iter_get_uncertainty(model, aggregated_tokens_list, images, ps_idx, args.conf_range).squeeze(0)
            
            poses = poses.squeeze(0)
            poses_inv = poses_inv.squeeze(0)
            depth_map = depth_map.squeeze(0)
            images = images.squeeze(0) 
            
            extrinsic = torch.eye(4, device=poses.device)
            
            extrinsics = [extrinsic]
            
            for pose_inv in poses_inv:
                extrinsic = extrinsic @ pose_inv
                extrinsics.append(extrinsic.inverse())
                
            extrinsic = torch.stack(extrinsics, dim=0)  
            extrinsic = extrinsic[:, :3, :]
            
    else:
        assert len(ref_imgs) % 2 == 0
        
        half_ref = len(ref_imgs) // 2
        
        images = torch.stack(ref_imgs[:half_ref] + [tgt_img] + ref_imgs[half_ref:], dim=1)
        
        (
            pre_poses, pre_poses_inv, tgt_depths, ref_depths,
            _, _,
            _, _,
            _, _
        ) = model.get_cnn_prediction(images)
        
        pre_poses    , pre_unty     = pre_poses
        pre_poses_inv, pre_unty_inv = pre_poses_inv
        
        pre_tgt_unty, pre_ref_unty = pre_unty
        tgt_depths, _ = tgt_depths
        ref_depths, _ = ref_depths
        
        depth_map       = torch.concat([tgt_depths  , ref_depths[:, -1:]]  , dim=1).squeeze(0)
        uncertainty_map = torch.concat([pre_tgt_unty, pre_ref_unty[:, -1:]], dim=1).squeeze(0)
        
        depth_map, uncertainty_map = depth_map.permute(0, 2, 3, 1), uncertainty_map.permute(0, 2, 3, 1)
        
        pre_poses_inv = pre_poses_inv.squeeze(0)

        images = images.squeeze(0) 
        
        extrinsic = torch.eye(4, device=pre_poses.device)
        
        extrinsics = [extrinsic]
        
        for pre_pose_inv in pre_poses_inv:
            extrinsic = extrinsic @ pre_pose_inv
            extrinsics.append(extrinsic.inverse())
            
        extrinsic = torch.stack(extrinsics, dim=0)  
        extrinsic = extrinsic[:, :3, :]
        
    
    from models.vggt.utils.geometry import unproject_depth_map_to_point_map
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, 
                                                    gt_intrinsics if args.pretrained_model is not None else intrinsic)
    
    predictions = {"depth_conf": depth_conf if args.pretrained_model is None else (1 - uncertainty_map),
                  "world_points_from_depth": world_points,
                  "images": images,
                  "extrinsic": extrinsic,
                  }
    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()
            
    from visualization_tools.visual_util import predictions_to_glb
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=args.conf_thres,
        show_cam=args.show_cam,
        filter_z_percentile=args.filter_z_percentile,
        filter_z_percentile_per_img=args.filter_z_percentile_per_img,
    )
    glb_path = os.path.join(args.save_dir, "scene.glb")
    glbscene.export(glb_path)
    print(f"✅ GLB saved to {glb_path}")
    
    save_dir = args.save_dir
    for n in range(images.shape[0]):
        img_n = images[n]                    # (3, H, W)
        depth_n = depth_map[n]               # (H, W, 1)
        disp_n = 1 / (depth_n + 1e-8) # if not precomputed
        uncertainty_n = (1 - depth_conf)[n] if args.pretrained_model is None else uncertainty_map[n]   # (H, W, 1)

        save_path = os.path.join(save_dir, f"inference_frame_{n:02d}.png")
        visualize_inference_results(img_n, depth_n, disp_n, uncertainty_n, save_path)
        
    visualize_glb_with_gradio(glbscene, port=8080, save_dir=args.save_dir)
        
def get_reloc3r_poses(model, tgt_img, ref_imgs, dtype):
    
    assert len(ref_imgs) % 2 == 0
        
    half_ref = len(ref_imgs) // 2
    
    images = torch.stack(ref_imgs[:half_ref] + [tgt_img] + ref_imgs[half_ref:], dim=1)
    
    with torch.amp.autocast('cuda', dtype=dtype):
        aggregated_tokens_list, ps_idx = model.model.aggregator(images)
    
    # Predict Depth Maps
    if model.cfg.model.vggt.enable_depth:
        depth_map, depth_conf = model.model.depth_head(aggregated_tokens_list, images, ps_idx)
    else:
        depth_map = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
    # get position for pose estimation
    B, S, C_in, H, W = images.shape
    
    # we have S-1 pairs
    pos = None
    if model.model.aggregator.rope is not None:
        pos = model.model.aggregator.position_getter(B * (S-1), 
                                                    H // model.model.aggregator.patch_size, 
                                                    W // model.model.aggregator.patch_size, device=images.device)

    if model.model.aggregator.patch_start_idx > 0:
        # do not use position embedding for special tokens (camera and register tokens)
        # so set pos to 0 for the special tokens
        pos = pos + 1
        pos_special = torch.zeros(B * (S-1), model.model.aggregator.patch_start_idx, 2).to(images.device).to(pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)
    
        
    
    tgt_idx = [i for i in range(images.size(1)-1)]
    ref_idx = [i+1 for i in range(images.size(1)-1)]
    
    # get depths
    tgt_depths = (depth_map[:, tgt_idx].permute(0, 1, 4, 2, 3), depth_conf[:, tgt_idx].unsqueeze(1) if model.cfg.model.vggt.enable_depth else None)
    tgt_intrinsics = None
    
    ref_depths = (depth_map[:, ref_idx].permute(0, 1, 4, 2, 3), depth_conf[:, ref_idx].unsqueeze(1) if model.cfg.model.vggt.enable_depth else None)
    ref_intrinsics = None # we don't predict intrinsics in this case, using none for adaption to other cases
    
    # predict pose
    last_aggregated_tokens = aggregated_tokens_list[-1]
    poses, poses_inv = model.get_pairs_pose(last_aggregated_tokens, images, tgt_idx, ref_idx, pos) 
    
    return poses, poses_inv, aggregated_tokens_list, images, ps_idx, depth_map
    
   
def iter_get_uncertainty(model, aggregated_tokens_list, images, ps_idx, uncertainty_range):
    B, N, C, H, W = images.shape
    half_range = uncertainty_range // 2
    assert uncertainty_range % 2 == 1

    all_uncertainties = []

    for n in range(N):
        # Define local neighborhood indices
        start_idx = max(0, n - half_range)
        end_idx = min(N, n + half_range + 1)  # +1 since end is exclusive
        idx_list = list(range(start_idx, end_idx))

        # Reorder: target first
        if n in idx_list:
            idx_list.remove(n)
        idx_list = [n] + idx_list

        # Slice tokens for this local window (make a *copy* of sliced list)
        local_tokens = [t[:, idx_list].clone() for t in aggregated_tokens_list]
        local_images = images[:, idx_list]

        # Compute uncertainty in this local window
        tgt_uncertainty = get_uncertainty_local(
            model,
            local_tokens,
            local_images,
            ps_idx
        )

        all_uncertainties.append(tgt_uncertainty[:, 0])  # 0 = current target
        
        del local_tokens
        torch.cuda.empty_cache()

    # Stack back to [B, N, H, W] (or whatever shape uncertainty has)
    return torch.stack(all_uncertainties, dim=1)
        
        
        
def get_uncertainty_local(model, aggregated_tokens_list, images, ps_idx):
    B, N, C, H, W = images.shape  

    # Build all ordered pairs (i, j)
    i, j = torch.meshgrid(torch.arange(1), torch.arange(N), indexing='ij')
    mask = i != j
    i_idx = i[mask]
    j_idx = j[mask]
    P = i_idx.shape[0]

    pair_tokens_list_tgt_ref = []

    # Process and remove each layer token on the fly
    while aggregated_tokens_list:  # while list is not empty
        layer_tokens = aggregated_tokens_list.pop(0)  # remove first element each time

        tgt_tokens_l = layer_tokens[:, i_idx, :, :]
        ref_tokens_l = layer_tokens[:, j_idx, :, :]
        pair_tokens_l = torch.cat([tgt_tokens_l, ref_tokens_l], dim=-1)

        pair_tokens_list_tgt_ref.append(pair_tokens_l)

        # Explicitly clean up references to free GPU memory
        del layer_tokens, tgt_tokens_l, ref_tokens_l
        torch.cuda.empty_cache()

    # At this point aggregated_tokens_list is empty, no leftover big tensors
    pair_images_tgt = images[:, i_idx, :, :, :]

    tgt_uncertainty, _ = model.uncertainty_head(
        pair_tokens_list_tgt_ref,
        pair_images_tgt,
        ps_idx
    )

    tgt_uncertainty = tgt_uncertainty.view(B, 1, N - 1, *tgt_uncertainty.shape[2:])
    tgt_uncertainty_mean = torch.mean(tgt_uncertainty, dim=2)
    return tgt_uncertainty_mean
      
# import os
# import torch
# import torchvision.utils as vutils
# import matplotlib.cm as cm

# os.makedirs("./visualization", exist_ok=True)
# viridis = cm.get_cmap("viridis")

# # ============================================================
# # 1. Save images (5 frames)
# # images: [1, 5, 3, H, W]
# # ============================================================
# imgs_vis = images.squeeze(0).detach().cpu()  # [5,3,H,W]

# for i in range(imgs_vis.size(0)):
#     vutils.save_image(
#         imgs_vis[i],
#         f"./visualization/image_{i}.png",
#         normalize=True
#     )

# # ============================================================
# # 2. Save 4 uncertainty maps (fixed scale [0,1])
# # tgt_uncertainty: [1,1,4,H,W,1]
# # ============================================================
# unc = (
#     tgt_uncertainty
#     .squeeze(0)
#     .squeeze(0)
#     .squeeze(-1)
#     .detach()
#     .cpu()
# )  # [4,H,W]

# unc = unc.clamp(0.0, 1.0)

# for i in range(unc.size(0)):
#     u = unc[i].numpy()                     # [H,W]
#     u_color = viridis(u)[:, :, :3]         # RGB
#     u_color = torch.from_numpy(u_color).permute(2, 0, 1).float()
#     vutils.save_image(
#         u_color,
#         f"./visualization/uncertainty_{i}.png"
#     )

# # ============================================================
# # 3. Save mean uncertainty (same scale)
# # tgt_uncertainty_mean: [1,1,H,W,1]
# # ============================================================
# u_mean = (
#     tgt_uncertainty_mean
#     .squeeze(0)
#     .squeeze(0)
#     .squeeze(-1)
#     .detach()
#     .cpu()
# )  # [H,W]

# u_mean = u_mean.clamp(0.0, 1.0)

# u_color = viridis(u_mean.numpy())[:, :, :3]
# u_color = torch.from_numpy(u_color).permute(2, 0, 1).float()

# vutils.save_image(
#     u_color,
#     "./visualization/uncertainty_mean.png"
# )
      
      
        
import torch
import numpy as np
import matplotlib.pyplot as plt

def normalize_img(x: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] for visualization."""
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def visualize_inference_results(image: torch.Tensor,
                                     depth_map: torch.Tensor,
                                     disp_map: torch.Tensor,
                                     uncertainty_map: torch.Tensor,
                                     save_path: str):
    """
    Visualize RGB image, depth, disparity, and uncertainty in a 2x2 grid.

    Args:
        image (Tensor): (3, H, W) or (H, W, 3), RGB
        depth_map (Tensor): (H, W) or (H, W, 1)
        disp_map (Tensor): (H, W) or (H, W, 1)
        uncertainty_map (Tensor): (H, W) or (H, W, 1)
        save_path (str): Path to save the visualization.
    """
    # --- Convert image ---
    img = image.detach().cpu()
    if img.dim() == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)  # (H, W, 3)
    img = img.numpy()
    img = normalize_img(img)

    # --- Convert maps ---
    depth = normalize_img(depth_map.squeeze().detach().cpu().numpy())
    disp = normalize_img(disp_map.squeeze().detach().cpu().numpy())
    uncertainty = normalize_img(uncertainty_map.squeeze().detach().cpu().numpy())

    # --- Plot 2×2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(depth, cmap='plasma')
    axes[0, 1].set_title('Depth')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(disp, cmap='plasma')
    axes[1, 0].set_title('Disparity')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(uncertainty, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Uncertainty')
    axes[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"✅ Saved visualization grid → {save_path}")
    

import gradio as gr
import tempfile

def visualize_glb_with_gradio(glbscene, port: int = 8080, save_dir=None):
    """
    Visualize a trimesh.Scene (glbscene) interactively using Gradio's Model3D.
    
    Args:
        glbscene (trimesh.Scene): Scene object from predictions_to_glb
        port (int): Port for the local Gradio app
    """
    # 1. Save the GLB to a temporary or fixed directory
    os.makedirs(save_dir, exist_ok=True)
    glb_path = os.path.join(save_dir, "scene.glb")
    glbscene.export(glb_path)
    print(f"✅ GLB saved to {glb_path}")

    # 2. Build a simple Gradio UI to visualize it
    with gr.Blocks(title="GLB Scene Viewer") as demo:
        gr.Markdown("## 🧭 GLB Scene Visualization")
        gr.Markdown(f"**File:** `{glb_path}`")
        gr.Model3D(value=glb_path, height=600, clear_color=[0.13, 0.13, 0.14, 1])

    # 3. Launch the Gradio app
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
    
    
if __name__ == '__main__':
    main()