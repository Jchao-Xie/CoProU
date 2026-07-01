from eval.eval_datasets import nuScenesDataset
import os
from eval.tools import DepthMetrics
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


from path import Path

from omegaconf import OmegaConf

from eval.tools import write_to_file
from eval.eval_ultis import get_filenames, is_edge, segment_dataset, first_frame_as_base

from models.vggt.models.vggt import VGGT
from models.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.vggt.utils.geometry import unproject_depth_map_to_point_map

def display_str(l):
    return ''.join(['{:^15s}'.format(m) for m in l])

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pretrained-model', default=None, help='path to pre-trained CoProU-VO-MF model, if not provided going for VGGT')
parser.add_argument('--track-length', default=7, type=int, help='local odometry window size')
parser.add_argument('--stop-segment', default=100, type=int, help='number of segment to evaluate')
parser.add_argument('--segment-path', default='experiments/code/test_files.txt',
                    type=str, help='path of segment to evaluate following dynamo-depth')
parser.add_argument('--dataset-dir', default="storage/nuscenes_original_size", type=str, help='path to dataset')

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    pred_xyz *= scale
    alignment_error = pred_xyz - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2) / gtruth_xyz.shape[0])
    return rmse

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectories(
        gtruth_xyz, 
        pred_xyz, 
        title="Trajectory Comparison",
        save_path="trajectory_plot.png",
        step=10):
    """
    Cluster-safe plotting of ground-truth vs predicted trajectories (top-down X-Z view)
    with optional scatter markers every N frames.
    """

    import matplotlib
    matplotlib.use("Agg")  # Safe for clusters without display
    import matplotlib.pyplot as plt
    import numpy as np

    assert gtruth_xyz.shape == pred_xyz.shape, "Shape mismatch between GT and prediction"
    N = gtruth_xyz.shape[0]

    # indices for sparse scatter
    idx = np.arange(0, N, step)

    plt.figure(figsize=(7, 6))

    # Continuous trajectories
    plt.plot(gtruth_xyz[:, 0], gtruth_xyz[:, 2], 'k-', label='Ground Truth', linewidth=2)
    plt.plot(pred_xyz[:, 0], pred_xyz[:, 2], 'r--', label='Prediction', linewidth=2)

    # Scatter sampled points
    plt.scatter(gtruth_xyz[idx, 0], gtruth_xyz[idx, 2], c='black', s=12)
    plt.scatter(pred_xyz[idx, 0], pred_xyz[idx, 2], c='red', s=12)

    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Saved] 2D trajectory plot -> {save_path}")
    
def plot_trajectories_3d(gtruth_xyz, pred_xyz, title="3D Trajectory Comparison", save_path="trajectory_plot_3d.png"):
    """
    Cluster-safe 3D trajectory visualization.
    """
    assert gtruth_xyz.shape == pred_xyz.shape, "Shape mismatch between GT and prediction"

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gtruth_xyz[:, 0], gtruth_xyz[:, 1], gtruth_xyz[:, 2], 'k-', label='Ground Truth')
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], 'r--', label='Prediction')
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Saved] 3D trajectory plot -> {save_path}")
    
    
def eval_odom(args, model, val_segment, track_length, dtype):
    """Function to predict for a single image or folder of images
    """
    
    # Initialize the dataloader
    dataset = segment_dataset(track_length=track_length, segment_path=val_segment)
    loader = DataLoader(dataset, 1, False, num_workers=2, pin_memory=True, drop_last=False)
    N = len(loader)

    # Iterate
    f_id = -1; s = 0
    pred_poses = []
    gt_local_poses = []
    for batch_idx, inputs in tqdm(enumerate(loader)):
        images, gt_poses = inputs
        images = images.to(model.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                
                half_ref = images.size(1) // 2
                
                tgt_idx = [i for i in range(images.size(1)-1)]
                ref_idx = [i+1 for i in range(images.size(1)-1)]
                
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
                
                aggregated_tokens_list, _ = model.model.aggregator(images)
                last_aggregated_tokens = aggregated_tokens_list[-1]
                pred_pose_enc_ref2tgt = model.camera_head(last_aggregated_tokens[:, ref_idx],
                                                        last_aggregated_tokens[:, tgt_idx],
                                                        model.model.aggregator.patch_start_idx, 
                                                        images, 
                                                        pos)
                
                ref2tgt_poses, _ = pose_encoding_to_extri_intri(pred_pose_enc_ref2tgt, images.shape[-2:], pose_encoding_type=model.cfg.model.camera_head.pose_encoding_type)
                ref2tgt_poses = ref2tgt_poses.squeeze(0)
                # Get shape
                N, _, _ = ref2tgt_poses.shape
                # Bottom row [0, 0, 0, 1]
                bottom_row = torch.tensor([0, 0, 0, 1], dtype=ref2tgt_poses.dtype, device=ref2tgt_poses.device)
                bottom_row = bottom_row.view(1, 1, 4).expand(N, 1, 4)
                ref2tgt_poses = torch.cat([ref2tgt_poses, bottom_row], dim=1)      # [N, 4, 4]
                
            # # reorder the target image
            # # extrinsic: [B, N, 4, 4]
            # B, N, _, _ = extrinsic.shape
            # mid = N // 2  # middle index

            # # Move target (index 0) to the middle
            # poses = torch.cat([
            #     extrinsic[:, 1:mid+1],          # left context (ref1..ref_mid)
            #     extrinsic[:, 0:1],              # target pose
            #     extrinsic[:, mid+1:]           # right context
            # ], dim=1).squeeze(0)
            
            # gt_poses_tgt_ctr = torch.cat([
            #     gt_poses[:, 1:mid+1],          # left context (ref1..ref_mid)
            #     gt_poses[:, 0:1],              # target pose
            #     gt_poses[:, mid+1:]           # right context
            # ], dim=1).squeeze(0)
            
            base_pose = torch.eye(4)
            pred_window_poses = [base_pose]
            
            for ref2tgt_pose in ref2tgt_poses:
                base_pose = base_pose @ ref2tgt_pose.cpu()
                pred_window_poses.append(base_pose)
            pred_poses.append(torch.stack(pred_window_poses))
            gt_local_poses.append(first_frame_as_base(gt_poses.view(*gt_poses.shape[:-1], 3, 4).squeeze(0)))
            
            
    # Evaluate ates and speeds
    ates = []
    speeds = []
    num_windows = len(pred_poses)
    for i in range(0, num_windows):
        # local_xyzs = np.array(dump_xyz(pred_poses[i]))
        # gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i]))
        # local_xyzs = np.concatenate((local_xyzs[:,2:3],local_xyzs[:,0:1], local_xyzs[:,1:2]), 1)    # shift axis around
        local_xyzs, gt_local_xyzs = pred_poses[i][:, :3, 3].numpy(), gt_local_poses[i][:, :3, 3].numpy() # [N, 3]
        # local_xyzs = np.concatenate((local_xyzs[:,2:3],local_xyzs[:,0:1], local_xyzs[:,1:2]), 1)    # shift axis around
        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        speeds.append(np.sqrt(((gt_local_xyzs[1:] - gt_local_xyzs[:-1]) ** 2).sum(1)).mean())

    return ates, speeds


def readlines(filename):
    """ Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

from models.model_module import coprou
from omegaconf import OmegaConf
def model_from_pretrained(model_path):
    cfg_path = Path(model_path).parent / 'hparams.yaml'
    cfg = OmegaConf.load(cfg_path)
    model = coprou(cfg=cfg)
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Loaded model weights from {model_path}")
    
    return model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    args = parser.parse_args()
    
    if args.pretrained_model is not None:
        model = model_from_pretrained(args.pretrained_model).to(device)
        dtype = torch.float32
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        dtype = torch.bfloat16

    model.eval()
    
    # Get segments to visualize
    files = readlines(args.segment_path)
    val_segments = sorted(list(set([f.split()[0].split("/")[1] for f in files])))
    txt_path = os.path.join("eval/results/odometry", f'{args.pretrained_model}_len{args.track_length}.txt') if args.pretrained_model is not None else os.path.join("eval/results", f'VGGT.txt')
    npy_path = os.path.join("eval/results/odometry", f'{args.pretrained_model}_len{args.track_length}.npy') if args.pretrained_model is not None else os.path.join("eval/results", f'VGGT.npy')
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    
    # from data.nuscenes_config.splits import val as validation_list 
    # base_dir = "storage/nuscenes_original_size"
    # val_segments = sorted([
    #     folder_name.split('_')[0]
    #     for folder_name in os.listdir(base_dir)
    #     if os.path.isdir(os.path.join(base_dir, folder_name))                # check if it's a directory
    #     and folder_name.endswith("_0")                                      # check suffix
    #     and folder_name.split("_")[0] in validation_list                    # check validation list
    # ])
    # Iterate
    output_strs = [f'=== track_length: {args.track_length}']
    all_ates = []
    all_speeds = []
    for ii, val_segment in tqdm(enumerate(val_segments), desc='Evaluating segments', total=len(val_segments)):
        ates, speeds = eval_odom(args, model, os.path.join(args.dataset_dir, val_segment + '_0'), args.track_length, dtype)
        all_ates += ates
        all_speeds += speeds

        out_str = f'{val_segment:50s} Track={args.track_length} ATE: {np.mean(ates):0.3f} ± {np.std(ates):0.3f},  Speed: {np.mean(speeds):0.3f} ± {np.std(speeds):0.3f},  Len: {len(all_ates)}'
        output_strs.append(out_str)
    
    # Results
    output_strs.append(f'\nATE Trajectory error (Track={args.track_length}):  ')
    output_strs.append(f'Mean:   {np.mean(all_ates)}')
    output_strs.append(f'std:    {np.std(all_ates)}')
    output_strs.append('--')
    output_strs.append(f'Min:    {np.min(all_ates)}')
    output_strs.append(f'Median: {np.median(all_ates)}')
    output_strs.append(f'Max:    {np.max(all_ates)}')
    
    output_strs.append('==')
    output_strs.append('\nSpeed:  ')
    output_strs.append(f'Mean:   {np.mean(all_speeds)}')
    output_strs.append(f'std:    {np.std(all_speeds)}')
    output_strs.append('--')
    output_strs.append(f'Min:    {np.min(all_speeds)}')
    output_strs.append(f'Median: {np.median(all_speeds)}')
    output_strs.append(f'Max:    {np.max(all_speeds)}')
    output_strs.append('--')
    output_strs.append(f'len:    {len(all_speeds)}')

    # Write to terminal / out_path
    for s in output_strs:
        print(s)
    write_to_file(output_strs, txt_path)
    np.save(npy_path, np.stack((np.array(all_ates), np.array(all_speeds))).transpose((1,0)))


if __name__=="__main__":
    main()