import torch

from imageio import imread, imsave
# from skimage.transform import resize as imresize
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import glob

from models.vggt.models.vggt import VGGT
from models.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.vggt.utils.geometry import unproject_depth_map_to_point_map
from eval.eval_ultis import first_frame_as_base
from eval.tools import write_to_file
import datetime
from inference.traj_dataloader import make_dataloader
from inference.utils.sim3_alligment import rel_to_abs_poses, umeyama_sim3, se3_from_anchor, estimate_overlap_sim3, apply_sim3_to_abs_poses

from inference.utils.debugging import plot_window_alignment_debug

from eval.local_odometry import plot_trajectories, plot_trajectories_3d, compute_ate, readlines

import cv2

from data.nuscenes_config.splits import val as validation_list

from inverse_warp import pose_vec2mat

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained-model', default=None, help='path to pre-trained CoProU-VO-MF model')
parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='eval_output/', type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--test", action='store_true', help="using test dataset")
parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
parser.add_argument("--sequence-path", type=str, help="path to sequence to test")
parser.add_argument("--interval", type=int, default=1, help="define the interval of target- and reference image")
parser.add_argument("--window-length", type=int, default=8, help="define the length of sliding windows")
parser.add_argument("--sliding-step", type=int, default=7, help="how larger is the step")
parser.add_argument("--rescale-factor", type=float, default=1, help="factor to rescale images")
parser.add_argument("--eval-interval", type=int, default=1, help="how larger is the step")
parser.add_argument("--dataset", type=str, default='nusc', help="dataset")
parser.add_argument('--segment-path', default='experiments/code/test_files.txt',
                    type=str, help='path of segment to evaluate following dynamo-depth')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

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

@torch.no_grad()
def main():
    args = parser.parse_args()
    assert args.window_length > args.sliding_step
    
    # window_length, sliding_step = args.window_length, args.sliding_step
    
    if args.pretrained_model:
        model, cfg = model_from_pretrained(args.pretrained_model)
        model = model.to(device)
        dtype = torch.bfloat16 if 'bf16' in cfg.optim.amp.precision else torch.float32
        is_vggt = False
        img_size, patch_size = cfg.img_size, cfg.patch_size
        _resnet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1).to(device)
        _resnet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1).to(device)
        
        # _resnet_mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(1, 1, 3, 1, 1).to(device)
        # _resnet_std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1).to(device)

    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        dtype = torch.bfloat16
        args.pretrained_model = f'checkpoints/official_vggt_whole_squence/official_checkpoint.ckpt' # /{datetime.datetime.now().strftime("%m-%d-%H:%M")}
        is_vggt = True
        img_size, patch_size = 518, 14
        
    model.eval()
    
    # Find the index of "checkpoints" in the parts of the path
    parts = Path(args.pretrained_model).parts
    start_index = parts.index("checkpoints")

    # Build the relative path from "checkpoints" onward, and remove .pth.tar suffix
    relative_path = Path(*parts[start_index:]).with_suffix('')  # removes .tar
    relative_path = relative_path.with_suffix('')                # removes .pth
    output_dir = Path(args.output_dir)
    output_dir = output_dir / relative_path
    
    if 'nusc' in args.dataset_dir:
        scenes = sorted([
                            os.path.join(args.dataset_dir, folder.name)
                            for folder in Path(args.dataset_dir).iterdir()
                            if folder.is_dir() and folder.name.endswith("_0") and folder.name.split("_")[0] in validation_list
                        ])
        output_dir = output_dir / 'nusc'
        
    elif 'waymo' in args.dataset_dir:
        sequence_list = glob.glob(os.path.join(args.dataset_dir, "*"))            
        # sequence_list = [file_path.split(args.dataset_dir)[-1].lstrip('/') for file_path in sequence_list]
        sequence_list = sorted(sequence_list)
        with open('vggt_training/data/datasets/waymo_day_light_split_val.txt', 'r') as f:
            day_light_sequences = [line.strip() for line in f if line.strip()]

        scenes = [seq for seq in sequence_list if seq.split(args.dataset_dir)[-1].lstrip('/') in day_light_sequences]
        output_dir = output_dir / 'waymo'
    elif 'kitti' in args.dataset_dir:
        sequence_list = glob.glob(os.path.join(args.dataset_dir, "*"))  
        sequence_list = sorted(sequence_list)
        test_scences = ["01", "09", "10"]
        scenes = [
            seq for seq in sequence_list
            if os.path.basename(seq) in test_scences
        ]
        output_dir = output_dir / 'kitti_odometry'
    
    else:
        raise ValueError("The dataset is not supported")
    
    output_dir = output_dir / f'len{args.window_length}_step{args.sliding_step}_evalInterval{args.eval_interval}_rescale{args.rescale_factor}'
    txt_path = output_dir / f'len{args.window_length}_step{args.sliding_step}_evalInterval{args.eval_interval}_rescale{args.rescale_factor}.txt'
    
    output_strs = [f'=== window length: {args.window_length}, sliding step: {args.sliding_step}', f'model: {args.pretrained_model}']
    print(output_strs[0])
    global_ates = []
    
    
    for scene in tqdm(scenes):
        if 'nusc' in args.dataset_dir: 
            gt_poses_path = Path(scene) / 'poses.txt'
            gt_poses = np.loadtxt(gt_poses_path)
            align_matrix_np = np.array([
                [1,  0,  0,  0],
                [0,  1,  0,  0],
                [0,  0,  1,  0],
                [0,  0,  0,  1],
            ], dtype=float)

            test_files = sorted(Path(scene).glob("*.jpg"))
            gt_intrinsic = np.genfromtxt(test_files[0].parent / 'intrinsics.txt')

        elif 'waymo' in args.dataset_dir:
            gt_poses_path = Path(scene) / 'poses.txt'
            gt_poses = np.loadtxt(gt_poses_path)
            align_matrix_np = np.array([
                [ 0,   0,  1,  0],
                [-1,   0,  0,  0],
                [ 0,  -1,  0,  0],
                [ 0,   0,  0,  1],
            ], dtype=float)
            
            test_files_path = Path(scene) / 'images'
            test_files = sorted(test_files_path.glob("*.jpg"))
            gt_intrinsic = np.genfromtxt(test_files[0].parent.parent / 'intrinsics.txt')
            
        elif 'kitti' in args.dataset_dir:
            gt_poses_path = Path("eval/kitti_gt_poses") / Path(scene).name
            gt_poses_path = gt_poses_path.with_suffix(".txt")
            gt_poses = np.loadtxt(gt_poses_path)
            align_matrix_np = np.array([
                [1,  0,  0,  0],
                [0,  1,  0,  0],
                [0,  0,  1,  0],
                [0,  0,  0,  1],
            ], dtype=float)

            test_files = sorted((Path(scene)/ 'image_2').glob("*.png"))
                                
            calib_path = test_files[0].parent.parent / 'calib.txt'
            calib = {}
            with open(calib_path, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()]).reshape(3, 4)
            gt_intrinsic = calib['P2'][:, :3]

        gt_poses = gt_poses[::args.eval_interval]
        test_files = test_files[::args.eval_interval]
        ates = []
        n = len(test_files)
        
        # results = benchmark_window_length_cached_pos(
        #     model=model,
        #     cfg=cfg,
        #     is_vggt=is_vggt,
        #     dtype=torch.bfloat16,   # or torch.float16
        #     N_list=[2, 4, 8, 16, 32, 64, 128],
        #     H=294,
        #     W=518,
        # )
        
        # print("\n=== Runtime vs Window Length (B=1, cached pos) ===")
        # print(f"{'N':>4} | {'mean (ms)':>10} | {'std (ms)':>9}")
        # print("-" * 30)
        # for N, v in results.items():
        #     print(f"{N:>4d} | {v['mean_ms']:>10.2f} | {v['std_ms']:>9.2f}")
        # results = benchmark_window_length_cached_pos_both_devices(
        #     model=model,
        #     cfg=cfg,
        #     is_vggt=is_vggt,
        #     N_list=(2, 4, 8, 16, 32, 64),
        #     H=294,
        #     W=518,
        #     C=3,
        #     gpu_dtype=torch.bfloat16,
        #     cpu_dtype=torch.float32,
        #     devices=("cuda", "cpu"),
        #     warmup=5,
        #     runs=20,
        #     print_per_N=True,
        #     print_summary=True,
        # )
        # print("\nRaw results dict:")
        # print(results)
        # return
    
    
        # window_length = min(n, args.window_length)
        # sliding_step = args.sliding_step
        
        # sliding_times = (n - window_length + sliding_step) // sliding_step  + ((n - window_length + sliding_step) % sliding_step != 0)# +((n % sliding_step) != (window_length - args.sliding_step))

        # for iter in range(sliding_times):
            
        #     # the step of last iteration is different
        #     if iter == sliding_times - 1 and ((n % sliding_step) != (window_length - args.sliding_step)) and sliding_times > 1:
        #         sliding_step = n - len(poses) 
        #         assert sliding_step < window_length
            
        #     if iter == 0:
        #         img_idx = [i + iter * args.sliding_step for i in range(window_length)]
        #     elif iter == sliding_times - 1:
        #         img_idx = [i for i in range(n-(sliding_step +1), n)]
        #     else:
        #         img_idx = [i + args.sliding_step for i in img_idx]
                
        #     windows_imgs = load_and_preprocess_images([test_files[i] for i in img_idx], target_size=cfg.img_size if 'official_vggt' not in args.pretrained_model else 518).to(device).unsqueeze(0)
        loader = make_dataloader(test_files=test_files, window_length=args.window_length, sliding_step=args.sliding_step,
                                    img_size=img_size, patch_size=patch_size, rescale_factor=args.rescale_factor, intrinsic=gt_intrinsic)
        sliding_times = loader.dataset.sliding_times
        for iter, windows in enumerate(loader):
            windows_imgs, sliding_step, img_idx = windows
            windows_imgs = windows_imgs.to(device)
            tgt_idx = [i for i in range(windows_imgs.size(1)-1)]
            ref_idx = [i+1 for i in range(windows_imgs.size(1)-1)]
            
            if  is_vggt or not OmegaConf.select(cfg, "model.cnn.enabled", default=False) :
                
                with torch.amp.autocast('cuda', dtype=dtype):
                    if is_vggt is False:
                        aggregated_tokens_list, _ = model.model.aggregator(windows_imgs)
                    else:
                        aggregated_tokens_list, _ = model.aggregator(windows_imgs)

                if is_vggt is False and cfg.model.camera_head.type is not None:
                
                    if iter == 0 or iter == sliding_times - 1:
                        # get position for pose estimation
                        B, S, C_in, H, W = windows_imgs.shape
                        
                        # we have S-1 pairs
                        pos = None
                        if model.model.aggregator.rope is not None:
                            pos = model.model.aggregator.position_getter(B * (S-1), 
                                                                        H // model.model.aggregator.patch_size, 
                                                                        W // model.model.aggregator.patch_size, device=windows_imgs.device)

                        if model.model.aggregator.patch_start_idx > 0:
                            # do not use position embedding for special tokens (camera and register tokens)
                            # so set pos to 0 for the special tokens
                            pos = pos + 1
                            pos_special = torch.zeros(B * (S-1), model.model.aggregator.patch_start_idx, 2).to(windows_imgs.device).to(pos.dtype)
                            pos = torch.cat([pos_special, pos], dim=1)
                        
                    

                    last_aggregated_tokens = aggregated_tokens_list[-1]
                    pred_pose_enc_ref2tgt = model.camera_head(last_aggregated_tokens[:, ref_idx],
                                                            last_aggregated_tokens[:, tgt_idx],
                                                            model.model.aggregator.patch_start_idx, 
                                                            windows_imgs, 
                                                            pos)
                        
                    ref2tgt_poses, _ = pose_encoding_to_extri_intri(pred_pose_enc_ref2tgt, windows_imgs.shape[-2:], pose_encoding_type=model.cfg.model.camera_head.pose_encoding_type)
                    ref2tgt_poses = ref2tgt_poses.squeeze(0)
                    # Get shape
                    N, _, _ = ref2tgt_poses.shape
                    # Bottom row [0, 0, 0, 1]
                    bottom_row = torch.tensor([0, 0, 0, 1], dtype=ref2tgt_poses.dtype, device=ref2tgt_poses.device)
                    bottom_row = bottom_row.view(1, 1, 4).expand(N, 1, 4)
                    ref2tgt_poses = torch.cat([ref2tgt_poses, bottom_row], dim=1)      # [N, 4, 4]
                else: 
                    if is_vggt is True:
                        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                        abs_poses, _ = pose_encoding_to_extri_intri(pose_enc , windows_imgs.shape[-2:], pose_encoding_type='absT_quaR_FoV') # vanilla vggt
                        abs_poses = abs_poses.squeeze(0)
                    else:
                        pose_enc = model.camera_head(aggregated_tokens_list)[-1] if not model.cfg.model.vggt.enable_camera else model.model.camera_head(aggregated_tokens_list)[-1]
                        abs_poses, _ = pose_encoding_to_extri_intri(pose_enc , windows_imgs.shape[-2:], pose_encoding_type=model.cfg.model.camera_head.pose_encoding_type) # 
                        abs_poses = abs_poses.squeeze(0)
                    # Get shape
                    N, _, _ = abs_poses.shape
                    # Bottom row [0, 0, 0, 1]
                    bottom_row = torch.tensor([0, 0, 0, 1], dtype=abs_poses.dtype, device=abs_poses.device)
                    bottom_row = bottom_row.view(1, 1, 4).expand(N, 1, 4)
                    abs_poses = torch.cat([abs_poses, bottom_row], dim=1)      # [N, 4, 4]
                    ref2tgt_poses = abs_poses[tgt_idx] @ torch.inverse(abs_poses[ref_idx]) 
                    
            else: 
                B, N, C, H, W = windows_imgs.shape
                windows_imgs = (windows_imgs - _resnet_mean) / _resnet_std
                pair_images_tgt = windows_imgs[:, tgt_idx]
                pair_images_ref = windows_imgs[:, ref_idx]
                
                pair_images_tgt = pair_images_tgt.view(B*(N-1), C, H, W)
                pair_images_ref = pair_images_ref.view(B*(N-1), C, H, W)
                
                ref2tgt_poses = model.pose_net(pair_images_ref, pair_images_tgt)
                # tgt2ref_poses = model.pose_net(pair_images_tgt, pair_images_ref)
                
                ref2tgt_poses = pose_vec2mat(ref2tgt_poses)
                # tgt2ref_poses = pose_vec2mat(tgt2ref_poses)
                
                bottom_row = torch.tensor([0, 0, 0, 1], dtype=ref2tgt_poses.dtype, device=ref2tgt_poses.device)
                bottom_row = bottom_row.view(1, 1, 4).expand(N-1, 1, 4)
                ref2tgt_poses = torch.cat([ref2tgt_poses, bottom_row], dim=1)      # [N, 4, 4]
                # ref2tgt_poses = torch.inverse(tgt2ref_poses)
            
            current_rel_poses = ref2tgt_poses.cpu().numpy()          # [W-1, 4, 4]
            current_window_abs_poses = rel_to_abs_poses(current_rel_poses)   # [W, 4, 4]

            if iter == 0:
                aligned_window_abs_poses = current_window_abs_poses
                last_window_abs_poses = aligned_window_abs_poses
                
                global_abs_poses = [x.copy() for x in aligned_window_abs_poses]

                poses = [p[:3, :].reshape(1, 12) for p in aligned_window_abs_poses]

            else:
                # overlap count = current window length - sliding_step
                previous_overlap_abs = last_window_abs_poses[sliding_step:]      # [O, 4, 4]
                current_overlap_abs  = current_window_abs_poses[:-sliding_step]  # [O, 4, 4]

                if previous_overlap_abs.shape[0] > 0 and current_overlap_abs.shape[0] > 0:
                    s, R, t = estimate_overlap_sim3(
                        current_overlap_abs=current_overlap_abs,
                        previous_overlap_abs=previous_overlap_abs,
                        min_scale=0.1,
                        max_scale=10.0,
                    )
                else:
                    s, R, t = 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

                aligned_window_abs_poses = apply_sim3_to_abs_poses(
                    current_window_abs_poses, s, R, t
                )

                # last_window_abs_poses = aligned_window_abs_poses
                
                # debug_dir = output_dir / "debug_alignment" / Path(scene).stem
                # debug_path = debug_dir / f"iter_{iter:04d}.png"

                # plot_window_alignment_debug(
                #     current_window_abs_poses=current_window_abs_poses,
                #     aligned_window_abs_poses=aligned_window_abs_poses,
                #     last_window_abs_poses=last_window_abs_poses,
                #     sliding_step=sliding_step,
                #     save_path=str(debug_path),
                #     title=f"{Path(scene).stem} | iter={iter} | W={windows_imgs.size(1)} | step={sliding_step} | scale={s:.4f}",
                # )
                
                # append only the NEW tail
                new_abs_poses = aligned_window_abs_poses[-sliding_step:]
                for abs_pose in new_abs_poses:
                    global_abs_poses.append(abs_pose.copy())
                    poses.append(abs_pose[:3, :].reshape(1, 12))
                # updata last_window_poses
                last_window_abs_poses = np.stack(global_abs_poses[-windows_imgs.size(1):], axis=0)
                # assert ref2tgt_poses.shape[0] == last_window_poses.shape[0]
                # last_window_poses = ref2tgt_poses.cpu().numpy()
            
            gt_poses_window = [gt_poses[i] for i in img_idx] 
            gt_poses_window = torch.from_numpy(np.array(gt_poses_window))
            gt_poses_window = first_frame_as_base(gt_poses_window.view(*gt_poses_window.shape[:-1], 3, 4).squeeze(0))
            align_matrix = torch.from_numpy(align_matrix_np).unsqueeze(0)
            gt_poses_window = torch.inverse(align_matrix) @ gt_poses_window @ align_matrix
            base_pose = torch.eye(4)
            pred_window_poses = [base_pose]
                
            for ref2tgt_pose in ref2tgt_poses:
                base_pose = base_pose @ ref2tgt_pose.cpu()
                pred_window_poses.append(base_pose)
                
            pred_poses = torch.stack(pred_window_poses)
            local_xyzs, gt_local_xyzs = pred_poses[:, :3, 3].numpy(), gt_poses_window[:, :3, 3].numpy() # [N, 3]

            ates.append(compute_ate(gt_local_xyzs, local_xyzs))

            if is_vggt or not OmegaConf.select(cfg, "model.cnn.enabled", default=False) :
                del aggregated_tokens_list
                torch.cuda.empty_cache()
        assert len(poses) == n
        

        poses = np.concatenate(poses, axis=0)
        scene_path = Path(scene)
        filename = output_dir / 'predicted_poses'/ (scene_path.stem + ".txt")
        gt_filename = output_dir / 'gt_poses'/ (scene_path.stem + ".txt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs(os.path.dirname(gt_filename), exist_ok=True)
        np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
        np.savetxt(gt_filename, gt_poses, delimiter=' ', fmt='%1.8e')
        output_strs.append(f'\n{scene_path} -- Local ATE: {np.mean(ates):0.3f} ± {np.std(ates):0.3f}')
        print(output_strs[-1])
        poses = poses.reshape(-1, 3, 4)
        gt_poses = torch.from_numpy(np.array(gt_poses))
        gt_poses = first_frame_as_base(gt_poses.view(*gt_poses.shape[:-1], 3, 4).squeeze(0))
        gt_poses = torch.inverse(align_matrix) @ gt_poses @ align_matrix
        global_xyzs, gt_global_xyzs = poses[:, :3, 3], gt_poses[:, :3, 3].numpy() # [N, 3]
        global_ate = compute_ate(gt_global_xyzs, global_xyzs)
        global_ates.append(global_ate)
        output_strs.append(f'Global ATE:{global_ate}')
        print(output_strs[-1])
        
    # Results
    output_strs.append(f'\nATE Trajectory error (window length={args.window_length}, sliding step={args.sliding_step}):  ')
    output_strs.append(f'Mean:   {np.mean(global_ates)}')
    output_strs.append(f'std:    {np.std(global_ates)}')
    output_strs.append('--')
    output_strs.append(f'Min:    {np.min(global_ates)}')
    output_strs.append(f'Median: {np.median(global_ates)}')
    output_strs.append(f'Max:    {np.max(global_ates)}')
    # Write to terminal / out_path
    write_to_file(output_strs, txt_path)

@torch.no_grad()
def benchmark_window_length_cached_pos(
    model,
    cfg,
    is_vggt: bool,
    N_list=(2, 4, 6, 8, 10),
    H=294,
    W=518,
    C=3,
    dtype=torch.bfloat16,
    device="cuda",
    warmup=5,
    runs=20,
):
    """
    Benchmark forward-pass latency vs window length N with:
    - B = 1
    - cached positional encodings
    - aggregator + camera_head
    """

    model.eval()
    results = {}

    for N in N_list:
        B = 1
        windows_imgs = torch.randn(
            B, N, C, H, W,
            device=device,
            dtype=dtype,
        )
        tgt_idx = [i for i in range(windows_imgs.size(1)-1)]
        ref_idx = [i+1 for i in range(windows_imgs.size(1)-1)]
        # -------------------------
        # Precompute pos ONCE
        # -------------------------
        pos = None
        if (not is_vggt) and cfg.model.camera_head.type is not None:
            agg = model.model.aggregator
            _, S, _, H_, W_ = windows_imgs.shape

            if agg.rope is not None:
                pos = agg.position_getter(
                    B * (S - 1),
                    H_ // agg.patch_size,
                    W_ // agg.patch_size,
                    device=device,
                )

            if agg.patch_start_idx > 0 and pos is not None:
                pos = pos + 1
                pos_special = torch.zeros(
                    B * (S - 1),
                    agg.patch_start_idx,
                    2,
                    device=device,
                    dtype=pos.dtype,
                )
                pos = torch.cat([pos_special, pos], dim=1)

        # -------------------------
        # Warm-up
        # -------------------------
        for _ in range(warmup):
            with torch.amp.autocast("cuda", dtype=dtype):
                if not OmegaConf.select(cfg, "model.cnn.enabled", default=False) or is_vggt:
                    if not is_vggt:
                        aggregated_tokens_list, _ = model.model.aggregator(windows_imgs)
                    else:
                        aggregated_tokens_list, _ = model.aggregator(windows_imgs)

                    if (not is_vggt) and cfg.model.camera_head.type is not None:
                        last_tokens = aggregated_tokens_list[-1]
                        _ = model.camera_head(
                            last_tokens[:, tgt_idx],
                            last_tokens[:, ref_idx],
                            model.model.aggregator.patch_start_idx,
                            windows_imgs,
                            pos,
                        )

        torch.cuda.synchronize()

        # -------------------------
        # Timed runs
        # -------------------------
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_ms = []

        for _ in range(runs):
            start.record()

            with torch.amp.autocast("cuda", dtype=dtype):
                if not OmegaConf.select(cfg, "model.cnn.enabled", default=False) or is_vggt:
                    if not is_vggt:
                        aggregated_tokens_list, _ = model.model.aggregator(windows_imgs)
                    else:
                        aggregated_tokens_list, _ = model.aggregator(windows_imgs)

                    if (not is_vggt) and cfg.model.camera_head.type is not None:
                        last_tokens = aggregated_tokens_list[-1]
                        _ = model.camera_head(
                            last_tokens[:, tgt_idx],
                            last_tokens[:, ref_idx],
                            model.model.aggregator.patch_start_idx,
                            windows_imgs,
                            pos,
                        )

            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        results[N] = {
            "mean_ms": sum(times_ms) / len(times_ms),
            "std_ms": torch.tensor(times_ms).std().item(),
        }

        print(
            f"N={N:2d} | "
            f"mean = {results[N]['mean_ms']:.2f} ms | "
            f"std = {results[N]['std_ms']:.2f} ms"
        )

    return results

import time
from typing import Iterable, Dict, Any, Optional

import torch
from omegaconf import OmegaConf


@torch.no_grad()
def benchmark_window_length_cached_pos_both_devices(
    model,
    cfg,
    is_vggt: bool,
    N_list: Iterable[int] = (2, 4, 6, 8, 10),
    H: int = 154,
    W: int = 518,
    C: int = 3,
    gpu_dtype: torch.dtype = torch.bfloat16,
    cpu_dtype: torch.dtype = torch.float32,
    devices: Iterable[str] = ("cuda", "cpu"),
    warmup: int = 5,
    runs: int = 20,
    pin_memory: bool = True,
    non_blocking: bool = True,
    print_per_N: bool = True,
    print_summary: bool = True,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Benchmark forward-pass latency vs window length N on both GPU and CPU with:
    - B = 1
    - cached positional encodings (cached per N per device)
    - aggregator + camera_head

    Returns:
      results[device][N] = {"mean_ms": float, "std_ms": float}
    """

    def _sync_if_cuda(dev: str) -> None:
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _now_s(dev: str) -> float:
        # For CPU timing we use perf_counter; for CUDA we use cuda Events.
        return time.perf_counter()

    def _std_ms(vals_ms):
        if len(vals_ms) <= 1:
            return 0.0
        return torch.tensor(vals_ms, dtype=torch.float64).std(unbiased=True).item()

    # Respect user-provided device list but skip CUDA if unavailable
    devices = list(devices)
    if any(d.startswith("cuda") for d in devices) and (not torch.cuda.is_available()):
        devices = [d for d in devices if not d.startswith("cuda")]

    model.eval()
    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for dev in devices:
        dev_results: Dict[int, Dict[str, float]] = {}
        device = torch.device(dev)

        # Move model once per device
        model = model.to(device)

        # Choose dtype per device
        dtype = gpu_dtype if dev.startswith("cuda") else cpu_dtype

        if print_summary:
            print("\n" + "=" * 80)
            print(f"Benchmark on device: {dev} | dtype: {dtype} | B=1 | HxW={H}x{W}")
            print("=" * 80)

        for N in N_list:
            B = 1

            # Input creation: for CPU, pin+non_blocking is irrelevant; for CUDA it can help if you later copy.
            windows_imgs = torch.randn(B, N, C, H, W, device=device, dtype=dtype)

            tgt_idx = [i for i in range(windows_imgs.size(1) - 1)]
            ref_idx = [i + 1 for i in range(windows_imgs.size(1) - 1)]

            # -------------------------
            # Precompute pos ONCE (per N)
            # -------------------------
            pos: Optional[torch.Tensor] = None
            if (not is_vggt) and (OmegaConf.select(cfg, "model.camera_head.type", default=None) is not None):
                agg = model.model.aggregator
                _, S, _, H_, W_ = windows_imgs.shape

                if getattr(agg, "rope", None) is not None:
                    pos = agg.position_getter(
                        B * (S - 1),
                        H_ // agg.patch_size,
                        W_ // agg.patch_size,
                        device=device,
                    )

                if getattr(agg, "patch_start_idx", 0) > 0 and pos is not None:
                    pos = pos + 1
                    pos_special = torch.zeros(
                        B * (S - 1),
                        agg.patch_start_idx,
                        2,
                        device=device,
                        dtype=pos.dtype,
                    )
                    pos = torch.cat([pos_special, pos], dim=1)

            # -------------------------
            # Warm-up
            # -------------------------
            for _ in range(warmup):
                if dev.startswith("cuda"):
                    with torch.amp.autocast("cuda", dtype=dtype):
                        if (not OmegaConf.select(cfg, "model.cnn.enabled", default=False)) or is_vggt:
                            if not is_vggt:
                                aggregated_tokens_list, _ = model.model.aggregator(windows_imgs)
                            else:
                                aggregated_tokens_list, _ = model.aggregator(windows_imgs)

                            if (not is_vggt) and (OmegaConf.select(cfg, "model.camera_head.type", default=None) is not None):
                                last_tokens = aggregated_tokens_list[-1]
                                _ = model.camera_head(
                                    last_tokens[:, tgt_idx],
                                    last_tokens[:, ref_idx],
                                    model.model.aggregator.patch_start_idx,
                                    windows_imgs,
                                    pos,
                                )
                else:
                    # CPU: no autocast by default (keeps it stable/reproducible)
                    if (not OmegaConf.select(cfg, "model.cnn.enabled", default=False)) or is_vggt:
                        if not is_vggt:
                            aggregated_tokens_list, _ = model.model.aggregator(windows_imgs)
                        else:
                            aggregated_tokens_list, _ = model.aggregator(windows_imgs)

                        if (not is_vggt) and (OmegaConf.select(cfg, "model.camera_head.type", default=None) is not None):
                            last_tokens = aggregated_tokens_list[-1]
                            _ = model.camera_head(
                                last_tokens[:, tgt_idx],
                                last_tokens[:, ref_idx],
                                model.model.aggregator.patch_start_idx,
                                windows_imgs,
                                pos,
                            )

            _sync_if_cuda(dev)

            # -------------------------
            # Timed runs
            # -------------------------
            times_ms = []

            if dev.startswith("cuda"):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                for _ in range(runs):
                    start.record()
                    with torch.amp.autocast("cuda", dtype=dtype):
                        if (not OmegaConf.select(cfg, "model.cnn.enabled", default=False)) or is_vggt:
                            if not is_vggt:
                                aggregated_tokens_list, _ = model.model.aggregator(windows_imgs)
                            else:
                                aggregated_tokens_list, _ = model.aggregator(windows_imgs)

                            if (not is_vggt) and (OmegaConf.select(cfg, "model.camera_head.type", default=None) is not None):
                                last_tokens = aggregated_tokens_list[-1]
                                _ = model.camera_head(
                                    last_tokens[:, tgt_idx],
                                    last_tokens[:, ref_idx],
                                    model.model.aggregator.patch_start_idx,
                                    windows_imgs,
                                    pos,
                                )
                    end.record()
                    torch.cuda.synchronize()
                    times_ms.append(start.elapsed_time(end))

            else:
                # CPU timing (wall-clock). For more stable CPU numbers:
                # - consider setting torch.set_num_threads(k)
                # - consider setting torch.backends.mkldnn.enabled etc., depending on your environment
                for _ in range(runs):
                    t0 = _now_s(dev)
                    if (not OmegaConf.select(cfg, "model.cnn.enabled", default=False)) or is_vggt:
                        if not is_vggt:
                            aggregated_tokens_list, _ = model.model.aggregator(windows_imgs)
                        else:
                            aggregated_tokens_list, _ = model.aggregator(windows_imgs)

                        if (not is_vggt) and (OmegaConf.select(cfg, "model.camera_head.type", default=None) is not None):
                            last_tokens = aggregated_tokens_list[-1]
                            _ = model.camera_head(
                                last_tokens[:, tgt_idx],
                                last_tokens[:, ref_idx],
                                model.model.aggregator.patch_start_idx,
                                windows_imgs,
                                pos,
                            )
                    t1 = _now_s(dev)
                    times_ms.append((t1 - t0) * 1000.0)

            mean_ms = sum(times_ms) / len(times_ms)
            std_ms = _std_ms(times_ms)

            dev_results[int(N)] = {"mean_ms": float(mean_ms), "std_ms": float(std_ms)}

            if print_per_N:
                fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
                print(
                    f"[{dev}] N={N:3d} | "
                    f"mean = {mean_ms:8.2f} ms | std = {std_ms:6.2f} ms | "
                    f"FPS ≈ {fps:6.2f}"
                )

        results[dev] = dev_results

        if print_summary:
            # Device-level summary table
            print("\nSummary:", dev)
            print(f"{'N':>6} | {'mean (ms)':>10} | {'std (ms)':>9} | {'FPS':>8}")
            print("-" * 44)
            for N in N_list:
                m = dev_results[int(N)]["mean_ms"]
                s = dev_results[int(N)]["std_ms"]
                fps = 1000.0 / m if m > 0 else float("inf")
                print(f"{int(N):6d} | {m:10.2f} | {s:9.2f} | {fps:8.2f}")

    return results

if __name__ == '__main__':
    main()