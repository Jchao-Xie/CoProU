# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse

from .global_odometry import GlobalEvalOdom
import numpy as np
from data.nuscenes_config.splits import val as validation_list
from glob import glob
import os


parser = argparse.ArgumentParser(description='CoProU-VO-MF evaluation')
parser.add_argument('--result', type=str, required=True,
                    help="Result directory")
parser.add_argument('--align', type=str,
                    choices=['scale', 'scale_7dof', '7dof', '6dof'],
                    default=None,
                    help="alignment type")
parser.add_argument('--seqs',
                    nargs="+",
                    type=str,
                    help="sequences to be evaluated",
                    default=None)
parser.add_argument("--test", action='store_true', help="using test dataset")
parser.add_argument("--interval", type=int, default=1, help="define the interval of target- and reference image")
parser.add_argument('--dataset', type=str,
                    choices=['waymo', 'kitti', 'nusc'],
                    default=None,
                    help="alignment type")

args = parser.parse_args()

eval_tool = GlobalEvalOdom()
gt_dir_dict = {
                "waymo": 'eval/waymo_gt_poses',
                "nusc": 'eval/nusc_gt_poses',
                "kitti": 'eval/kitti_gt_poses'
                }

result_dir = args.result if not args.test else args.result + 'test/'
available_seqs = sorted(glob(os.path.join(result_dir, "*.txt")))
gt_root = os.path.join(os.path.dirname(result_dir), "gt_poses")

if 'waymo' in args.dataset:
    gt_dir = gt_root if os.path.isdir(gt_root) else gt_dir_dict["waymo"]
    axis_transformation = np.array([
                [ 0,   0,  1,  0],
                [-1,   0,  0,  0],
                [ 0,  -1,  0,  0],
                [ 0,   0,  0,  1],
            ], dtype=float)
    with open('vggt_training/data/datasets/waymo_day_light_split_val.txt', 'r') as f:
        day_light_sequences = [line.strip() for line in f if line.strip()]
    eval_seqs = [seq for seq in available_seqs 
                if os.path.basename(seq).split(".")[0]
                in day_light_sequences]
    
elif 'nusc' in args.dataset:
    gt_dir = gt_root if os.path.isdir(gt_root) else gt_dir_dict["nusc"]
    axis_transformation = np.array([
                [1,  0,  0,  0],
                [0,  1,  0,  0],
                [0,  0,  1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
    eval_seqs = [
                    seq for seq in available_seqs
                    if os.path.basename(seq).split(".")[0].split("_")[0] in validation_list
                ]
elif 'kitti' in args.dataset:
    gt_dir = gt_root if os.path.isdir(gt_root) else gt_dir_dict["kitti"]
    axis_transformation = np.array([
                [1,  0,  0,  0],
                [0,  1,  0,  0],
                [0,  0,  1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
    test_scences = ["01", "09", "10"]
    eval_seqs = [
                seq for seq in available_seqs
                if os.path.basename(seq).split(".")[0] in test_scences
            ]
else:
    raise RuntimeError("not supporting this dataset")

if args.interval > 1:
    result_dir = result_dir + f"interval_{args.interval}/"

continue_flag = "y"
print("Evaluate result in {}.".format(result_dir))
if continue_flag == "y":
    eval_tool.eval(
        gt_dir,
        result_dir,
        eval_seqs=eval_seqs,
        alignment=args.align,
        seqs=args.seqs,
        interval=args.interval,
        axis_transformation = axis_transformation,
        dataset=args.dataset,
    )
else:
    print("Double check the path!")
