# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse

from kitti_odometry import KittiEvalOdom

parser = argparse.ArgumentParser(description='KITTI evaluation')
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

args = parser.parse_args()

eval_tool = KittiEvalOdom()
gt_dir = "./nusc_eval/gt_poses/"
result_dir = args.result if not args.test else args.result + 'test/'
if args.interval > 1:
    result_dir = result_dir + f"interval_{args.interval}/"

continue_flag = "y"
print("Evaluate result in {}.".format(result_dir))
if continue_flag == "y":
    eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
        interval=args.interval,
    )
else:
    print("Double check the path!")
