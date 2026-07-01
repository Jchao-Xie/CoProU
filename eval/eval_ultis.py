## reference from https://github.com/YihongSun/Dynamo-Depth.git
import os, cv2
import os.path as osp
import numpy as np
import imageio
import torch
import matplotlib as mpl
import matplotlib.cm as cm
import torch.utils.data as data
from models.vggt.utils.load_fn import load_and_preprocess_images
from path import Path


def get_filenames(segment_name, cam_name, eval_img_type, eval_img_ext, data_path):
    """ Return the list of filenames given a segment path
    """
    cam_name, img_type, img_ext = cam_name, eval_img_type, eval_img_ext
    rgb_dir_path = osp.join(data_path, segment_name, cam_name, 'rgb', img_type)
    frame_indices = sorted([int(osp.splitext(f)[0]) for f in os.listdir(rgb_dir_path) if osp.splitext(f)[1] == img_ext])
    return [f'{segment_name} {i}' for i in frame_indices]

def is_edge(filename, opt):
    """ Determine if the given filename is on the edge of the sequence given the range of opt.frame_ids
        Only used during evaluation
    """
    cam_name, img_type, img_ext = opt.cam_name, opt.eval_img_type, opt.eval_img_ext
    seg_name, frame_index = filename.split()[0], int(filename.split()[1])
    left_index, right_index = frame_index + np.min(opt.frame_ids), frame_index + np.max(opt.frame_ids)
    left_bound = osp.join(opt.data_path, seg_name, cam_name, 'rgb', img_type, f'{left_index:06}{img_ext}')
    right_bound = osp.join(opt.data_path, seg_name, cam_name, 'rgb', img_type, f'{right_index:06}{img_ext}')
    return (not osp.exists(left_bound)) or (not osp.exists(right_bound))


class segment_dataset(data.Dataset):
    """load the image sequence in segment for odometry evaluation

    Args:
        track_length (int): Length of windows for local odometry evaluation
        segment_path (str): Path to segment+
        skip_frames (int): interval of adjacent images
    """
    
    def __init__(self, track_length, segment_path, skip_frames=1):
        super().__init__()
        self.track_length = track_length
        self.segment_path = segment_path
        self.k = skip_frames
        self.crawl_folders(track_length)
        
    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        intrinsics = np.genfromtxt(osp.join(self.segment_path, 'intrinsics.txt')).astype(np.float32)
        poses =  np.genfromtxt(osp.join(self.segment_path, 'poses.txt')).astype(np.float32)
        imgs = sorted(Path(self.segment_path).files('*.jpg'))
        for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
            sample = {'intrinsics': intrinsics, 'imgs': [], 'gt_poses': []}
            for j in shifts:
                sample['imgs'].append(imgs[i+j])
                sample['gt_poses'].append(poses[i+j])
            sample['gt_poses'] = np.array(sample['gt_poses'])
            sequence_set.append(sample)
        self.samples = sequence_set
    
    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = load_and_preprocess_images(sample['imgs'])
        gt_poses = sample['gt_poses']
        
        return imgs, gt_poses
     
    def __len__(self):
        return len(self.samples)
    
    
from models.vggt.utils.geometry import closed_form_inverse_se3
def first_frame_as_base(extrinsic):
    N = extrinsic.shape[0]
    bottom = torch.tensor([0, 0, 0, 1], device=extrinsic.device, dtype=extrinsic.dtype)
    bottom = bottom.view(1, 1, 4).expand(N, -1, -1)  # (N, 1, 4)
    extrinsic_h = torch.cat([extrinsic, bottom], dim=1)  # (N, 4, 4)
    base_extrinsic = extrinsic_h[0]
    inv_base_extrinsic = closed_form_inverse_se3(base_extrinsic.unsqueeze(0))
    extrinsic = inv_base_extrinsic @ extrinsic_h 
    # extrinsic = extrinsic[:, :3, :]
    
    return extrinsic


import torch
import matplotlib.pyplot as plt

def vis_depth(depth_p, name):
    # Prepare tensors
    # disp_vis = disp_p.squeeze().detach().cpu()
    depth_vis = depth_p.detach().cpu()
    disp_vis = 1 / depth_vis

    # Optional: normalize for better contrast
    disp_vis = (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min() + 1e-8)
    depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)

    # Plot and save
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(disp_vis.numpy(), cmap='plasma')
    plt.colorbar()
    plt.title("Disparity")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_vis.numpy(), cmap='plasma')
    plt.colorbar()
    plt.title("Depth")

    plt.tight_layout()
    plt.savefig(f"disp_depth_vis_{name}.png", dpi=200)
    plt.close()