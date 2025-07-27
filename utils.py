from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
import os
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt



def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.45 + tensor.numpy()*0.225
    return array


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix, filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
                            save_path/'{}_model_best.pth.tar'.format(prefix))
            
def unnormalize_and_save(image, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], name="unnormalized_image.png"):
    """
    Unnormalize a single image and save it for visualization.
    Args:
        image: Normalized image tensor of shape (C, H, W).
        mean: List of mean values for each channel.
        std: List of standard deviation values for each channel.
        name: file name
    Returns:
        Unnormalized image tensor.
    """
    unnormalized_image = image.clone().detach()
    for t, m, s in zip(unnormalized_image, mean, std):
        t.mul_(s).add_(m)

    # Convert the unnormalized tensor to a PIL image for saving
    pil_image = TF.to_pil_image(unnormalized_image)
    pil_image.save(os.path.join('visualization', name))

    return unnormalized_image

def depth_visualization(depth1, depth2, diff_depth, name):
    
    depth1 = depth1.clone().detach()
    depth2 = depth2.clone().detach()
    diff_depth = diff_depth.clone().detach()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(depth1.squeeze(0).cpu().numpy() if depth1.size(0)==1 else depth1.permute(1, 2, 0).cpu().numpy(), cmap='viridis' if depth1.size(0) == 1 else None, interpolation='nearest')
    plt.colorbar(label='Depth Value')
    plt.title('Original Depth')
    plt.axis('off')

    # Visualize inverse depth
    plt.subplot(1, 3, 2)
    plt.imshow(depth2.squeeze(0).cpu().numpy() if depth2.size(0)==1 else depth2.permute(1, 2, 0).cpu().numpy(), cmap='viridis' if depth2.size(0) == 1 else None, interpolation='nearest')
    plt.colorbar(label='Depth Value')
    plt.title('warped_depth')
    plt.axis('off')

    # Visualize the mask
    plt.subplot(1, 3, 3)
    plt.imshow(diff_depth.squeeze(0).cpu().numpy() if diff_depth.size(0)==1 else diff_depth.permute(1, 2, 0).mean(2).cpu().numpy(), cmap='viridis' if diff_depth.size(0) == 1 else None, interpolation='nearest')
    plt.colorbar(label='Error Value')
    plt.title('depth error')
    plt.axis('off')
    
    # Save the figure
    save_path = f"visualization/{name}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def mask_visualization(mask, name):
    mask = mask.clone().detach()
    mask = mask.squeeze(0).cpu().numpy()
    # Visualizing the weight mask
    plt.figure(figsize=(10, 5))
    plt.title("Weight Mask Visualization")
    plt.imshow(mask, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=name)
    plt.axis('off')
    save_path = f"visualization/{name}.png"
    plt.savefig(save_path)
    plt.close()

def heapmap_visualization(heatmap, name):
    heatmap = heatmap.clone().detach()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(heatmap.squeeze(0).cpu().numpy(), cmap='viridis')
    plt.axis('off')
    # plt.colorbar(label='Error Value')
    
    # Save the figure
    save_path = f"visualization/{name}.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def error_visualization(diff_depth, name):
    
    diff_depth = diff_depth.clone().detach()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(diff_depth.permute(1, 2, 0).mean(2).cpu().numpy(), cmap='inferno')
    plt.axis('off')
    # plt.colorbar(label='Error Value')
    
    # Save the figure
    save_path = f"visualization/{name}.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


