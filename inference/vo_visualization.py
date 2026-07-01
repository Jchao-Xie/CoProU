import torch
import time
import os
import glob
# from skimage.transform import resize as imresize
from PIL import Image
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from inverse_warp import pose_vec2mat
from inverse_warp import *
from point_cloud import iter_get_uncertainty

import cv2
import imageio



from models.model_module import coprou
from omegaconf import OmegaConf
from models.vggt.models.vggt import VGGT
from models.vggt.utils.load_fn import load_and_preprocess_images

def model_from_pretrained(model_path):
    cfg_path = Path(model_path).parent / 'hparams.yaml'
    cfg = OmegaConf.load(cfg_path)
    model = coprou(cfg=cfg)
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Loaded model weights from {model_path}")
    
    return model, cfg
    
def readlines(filename):
    """ Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def colorize_map(map_2d, cmap="inferno", normalize=True, scale=1):
    if isinstance(map_2d, torch.Tensor):
        map_2d = map_2d.detach().cpu().numpy()
    map_2d = np.squeeze(map_2d)
    map_norm = cv2.normalize(scale * map_2d, None, 0, 255, cv2.NORM_MINMAX) if normalize else scale * map_2d
    map_uint8 = map_norm.astype(np.uint8)

    # Apply selected colormap
    if cmap == "magma":
        return cv2.applyColorMap(map_uint8, cv2.COLORMAP_MAGMA)
    elif cmap == "viridis":
        return cv2.applyColorMap(map_uint8, cv2.COLORMAP_VIRIDIS)
    else:
        return cv2.applyColorMap(map_uint8, cv2.COLORMAP_INFERNO)
    
    
    
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained-model', default=None, help='path to pre-trained CoProU-VO-MF model, if not provided going for VGGT')
    parser.add_argument('--file-names', default="inference/video_file_paths.txt", help='path to folder to visualize')
    parser.add_argument("--conf-range", default=5, type=int, help="range of uncertainty windows")
    parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    args = parser.parse_args()

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
    
    file_path = args.file_names
    filenames = readlines(file_path)
    
    for filename in filenames:

        test_files = []
        for ext in args.img_exts:
            test_files += glob.glob(os.path.join(filename, f"*.{ext}"))
        test_files.sort()
        
        # Set output video file
        if 'kitti' in filename:
            video_path = os.path.join('video_output', 'kitti', filename.split(os.sep)[-4] + ".mp4")
        elif 'waymo' in filename:
            video_path = os.path.join('video_output', 'waymo', filename.split(os.sep)[-2] + ".mp4")
        else:
            video_path = os.path.join('video_output', 'nuscenes', os.path.basename(filename) + ".mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)        
        frame_width = load_and_preprocess_images([test_files[0]]).size(-1) * 3
        frame_height = load_and_preprocess_images([test_files[0]]).size(-2)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
        out = cv2.VideoWriter(video_path, fourcc, 10, (frame_width, frame_height))

        print('{} files to test'.format(len(test_files)))

        n = len(test_files)
        
        assert args.conf_range % 2 == 1 # odd
        
        begin, end = args.conf_range // 2, n - args.conf_range

        for iter in tqdm(range(begin, end)):
            
            if iter == begin:
                img_idxs = [i for i in range(iter - args.conf_range // 2, iter + args.conf_range // 2 + 1)]
                input_imgs = load_and_preprocess_images([test_files[i] for i in img_idxs]).to(device).unsqueeze(0)
            else:
                img_idxs = [i+1 for i in img_idxs]
                input_imgs = torch.cat([input_imgs[:, 1:], load_and_preprocess_images([test_files[img_idxs[-1]]]).to(device).unsqueeze(0)],
                                               dim=1)
            
            
            with torch.amp.autocast('cuda', dtype=dtype):
                aggregated_tokens_list, ps_idx = model.model.aggregator(input_imgs) if args.pretrained_model is not None else model.aggregator(input_imgs)
                
            # Predict Depth Maps
            if args.pretrained_model is None:
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, input_imgs, ps_idx)
            elif model.cfg.model.vggt.enable_depth:
                depth_map, depth_conf = model.model.depth_head(aggregated_tokens_list, input_imgs, ps_idx)
            else:
                depth_map = model.depth_head(aggregated_tokens_list, input_imgs, ps_idx)
                
            uncertainty_map = iter_get_uncertainty(model, aggregated_tokens_list, input_imgs, ps_idx, args.conf_range)

            depth_map = depth_map[:, depth_map.size(1)//2].squeeze()
            uncertainty_map = uncertainty_map[:, uncertainty_map.size(1)//2].squeeze()
            depth_color = colorize_map(1 / depth_map, cmap="magma")
            uncertainty_color = colorize_map(uncertainty_map, cmap="viridis")

            # Stack: [depth | RGB | uncertainty]
            rgb_img = Image.open(test_files[iter])
            H, W = depth_map.shape[:2]
            rgb_img = rgb_img.resize((W, H)) 
            rgb = np.array(rgb_img)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            combined_frame = np.hstack((depth_color, rgb_bgr, uncertainty_color))
            assert combined_frame.shape[1] == frame_width and combined_frame.shape[0] == frame_height


            # Write frame
            out.write(combined_frame.astype(np.uint8))


        out.release()
        print(f"Saved video to: {video_path}")


if __name__ == '__main__':
    main()
