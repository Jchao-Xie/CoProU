import torch
import time

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

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument('--pretrained-dispnet', required=True, help='path to pre-trained dispnet model')
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def depth_anything_transform(raw_image, input_size=256):        
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_AREA,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # h, w = raw_image.shape[:2]
    
    image = raw_image / 255.0
    
    image = transform({'image': image})['image']
    
    
    # image = torch.from_numpy(image).unsqueeze(0)
    
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # image = image.to(DEVICE)
    
    return image

def load_tensor_image(filename, args):
    # img = imread(filename).astype(np.float32)
    # h, w, _ = img.shape
    # if (not args.no_resize) and (h != args.img_height or w != args.img_width):
    #     img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    # img = np.transpose(img, (2, 0, 1))
    # Read the image using Pillow
    img = Image.open(filename).convert("RGB")
    img = np.array(img).astype(np.float32)  # Convert to NumPy array with float32

    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        # Resize the image using Pillow
        img = Image.fromarray(img.astype(np.uint8))  # Convert back to Pillow Image
        img = img.resize((args.img_width, args.img_height), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32)  # Convert back to NumPy array
    img = np.transpose(img, (2, 0, 1))  # Rearrange dimensions
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

def colorize_map(map_2d, cmap="inferno"):
    if isinstance(map_2d, torch.Tensor):
        map_2d = map_2d.detach().cpu().numpy()
    map_2d = np.squeeze(map_2d)
    map_norm = cv2.normalize(map_2d, None, 0, 255, cv2.NORM_MINMAX)
    map_uint8 = map_norm.astype(np.uint8)

    # Apply selected colormap
    if cmap == "magma":
        return cv2.applyColorMap(map_uint8, cv2.COLORMAP_MAGMA)
    elif cmap == "viridis":
        return cv2.applyColorMap(map_uint8, cv2.COLORMAP_VIRIDIS)
    else:
        return cv2.applyColorMap(map_uint8, cv2.COLORMAP_INFERNO)

def fps_testing(pose_net, tensor_img1, tensor_img2, n_iters=100):
    def _test_on_device(device_str):
        device = torch.device(device_str)
        pose_net_device = pose_net.to(device).eval()
        img1 = tensor_img1.to(device)
        img2 = tensor_img2.to(device)

        if device_str == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Warm-up
            for _ in range(10):
                _ = pose_net_device(img1, img2)

            torch.cuda.synchronize()
            start_event.record()
            for _ in range(n_iters):
                with torch.no_grad():
                    _ = pose_net_device(img1, img2)
                    torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_time_s = elapsed_time_ms / n_iters / 1000

        else:  # CPU timing using time.time()
            # Warm-up
            for _ in range(10):
                _ = pose_net_device(img1, img2)

            start_time = time.time()
            for _ in range(n_iters):
                with torch.no_grad():
                    _ = pose_net_device(img1, img2)
            end_time = time.time()

            elapsed_time_s = end_time - start_time
            avg_time_s = elapsed_time_s / n_iters

        fps = 1.0 / avg_time_s
        print(f"[{device_str.upper()}] Average FPS over {n_iters} iterations: {fps:.2f}")
        return fps

    # Run test on both GPU and CPU (if available)
    _test_on_device('cpu')
    
    if torch.cuda.is_available():
        _test_on_device('cuda')
    else:
        print("CUDA not available, skipping GPU test.")

@torch.no_grad()
def main():
    args = parser.parse_args()

    weights_pose = torch.load(args.pretrained_posenet)
    pose_net = models.PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()
    test_fps = True
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = fine_tuning_DepthAnythingV2(**model_configs[args.encoder])
    disp_net = depth_anything.to(device)
    weights_disp = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights_disp['state_dict'], strict=False)
    disp_net.eval()

    image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()
    
    # Set output video file
    video_path = "seq" + f"{args.sequence}" + ".mp4"
    frame_width = args.img_width * 3
    frame_height = args.img_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(video_path, fourcc, 10, (frame_width, frame_height))

    print('{} files to test'.format(len(test_files)))

    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)

    for iter in tqdm(range(n - 1)):

        tensor_img2 = load_tensor_image(test_files[iter+1], args)
        img_pil = Image.open(test_files[iter]).convert("RGB")
        img_pil = img_pil.resize((args.img_width, args.img_height), Image.Resampling.LANCZOS)
        img = np.array(img_pil).astype(np.float32)
        rgb_for_video = np.array(img_pil).astype(np.uint8)
        # # Save resized image
        # resized_img_path = "resized_image.jpg"
        # Image.fromarray(img.astype(np.uint8)).save(resized_img_path)
        # img = np.transpose(img, (2, 0, 1))
        dan_img = depth_anything_transform(img)
        dan_img = torch.tensor(dan_img, device=device).unsqueeze(0)
        
        if test_fps and iter == 0:
            fps_testing(pose_net, tensor_img1, tensor_img2)
        pose = pose_net(tensor_img1, tensor_img2)
        depth_package = disp_net(dan_img, tensor_img1.size()[-2:])
        depth, uncertainty = depth_package[0]

        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        global_pose = global_pose @  np.linalg.inv(pose_mat)

        poses.append(global_pose[0:3, :].reshape(1, 12))
        
        depth_color = colorize_map(1 / depth, cmap="magma")
        uncertainty_color = colorize_map(uncertainty, cmap="viridis")

        # Stack: [depth | RGB | uncertainty]
        rgb_bgr = cv2.cvtColor(rgb_for_video, cv2.COLOR_RGB2BGR)
        combined_frame = np.hstack((depth_color, rgb_bgr, uncertainty_color))

        # Write frame
        out.write(combined_frame.astype(np.uint8))

        # update
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
    out.release()
    print(f"Saved video to: {video_path}")


if __name__ == '__main__':
    main()
