## reference from https://github.com/YihongSun/Dynamo-Depth.git

import torch
import torch.nn as nn
import numpy as np


class DepthMetrics(nn.Module):
    """ Compute depth performance
    """
    def __init__(self, img_bound, min_depth, max_depth):
        super(DepthMetrics, self).__init__()
        self.depth_metric_names = ['de:abs_rel', 'de:sq_rel', 'de:rms', 'de:log_rms', 'da:a1', 'da:a2', 'da:a3']
        self.img_bound = img_bound
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def forward(self, inputs, outputs, mask=None):
        disp_pred = outputs[('disp_scaled', 0, 0)]  # (B, 1, H, W)
        depth_gt = inputs['depth_gt']               # (B, dataset.max_depth_samp, 3)    # include padding
        depth_valid = inputs['depth_valid']         # (B, dataset.max_depth_samp,)      # specify original
        gt_dim = inputs['gt_dim']                   # (B, 2,) - ground truth image dim
        uncertainty = outputs['uncertainty']

        metrics = {metric : 0 for metric in self.depth_metric_names}
        if mask is not None:
            mask_labels = [l.item() for l in torch.unique(mask)] # split based on mask values
            metrics.update({f'{metric}_mask' : {l : [0,0] for l in mask_labels} for metric in self.depth_metric_names})

        for bi, (disp_p, depth_g, valid, dim, uncertainty_map) in enumerate(zip(disp_pred, depth_gt, depth_valid, gt_dim, uncertainty)):
            
            uncertainty_thres = 0.3

            gt_height, gt_width = dim[0].item(), dim[1].item()
            up, down = int(self.img_bound[0] * gt_height), int(self.img_bound[1] * gt_height)
            left, right = int(self.img_bound[2] * gt_width), int(self.img_bound[3] * gt_width)

            valid = torch_and(valid,
                              depth_g[:,0] >= up,                     # check for image bounds
                              depth_g[:,0] < down,
                              valid, depth_g[:,1] >= left,
                              valid, depth_g[:,1] < right,
                              depth_g[:,2] > self.min_depth, # check for depth bounds
                              depth_g[:,2] < self.max_depth,
                              )
            
            valid_ind = depth_g[:,0][valid].long(), depth_g[:,1][valid].long()
            depth_p = 1 / nn.functional.interpolate(disp_p[None], (gt_height, gt_width), mode='bilinear', align_corners=False).squeeze()
            uncertainty_p = nn.functional.interpolate(uncertainty_map[None], (gt_height, gt_width), mode='bilinear', align_corners=False).squeeze()

            # valid_map = torch.zeros_like(uncertainty_p, device=uncertainty_p.device)
            # valid_map[valid_ind[0], valid_ind[1]] = 1
            # valid_map = uncertainty_p <=0.3 * valid_map

            # _ =  uncertainty_p[depth_g[:,0][valid].long(), depth_g[:,1][valid].long()] <= 0.3
            d_gt = depth_g[:,2][valid]
            d_pd = depth_p[valid_ind]
            u_p = (uncertainty_p <= uncertainty_thres)[valid_ind]
            d_gt, d_pd = d_gt[u_p], d_pd[u_p]

            # median scaling and clamp
            scale = torch.median(d_gt) / torch.median(d_pd)
            d_pd *= scale
            d_pd = torch.clamp(d_pd, self.min_depth, self.max_depth)

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt, torch.nn.functional as F, torch

# save_path = f"debug_depth_{bi}.png"

# # reconstruct scaled dense depth map
# scaled_depth_p = torch.clamp(depth_p * scale, self.min_depth, self.max_depth)
# depth_gv = depth_g[valid]

# # === sample uncertainty for the same GT pixels ===
# y = depth_gv[:, 0].long().clamp(0, scaled_depth_p.shape[0] - 1)
# x = depth_gv[:, 1].long().clamp(0, scaled_depth_p.shape[1] - 1)
# u_vals = uncertainty_p[y, x]

# # === apply uncertainty threshold ===
# 
# mask_u = u_vals <= uncertainty_thres
# depth_gv = depth_gv[mask_u]          # keep only low-uncertainty points
# gt_vals = depth_gv[:, 2]

# # compute per-point prediction and errors
# pred_vals = scaled_depth_p[y[mask_u], x[mask_u]]
# abs_err = torch.abs(pred_vals - gt_vals)
# rmse = torch.sqrt(torch.mean(abs_err ** 2)).item()
# max_err = torch.quantile(abs_err, 0.98).item() if len(abs_err) > 0 else 1.0  # clip for color scaling

# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # ---- Left: predicted (scaled) + sparse GT ----
# im_pred = axs[0].imshow(scaled_depth_p.detach().cpu(), cmap='plasma',
#                         vmin=self.min_depth, vmax=self.max_depth)
# sc_gt = axs[0].scatter(depth_gv[:, 1].cpu(), depth_gv[:, 0].cpu(),
#                     c=gt_vals.cpu(), s=3, cmap='viridis',
#                     vmin=self.min_depth, vmax=self.max_depth, alpha=0.9)
# axs[0].set_title(f"Predicted (scaled) + GT (uncertainty ≤ {uncertainty_thres}) | scale={scale.item():.3f}")
# axs[0].axis("off")
# plt.colorbar(im_pred, ax=axs[0], fraction=0.046, pad=0.04, label="Predicted Depth (m)")
# plt.colorbar(sc_gt, ax=axs[0], fraction=0.046, pad=0.08, label="GT Depth (m)")

# # ---- Right: sparse RMSE scatter ----
# axs[1].imshow(scaled_depth_p.detach().cpu(), cmap='gray', alpha=0.2)
# sc_err = axs[1].scatter(depth_gv[:, 1].cpu(), depth_gv[:, 0].cpu(),
#                         c=abs_err.cpu(), s=3, cmap='inferno',
#                         vmin=0, vmax=max_err, alpha=0.9)
# axs[1].set_title(f"RMSE Scatter (uncertainty ≤ {uncertainty_thres}) | mean RMSE={rmse:.3f} m")
# axs[1].axis("off")
# plt.colorbar(sc_err, ax=axs[1], fraction=0.046, pad=0.04, label="Abs Error (m)")

# plt.tight_layout()
# plt.savefig(save_path, dpi=200)
# plt.close(fig)

# print(f"[Depth Debug] Saved {save_path}")
# print(f"[Depth Debug] scale={scale.item():.3f}, mean RMSE={rmse:.3f}, GT median={torch.median(gt_vals):.3f}, Pred median={torch.median(pred_vals):.3f}")

            depth_errors = compute_errors(d_gt, d_pd)
            for i, metric in enumerate(self.depth_metric_names):
                metrics[metric] += depth_errors[i]

            if mask is not None:
                m_valid = mask[bi][valid_ind]
                for l in mask_labels:
                    m = m_valid == l

                    dgm, dpm = d_gt[m], d_pd[m]
                    cnt = dgm.shape[0]
                    if cnt == 0:
                        continue
                    depth_errors = compute_errors(dgm, dpm)
                    
                    for i, metric in enumerate(self.depth_metric_names):
                        metrics[f'{metric}_mask'][l][0] += depth_errors[i].item() * cnt  
                        metrics[f'{metric}_mask'][l][1] += cnt  
        
        for metric in self.depth_metric_names:
            metrics[metric] = metrics[metric] / disp_pred.size(0)
            
        return metrics
    
def torch_and(*args):
    """ Accept a list of arugments of torch.Tensor of the same shape, compute element-wise and operation for all of them
        Output tensor has the same shape as the input tensors
    """
    out = args[0]
    for a in args:
        assert out.size() == a.size(), "Sizes must match: [{}]".format(', '.join([str(x.size()) for x in args]))
        out = torch.logical_and(out, a)
    return out

def compute_errors(gt, pred):
    """ Computation of error metrics between predicted and ground truth depths
        https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/evaluate_depth.py#L27
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def write_to_file(data_list, fname, bool_newline=True):
    """ Write the given list of strings into the file
    """
    with open(fname, 'w') as fh:
        if bool_newline:
            fh.writelines([d+'\n' for d in data_list])
        else:
            fh.writelines(data_list)

import torch
import numpy as np
import trimesh
import gradio as gr
from pathlib import Path

def visualize_pointcloud_with_gradio(depth, K, inv_K, camTcam=None, image=None, save_path="scene.glb"):
    """
    Args:
        depth: (B, 1, H, W) torch.Tensor
        K: (B, 4, 4) torch.Tensor
        inv_K: (B, 4, 4) torch.Tensor
        camTcam: (B, 4, 4) torch.Tensor, camera-to-world transformation
        image: (B, 3, H, W) torch.Tensor or None, used for coloring points
        save_path: str, path to save .glb scene

    Returns:
        None (launches gradio viewer)
    """
    B, _, H, W = depth.shape
    assert B == 1, "Only batch size 1 supported for visualization"

    if camTcam is None:
        camTcam = torch.eye(4).to(depth.device) 
    # Step 1: create pixel grid
    meshgrid = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)  # (2, H, W)
    ones = np.ones((1, H, W), dtype=np.float32)
    pix_coords = np.concatenate([id_coords, ones], axis=0)  # (3, H, W)
    pix_coords = torch.from_numpy(pix_coords).unsqueeze(0).to(depth.device)  # (1, 3, H, W)

    # Step 2: backproject to camera frame
    cam_points = torch.matmul(inv_K[:, :3, :3], 
                              pix_coords.view(B, 3, -1)) * depth.view(B, 1, -1)
    cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=1)  # (B, 4, H*W)

    # Step 3: transform to world frame
    world_points = torch.matmul(camTcam, cam_points_h)[:, :3, :]  # (B, 3, N)
    world_points = world_points.squeeze(0).permute(1, 0).cpu().numpy()  # (N, 3)
    
    # Step 3.5: fix axis orientation for visualization
    world_points[:, 1] *= -1  # Flip Y (down→up)
    world_points[:, 2] *= -1  # Flip Z (forward→backward)

    # Step 4: get colors
    if image is not None:
        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img_np = np.clip(img_np, 0, 1)
        colors = img_np.reshape(-1, 3)
    else:
        colors = np.ones_like(world_points) * 0.7  # gray

    # Step 5: create trimesh point cloud
    cloud = trimesh.points.PointCloud(world_points, colors=colors)
    scene = trimesh.Scene([cloud])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(save_path)

    # Step 6: visualize with Gradio
    with gr.Blocks() as demo:
        gr.Markdown("### 🌍 3D Point Cloud Viewer")
        gr.Model3D(value=save_path, height=600, clear_color=[0.13, 0.13, 0.14, 1])
    demo.launch(share=True)