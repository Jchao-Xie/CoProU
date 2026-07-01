from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp
import math
from utils import unnormalize_and_save, depth_visualization, mask_visualization, error_visualization

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


# photometric loss
# geometry consistency loss
def compute_photo_and_geometry_loss(
    tgt_img,
    ref_imgs,
    intrinsics,
    tgt_depth,
    ref_depths,
    poses,
    poses_inv,
    tgt_intrinsic,
    ref_intrinsics,
    res_mean,
    res_std,
    loss_cfg=None,
):

    photo_loss = 0
    geometry_loss = 0


    for ref_img, ref_depth, pose, pose_inv, ref_intrinsic in zip(ref_imgs, ref_depths, poses, poses_inv, ref_intrinsics):

        b, _, h, w = tgt_img.size()
        tgt_img_scaled = (tgt_img - res_mean) / res_std
        ref_img_scaled = (ref_img - res_mean) / res_std
        intrinsic_scaled = intrinsics
        tgt_depth_scaled = tgt_depth
        ref_depth_scaled = ref_depth

        photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, 
                                                            pose, intrinsic_scaled, tgt_intrinsic, ref_intrinsic, loss_cfg)
        photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, 
                                                            pose_inv,intrinsic_scaled, ref_intrinsic, tgt_intrinsic, loss_cfg)

        photo_loss += (photo_loss1 + photo_loss2)
        geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss, geometry_loss

# multi-frames
# photometric loss
# geometry consistency loss
def compute_photo_and_geometry_loss_multi_frames(
    tgt_imgs,
    ref_imgs,
    intrinsics,
    tgt_depths,
    ref_depths,
    poses,
    poses_inv,
    tgt_intrinsics,
    ref_intrinsics,
    res_mean,
    res_std,
    loss_cfg=None,
):

    photo_loss = 0
    geometry_loss = 0
    
    B, S, _, H, W = tgt_imgs.shape


    tgt_img_scaled = (tgt_imgs - res_mean) / res_std
    ref_img_scaled = (ref_imgs - res_mean) / res_std
    
    tgt_img_scaled = tgt_img_scaled.view(B * S, -1, H, W)
    ref_img_scaled = ref_img_scaled.view(B * S, -1, H, W)
    
    tgt_intrinsics = tgt_intrinsics.view(B * S, 3, 3) if tgt_intrinsics is not None else None
    ref_intrinsics = ref_intrinsics.view(B * S, 3, 3) if ref_intrinsics is not None else None
    
    intrinsic_scaled = intrinsics.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3)  # broadcast intrinsic
    
    reshape_or_none = lambda x: x.view(B * S, -1, H, W) if x is not None else None
    tgt_depth_scaled = [reshape_or_none(x) for x in tgt_depths]
    ref_depth_scaled = [reshape_or_none(x) for x in ref_depths]
    
    poses = process_poses(poses, B, S, H, W)
    poses_inv = process_poses(poses_inv, B, S, H, W)
    
    # with torch.amp.autocast(device_type='cuda', enabled=False):
    

    diff_img1, diff_depth1, valid_mask1, pose_tgt2ref, tgt_depth1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, 
                                                        poses, intrinsic_scaled, tgt_intrinsics, ref_intrinsics, loss_cfg)
    diff_img2, diff_depth2, valid_mask2, pose_ref2tgt, tgt_depth2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, 
                                                        poses_inv,intrinsic_scaled, ref_intrinsics, tgt_intrinsics, loss_cfg)

    
    if loss_cfg.get("per_pixel_min", False):
    
        diff_img1, diff_depth1, valid_mask1 = diff_img1.view(B, S, -1, H, W), diff_depth1.view(B, S, -1, H, W), valid_mask1.view(B, S, -1, H, W)
        diff_img2, diff_depth2, valid_mask2 = diff_img2.view(B, S, -1, H, W), diff_depth2.view(B, S, -1, H, W), valid_mask2.view(B, S, -1, H, W)
        tgt_depth1, tgt_depth2 = tgt_depth1.view(B, S, -1, H, W), tgt_depth2.view(B, S, -1, H, W)

        # per-pixel-minimum
        diff_img_tgt2ref1, diff_depth_tgt2ref1, valid_mask_tgt2ref1 = diff_img1[:, 1:], diff_depth1[:, 1:], valid_mask1[:, 1:]
        diff_img_tgt2ref2, diff_depth_tgt2ref2, valid_mask_tgt2ref2 = diff_img2[:, :-1], diff_depth2[:, :-1], valid_mask2[:, :-1]
        
        # # visualize tgt imgs and ref imgs to verify
        # tgt_imgs1, ref_imgs1 = tgt_img_scaled.view(B, S, -1, H, W), ref_img_scaled.view(B, S, -1, H, W)
        # img_pair1 = torch.stack([tgt_imgs1,   ref_imgs1],   dim=2)
        # tgt_imgs2, ref_imgs2 = ref_img_scaled.view(B, S, -1, H, W), tgt_img_scaled.view(B, S, -1, H, W)
        # img_pair2 = torch.stack([tgt_imgs2,   ref_imgs2],   dim=2)
        # img_tgt2ref1, img_tgt2ref2= img_pair1[:, 1:], img_pair2[:, :-1]
        # unnormalize_and_save(img_tgt2ref1[0][0][0], name='tgt_img1.png')
        # unnormalize_and_save(img_tgt2ref1[0][0][1], name='ref_img1.png')
        # unnormalize_and_save(img_tgt2ref2[0][0][0], name='tgt_img2.png')
        # unnormalize_and_save(img_tgt2ref2[0][0][1], name='ref_img2.png')
        # torch.isclose(img_tgt2ref2[0][0][0], img_tgt2ref2[0][0][0])
        
        diff_img_pair   = torch.stack([diff_img_tgt2ref1,   diff_img_tgt2ref2],   dim=2)
        diff_depth_pair = torch.stack([diff_depth_tgt2ref1, diff_depth_tgt2ref2], dim=2)
        valid_mask_pair = torch.stack([valid_mask_tgt2ref1, valid_mask_tgt2ref2], dim=2)
        
        diff_img_mean = diff_img_pair.mean(dim=3, keepdim=True)
        
        indices = torch.argmin(diff_img_mean, dim=2, keepdim=True)
        
        diff_img   = torch.gather(diff_img_mean,   2, indices).squeeze(2).clamp(min=loss_cfg.photometric_clamp_min)
        diff_depth = torch.gather(diff_depth_pair, 2, indices).squeeze(2)
        valid_mask = torch.gather(valid_mask_pair, 2, indices).squeeze(2)
        tgt_depth  = tgt_depth1[:, 1:]
        
        # weighting the photometric loss with disp weighting
        if loss_cfg.get("disp_weighting", False):
            disp = 0.1 / tgt_depth - 0.001 # in range (0,1)
            disp_weighting = 0.5 + 0.5 * disp.detach()
            diff_img *= disp_weighting
            
        # multiply 2 to get similar gradient magnitude as without per_pixel_min_loss
        photo_loss = 2 * mean_on_mask(diff_img, valid_mask)
        geometry_loss = 2 * mean_on_mask(diff_depth, valid_mask)
    else:
        disp_weighting1 = 1.0
        disp_weighting2 = 1.0
        if loss_cfg.get("disp_weighting", False):
            disp1 = 0.1 / tgt_depth1 - 0.001 # in range (0,1)
            disp_weighting1 = 0.5 + 0.5 * disp1.detach()
            
            disp2 = 0.1 / tgt_depth2 - 0.001 # in range (0,1)
            disp_weighting2 = 0.5 + 0.5 * disp2.detach()
            
        photo_loss = (mean_on_mask(disp_weighting1 * diff_img1.mean(1, True).clamp(min=loss_cfg.photometric_clamp_min), valid_mask1)
                    + mean_on_mask(disp_weighting2 * diff_img2.mean(1, True).clamp(min=loss_cfg.photometric_clamp_min), valid_mask2))
        geometry_loss = (mean_on_mask(diff_depth1, valid_mask1)
                      + mean_on_mask(diff_depth2, valid_mask2))
    smooth_loss = compute_smooth_loss_multi_frames(tgt_depth_scaled, tgt_img_scaled, ref_depth_scaled, ref_img_scaled)
    
    poses_consistancy_loss = inv_consistency_loss(pose_tgt2ref, pose_ref2tgt) # when min_loss the first and last relative pose are not used in bidirection

    return photo_loss, smooth_loss, geometry_loss, poses_consistancy_loss

def compute_smooth_loss_multi_frames(tgt_depths, tgt_imgs, ref_depths, ref_imgs,):

    loss = get_smooth_loss(tgt_depths[0], tgt_imgs)

    loss += get_smooth_loss(ref_depths[0], ref_imgs)

    return loss

def process_poses(poses, B, S, H, W):
    pose_mat, (tgt_uncertainty, ref_uncertainty) = poses

    pose_mat = pose_mat.view(B * S, *pose_mat.shape[2:])
    tgt_uncertainty = tgt_uncertainty.view(B * S, -1, H, W)
    ref_uncertainty = ref_uncertainty.view(B * S, -1, H, W)

    return pose_mat, (tgt_uncertainty, ref_uncertainty)

def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, tgt_intrinsic, ref_intrinsic, loss_cfg):
    
    pose, unty = pose
    tgt_unty, ref_unty = unty
    
    tgt_depth, tgt_depth_conf = tgt_depth
    ref_depth, ref_depth_conf = ref_depth

    ref_img_warped, valid_mask, projected_depth, computed_depth, ref_unty_warped = inverse_warp(ref_img, tgt_depth, ref_depth, ref_unty, pose, intrinsic, tgt_intrinsic, ref_intrinsic, loss_cfg)

    ref_unty_warped = ref_unty_warped.clamp(0, 1)
    
    diff_img = (tgt_img - ref_img_warped).abs()

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    
    combined_unty = torch.sqrt(tgt_unty**2 + ref_unty_warped**2) if loss_cfg.uncertainty.CoProU else tgt_unty

    if loss_cfg.with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)) * valid_mask
        valid_mask = auto_mask

    if loss_cfg.with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    
    if loss_cfg.uncertainty.type is not None:
        diff_img = diff_img / combined_unty + torch.log(combined_unty)
    
        if loss_cfg.geo.linear_down_sampling:
            diff_depth = diff_depth * (1 - combined_unty).detach() if loss_cfg.geo.detach else diff_depth * (1 - combined_unty)
        else:
            diff_depth = diff_depth / combined_unty.detach() if loss_cfg.geo.detach else diff_depth / combined_unty

    if loss_cfg.with_mask == True:
        diff_mask = ((computed_depth - projected_depth) / (computed_depth + projected_depth)).abs().clamp(0, 1)
        weight_mask = (1 - diff_mask).detach()
        diff_img = diff_img * weight_mask
        
    # compute all loss
    # reconstruction_loss = mean_on_mask(diff_img.clamp(min=loss_cfg.photometric_clamp_min), valid_mask)  ## clamping

    # geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)
    # unnormalize_and_save(ref_img_warped[0], name='ref_img_warped.png')
    # unnormalize_and_save(tgt_img[0], name='tgt_img.png')
    # unnormalize_and_save(ref_img[0], name='ref_img.png')
    # depth_visualization((computed_depth)[0], (projected_depth)[0], (diff_depth)[0], 'depth_error')
    # depth_visualization((1 / computed_depth)[0], (1 / projected_depth)[0], (diff_depth)[0], 'disp')
    # depth_visualization((tgt_unty)[0], (ref_unty_warped)[0], (combined_unty)[0], 'uncerntainty')
    # depth_visualization((tgt_unty)[0], (ref_unty)[0], (combined_unty)[0], 'uncerntainty_origin')
    # depth_visualization((tgt_depth)[0], (ref_depth)[0], (diff_depth)[0], 'depth_origin')
    # depth_visualization(tgt_img[0] * valid_mask[0], ref_img_warped[0] * valid_mask[0], diff_img[0] * valid_mask[0], 'image_error')
    # depth_visualization((computed_depth)[0], (ref_depth)[0], (diff_depth)[0], 'depth')
    # mask_visualization(valid_mask[0], 'valid_mask')
    # error_visualization(diff_img[0], "image error only")


    return diff_img, diff_depth, valid_mask, pose, tgt_depth


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff).float()
    precentage = mask.mean().item()
    if precentage > 0.05:
        mean_value = (diff * mask).sum() / mask.sum()
        if precentage <= 0.25:
            print("alert! below 25%")
    else:
        mean_value = diff.sum() * 0
        print(f"bad new :( precentage: {precentage}")
        # raise RuntimeError("Valid mask coverage below 5% — aborting computation.")
    return mean_value

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs, res_mean, res_std):

    loss = get_smooth_loss(tgt_depth[0], (tgt_img - res_mean) / res_std)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth[0], (ref_img - res_mean) / res_std)

    return loss


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

def inv_consistency_loss(pose_tgt2ref, pose_ref2tgt, lambda_R=1.0, lambda_t=1.0, eps=1e-6):
    # pose_*: (..., 4, 4)
    R_tr = pose_tgt2ref[..., :3, :3]
    t_tr = pose_tgt2ref[..., :3, 3]
    R_rt = pose_ref2tgt[..., :3, :3]
    t_rt = pose_ref2tgt[..., :3, 3]

    # product should be identity
    R_diff = R_tr @ R_rt
    t_diff = t_tr + (R_tr @ t_rt.unsqueeze(-1)).squeeze(-1)

    # rotation geodesic via trace
    cos = (R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    loss_rot = torch.arccos(torch.clamp(cos, -1.0 + eps, 1.0 - eps))

    # translation cycle residual (no normalization, no huber)
    loss_trans = t_diff.norm(dim=-1)

    return lambda_R * loss_rot.mean() + lambda_t * loss_trans.mean()

