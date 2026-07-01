import lightning as L
from hydra.utils import instantiate
from models.vggt.models.vggt import VGGT
from coprou_training.train_util import load_pretrained, safe_inverse
from vggt_training.train_utils.freeze import freeze_modules

from models.vggt.heads.dpt_head import DPTHead
from models.vggt.heads.camera_head import CameraHead
from models.PoseHead import PoseHead

from vggt_training.train_utils.optimizer import construct_optimizers
from vggt_training.train_utils.general import AverageMeter

from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_photo_and_geometry_loss_multi_frames
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from depth_anything_v2.dpt import fine_tuning_DepthAnythingV2
import models
from omegaconf import OmegaConf

from utils import tensor2array
import time
import torch
import torch.nn.functional as F
import math

from inverse_warp import pose_vec2mat

class coprou(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)  # saves all args into self.hparams
        self.cfg = cfg
        
        if not OmegaConf.select(self.cfg, "model.cnn.enabled", default=False):
            ## model configs for different size, follow depthanythingv2
            model_configs = {
                'vits': {'patch_embed': 'dinov2_vits14_reg', 'features': 64, 'out_channels': [48, 96, 192, 384], 'intermediate_layer_idx': [2, 5, 8, 11]},
                'vitb': {'patch_embed': 'dinov2_vitbl14_reg', 'features': 128, 'out_channels': [96, 192, 384, 768], 'intermediate_layer_idx':[2, 5, 8, 11]},
                'vitl': {'patch_embed': 'dinov2_vitl14_reg', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'intermediate_layer_idx':[4, 11, 17, 23]},
            }

            self.model = VGGT(model_type= model_configs[self.cfg.model.vggt.model_type]['patch_embed'],
                            features=model_configs[self.cfg.model.vggt.model_type]['features'],
                            out_channels=model_configs[self.cfg.model.vggt.model_type]['out_channels'],
                            intermediate_layer_idx=model_configs[self.cfg.model.vggt.model_type]['intermediate_layer_idx'],
                            enable_camera=cfg.model.vggt.enable_camera, 
                            enable_point=cfg.model.vggt.enable_point, 
                            enable_depth=cfg.model.vggt.enable_depth, 
                            enable_track=cfg.model.vggt.enable_track)
            
            if cfg.optim.load_vggt_pretrained_modules: # and self.cfg.model.vggt.model_type == "vitl":
                _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                state_dict = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
                load_pretrained(model=self.model, state_dict=state_dict, pretrained_modules=cfg.optim.load_vggt_pretrained_modules)
            else:
                # if we didn't load VGGT, we train VGGT from DINOv2
                _URL = {
                    'vits': "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
                    'vitb': "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
                    'vitl': "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
                }
                state_dict = torch.hub.load_state_dict_from_url(_URL[self.cfg.model.vggt.model_type], map_location="cpu")
                missing, unexpected = self.model.aggregator.patch_embed.load_state_dict(
                    state_dict, strict=False
                )
                print(f"Loaded patch_embed from {model_configs[self.cfg.model.vggt.model_type]['patch_embed']} distilled (missing={missing}, unexpected={unexpected})")
            
            if cfg.optim.frozen_module_names:
                self.model = freeze_modules(
                    self.model,
                    patterns=cfg.optim.frozen_module_names,
                )
            
            ### if we don't use vggt depth head, we use ours instead.
            if not cfg.model.vggt.enable_depth:
                self.depth_head = DPTHead(dim_in=2 * self.model.aggregator.patch_embed.embed_dim, 
                                        features=model_configs[self.cfg.model.vggt.model_type]['features'], 
                                        out_channels=model_configs[self.cfg.model.vggt.model_type]['out_channels'], 
                                        intermediate_layer_idx=model_configs[self.cfg.model.vggt.model_type]['intermediate_layer_idx'],
                                        output_dim=1, activation=cfg.model.depth_head.activation_type, without_conf=True)
            
            if not cfg.model.vggt.enable_camera:
                if cfg.model.camera_head.type == 'reloc3r':
                    self.camera_head = PoseHead(dim_in=4 * self.model.aggregator.patch_embed.embed_dim, 
                                            pose_encoding_type=self.cfg.model.camera_head.pose_encoding_type, 
                                            num_heads=self.model.aggregator.num_heads, depth=self.model.aggregator.depth//6,
                                            rope=self.model.aggregator.rope)
                else:
                    self.camera_head = CameraHead(dim_in=2 * self.model.aggregator.patch_embed.embed_dim, 
                                            pose_encoding_type=self.cfg.model.camera_head.pose_encoding_type, 
                                            num_heads=self.model.aggregator.num_heads, trunk_depth=self.model.aggregator.depth//6)
                
            if cfg.loss.uncertainty.type is None:
                # dummy useless head
                self.uncertainty_head = torch.nn.Linear(1, 1, bias=True)
            elif cfg.loss.uncertainty.type == 'pair-wise':
                ### pair-wise uncertainty head
                self.uncertainty_head = DPTHead(dim_in=4 * self.model.aggregator.patch_embed.embed_dim, 
                                                features=model_configs[self.cfg.model.vggt.model_type]['features'], 
                                                out_channels=model_configs[self.cfg.model.vggt.model_type]['out_channels'], 
                                                intermediate_layer_idx=model_configs[self.cfg.model.vggt.model_type]['intermediate_layer_idx'],
                                                output_dim=2, activation="sigmoid", conf_activation="sigmoid")
            elif cfg.loss.uncertainty.type == 'non-pair-wise':
                ### in this case, the uncertainty head is like depth head, but predict uncertainty
                self.uncertainty_head = DPTHead(dim_in=2 * self.model.aggregator.patch_embed.embed_dim, 
                                        features=model_configs[self.cfg.model.vggt.model_type]['features'], 
                                        out_channels=model_configs[self.cfg.model.vggt.model_type]['out_channels'], 
                                        intermediate_layer_idx=model_configs[self.cfg.model.vggt.model_type]['intermediate_layer_idx'],
                                        output_dim=1, activation="sigmoid", without_conf=True)
            else:
                raise ValueError(f"Unknown uncertainty_head.type: {cfg.model.uncertainty_head.type}")
        
        else:
            self.init_cnn()
        
        for name, value in (("_resnet_mean", [0.485, 0.456, 0.406]), ("_resnet_std", [0.229, 0.224, 0.225])):
            self.register_buffer(name, torch.tensor(value).view(1, 3, 1, 1), persistent=False)
            
        # Timing
        self._batch_start_time = None
        self._data_end_time = time.time()

        # Meters
        self.batch_time_meter = AverageMeter("batch_time")
        self.data_time_meter = AverageMeter("data_time")
        

    def forward(self, tgt_img, ref_imgs, input_size, res_tgt_img=None, res_ref_imgs=None):
            return self.model(tgt_img, ref_imgs, input_size, res_tgt_img, res_ref_imgs)
    
    def init_cnn(self):
        model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
        self.pose_net = models.PoseResNet(18, self.cfg.model.cnn.pose_with_pretrain)
        self.disp_net = fine_tuning_DepthAnythingV2(**model_configs[self.cfg.model.cnn.encoder_type])
        
        if self.cfg.model.cnn.depth_encoder == 'dan':
            weights = torch.load(f'checkpoints/depth_anything_v2_{self.cfg.model.cnn.encoder_type}.pth', map_location='cpu')
            weights = {k: v for k, v in weights.items() if 'pretrained' in k}
            self.disp_net.load_state_dict(weights, strict=False)
            for param in self.disp_net.pretrained.parameters():
                param.requires_grad = False

        elif self.cfg.model.cnn.depth_encoder == 'dinov2':
            weights = torch.load(f"checkpoints/dinov2_{self.cfg.model.cnn.encoder_type}14_pretrain.pth")
            self.disp_net.pretrained.load_state_dict(weights)
            for param in self.disp_net.pretrained.parameters():
                param.requires_grad = False
        else:
            print("[Warning] No pretrained backbone loaded for depth net")

        if self.cfg.model.cnn.pretrained_disp:
            print("=> using pre-trained weights for DispResNet")
            weights = torch.load(self.cfg.model.cnn.pretrained_disp)
            self.disp_net.load_state_dict(weights['state_dict'], strict=False)
            
        if self.cfg.model.cnn.pretrained_pose:
            print("=> using pre-trained weights for PoseResNet")
            weights = torch.load(self.cfg.model.cnn.pretrained_pose)
            self.pose_net.load_state_dict(weights['state_dict'], strict=False)
            
        if self.cfg.model.cnn.unfrozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        self._wrappers = construct_optimizers(self, self.cfg.optim)
        return [w.optimizer for w in self._wrappers]
    
    def on_train_start(self):
        acc = int(getattr(self.trainer, "accumulate_grad_batches", 1))
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            # Already in optimizer updates; do NOT divide by acc
            self._total_updates = int(self.trainer.max_steps)
        else:
            steps_per_epoch = int(self.trainer.num_training_batches)  # respects limit_train_batches
            updates_per_epoch = int(math.ceil(steps_per_epoch / max(1, acc)))
            self._total_updates = updates_per_epoch * max(1, int(self.trainer.max_epochs))
        
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure,
    ) :
        where = min((self.global_step + 1) / float(self._total_updates), 1.0 - 1e-8)
        wrapper = next(w for w in self._wrappers if w.optimizer is optimizer.optimizer)
        wrapper.step_schedulers(where)  # update param_group["lr"] etc.

        # Perform optimizer step
        optimizer.step(closure=optimizer_closure)
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
    
    def on_fit_start(self):
        # Instantiate from YAML and build param groups once
        if self.cfg.optim.get("gradient_clip"):
            self.grad_clipper = instantiate(self.cfg.optim.gradient_clip)
            self.grad_clipper.setup_clipping(self)  # self is an nn.Module
    
    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """
        Called after backward and right before optimizer.step().
        We do per-group clipping here; do NOT also set Trainer.gradient_clip_val.
        """
        if self.grad_clipper is None:
            return

        # Ensure grads are unscaled if AMP is on (Lightning usually does this only
        # when you use Trainer.gradient_clip_val; we handle it ourselves here).
        scaler = getattr(self.trainer, "scaler", None)
        if scaler is None:
            scaler = getattr(getattr(self.trainer.strategy, "precision_plugin", None), "scaler", None)
        if scaler is not None:
            try:
                scaler.unscale_(optimizer)  # safe no-op if already unscaled
            except Exception:
                pass

        norms = self.grad_clipper(self)  # clips in-place per config; returns norms
    
    def training_step(self, batch, batch_idx):
        self.time_end = time.time()
        images, intrinsics = batch["images"], batch["intrinsics"]
        w1, w2, w3, w4 = (self.hparams.loss.photo_loss_weight, 
                          self.hparams.loss.smooth_loss_weight, 
                          self.hparams.loss.geometry_consistency_weight,
                          self.hparams.loss.poses_consistency_weight
                        )
        if not OmegaConf.select(self.cfg, "model.cnn.enabled", default=False):
            if self.cfg.model.camera_head.type is None:
                (
                    poses, poses_inv, tgt_depths, ref_depths,
                    tgt_intrinsics, ref_intrinsics,
                    tgt_idx, ref_idx,
                    pair_images_tgt, pair_images_ref
                ) = self.get_prediction_multi_frames(images)
            ## To Do!!!!!!!!!!!!!!!!!!!!!!## To Do!!!!!!!!!!!!!!!!!!!!!!## To Do!!!!!!!!!!!!!!!!!!!!!!
            else:
                (
                    poses, poses_inv, tgt_depths, ref_depths,
                    tgt_intrinsics, ref_intrinsics,
                    tgt_idx, ref_idx,
                    pair_images_tgt, pair_images_ref
                ) = self.get_pairs_prediction(images)
        else: 
            (
                poses, poses_inv, tgt_depths, ref_depths,
                tgt_intrinsics, ref_intrinsics,
                tgt_idx, ref_idx,
                pair_images_tgt, pair_images_ref
            ) = self.get_cnn_prediction(images)

        # for warping and inverse using f32
        with torch.autocast(device_type="cuda", enabled=False):
            photometry_loss, smooth_loss, geometry_loss, poses_consistancy_loss = compute_photo_and_geometry_loss_multi_frames(   
                                                        pair_images_tgt,
                                                        pair_images_ref,
                                                        intrinsics,
                                                        tgt_depths,
                                                        ref_depths,
                                                        poses,
                                                        poses_inv,
                                                        tgt_intrinsics,
                                                        ref_intrinsics,
                                                        res_mean=self._resnet_mean,
                                                        res_std=self._resnet_std,
                                                        loss_cfg=self.hparams.loss,
                                                        )
        
        total_loss = w1 * photometry_loss + w2 * smooth_loss + w3 * geometry_loss + w4 * poses_consistancy_loss

        if self.global_step % 10 == 0:
            self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train/photo_loss", photometry_loss, on_step=True, on_epoch=False)
            self.log("train/smooth_loss", smooth_loss, on_step=True, on_epoch=False)
            self.log("train/geometry_loss", geometry_loss, on_step=True, on_epoch=False)
            self.log("train/poses_consistancy_loss", poses_consistancy_loss, on_step=True, on_epoch=False)

            for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
                lr = param_group["lr"]
                self.log(f"lr/group{i}", lr, on_step=True, on_epoch=False, prog_bar=True)

        # Always log per-step for epoch averaging
        self.log("train/total_loss_epoch", total_loss, on_step=False, on_epoch=True)
        self.log("train/photo_loss_epoch", photometry_loss, on_step=False, on_epoch=True)
        self.log("train/smooth_loss_epoch", smooth_loss, on_step=False, on_epoch=True)
        self.log("train/geometry_loss_epoch", geometry_loss, on_step=False, on_epoch=True)
        
        if batch_idx in self.trainer.datamodule.random_train_indices:
            tb = self.logger.experiment
            epoch = self.current_epoch

            tb.add_image(f'train Input/batch{batch_idx}', tensor2array(pair_images_tgt[0][0]), epoch)

            tb.add_image(f'train Dispnet Output Normalized/batch{batch_idx}',
                        tensor2array(1.0 / tgt_depths[0][0][0], max_value=None, colormap='magma'),
                        epoch)

            tb.add_image(f'train Depth Output/batch{batch_idx}',
                        tensor2array(tgt_depths[0][0][0], max_value=10),
                        epoch)

            tb.add_image(f'train Uncertainty as target/batch{batch_idx}',
                        tensor2array(poses[1][0][0][0], max_value=None),
                        epoch)
            tb.add_image(f'train Uncertainty as reference/batch{batch_idx}',
                        tensor2array(poses[1][1][0][0], max_value=None),
                        epoch)

        return total_loss

    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # Measure data time
        data_time = time.time() - self._data_end_time
        self.data_time_meter.update(data_time)

        if batch_idx % 10 == 0:
            self.log("avg/data_time", self.data_time_meter.average, prog_bar=True, on_step=True)
            self.log("val/data_time", self.data_time_meter.value, prog_bar=False, on_step=True)

        self._batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Measure batch time
        batch_time = time.time() - self._batch_start_time
        self.batch_time_meter.update(batch_time)

        if batch_idx % 10 == 0:
            self.log("avg/batch_time", self.batch_time_meter.average, prog_bar=True, on_step=True)
            self.log("val/batch_time", self.batch_time_meter.value, prog_bar=False, on_step=True)

        self._data_end_time = time.time()

    def on_train_epoch_start(self):
        self.batch_time_meter.reset()
        self.data_time_meter.reset()

    def validation_step(self, batch, batch_idx):
        images, intrinsics = batch["images"], batch["intrinsics"]
        
        if not OmegaConf.select(self.cfg, "model.cnn.enabled", default=False):
            if self.cfg.model.camera_head.type is None:
                (
                    poses, poses_inv, tgt_depths, ref_depths,
                    tgt_intrinsics, ref_intrinsics,
                    tgt_idx, ref_idx,
                    pair_images_tgt, pair_images_ref
                ) = self.get_prediction_multi_frames(images)
            ## To Do!!!!!!!!!!!!!!!!!!!!!!## To Do!!!!!!!!!!!!!!!!!!!!!!## To Do!!!!!!!!!!!!!!!!!!!!!!
            else:
                (
                    poses, poses_inv, tgt_depths, ref_depths,
                    tgt_intrinsics, ref_intrinsics,
                    tgt_idx, ref_idx,
                    pair_images_tgt, pair_images_ref
                ) = self.get_pairs_prediction(images)
        else: 
            (
                poses, poses_inv, tgt_depths, ref_depths,
                tgt_intrinsics, ref_intrinsics,
                tgt_idx, ref_idx,
                pair_images_tgt, pair_images_ref
            ) = self.get_cnn_prediction(images)
            
        with torch.autocast(device_type="cuda", enabled=False):
            photometry_loss, smooth_loss, geometry_loss, poses_consistancy_loss = compute_photo_and_geometry_loss_multi_frames(   ## To Do!!!!!!!!!!!!!!!!!!!!!!
                                                        pair_images_tgt,
                                                        pair_images_ref,
                                                        intrinsics,
                                                        tgt_depths,
                                                        ref_depths,
                                                        poses,
                                                        poses_inv,
                                                        tgt_intrinsics,
                                                        ref_intrinsics,
                                                        res_mean=self._resnet_mean,
                                                        res_std=self._resnet_std,
                                                        loss_cfg=self.hparams.loss,
                                                        )
        total_loss = photometry_loss

        self.log("val photo_loss", photometry_loss.item(), prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val smooth_loss", smooth_loss.item(), prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val geometry_loss", geometry_loss.item(), prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val poses_consistancy_loss", poses_consistancy_loss.item(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val total_loss", total_loss.item(), prog_bar=True, on_epoch=True, sync_dist=True)
        
        if batch_idx in self.trainer.datamodule.random_val_indices:
            tb = self.logger.experiment
            epoch = self.current_epoch

            
            tb.add_image(f'val Input/batch{batch_idx}', tensor2array(pair_images_tgt[0][0]), epoch)

            tb.add_image(f'val Dispnet Output Normalized/batch{batch_idx}',
                        tensor2array(1.0 / tgt_depths[0][0][0], max_value=None, colormap='magma'),
                        epoch)

            tb.add_image(f'val Depth Output/batch{batch_idx}',
                        tensor2array(tgt_depths[0][0][0], max_value=10),
                        epoch)

            tb.add_image(f'val Uncertainty as target/batch{batch_idx}',
                        tensor2array(poses[1][0][0][0], max_value=None),
                        epoch)
            tb.add_image(f'val Uncertainty as reference/batch{batch_idx}',
                        tensor2array(poses[1][1][0][0], max_value=None),
                        epoch)
            
    def get_cnn_prediction(self, images):
        def nearest_multiple(x: int, m: int = 14) -> int:
            down = (x // m) * m
            up = ((x + m - 1) // m) * m
            return up if (x - down) >= (up - x) else down
        
        B, N, C, H, W = images.shape
        H_D = nearest_multiple(H, 14)
        W_D = nearest_multiple(W, 14)
        images_out = images.clone()
        images_ = images.reshape(B * N, C, H, W)
        images_resized = F.interpolate(
            images_,
            size=(H_D, W_D),
            mode="bilinear",      # or "bicubic" if you prefer
            align_corners=False
        )
        images_resized = (images_resized - self._resnet_mean) / self._resnet_std
        depths, uncertainty_map = self.disp_net(images_resized, (H, W))

        depths, uncertainty_map = depths.reshape(B, N, 1, H, W), uncertainty_map.reshape(B, N, 1, H, W)
        
        tgt_idx = [i for i in range(images.size(1)-1)]
        ref_idx = [i+1 for i in range(images.size(1)-1)]
        
        images = (images - self._resnet_mean.unsqueeze(1)) / self._resnet_std.unsqueeze(1)
        pair_images_tgt = images[:, tgt_idx]
        pair_images_ref = images[:, ref_idx]
        
        pair_images_tgt_out = images_out[:, tgt_idx]
        pair_images_ref_out = images_out[:, ref_idx]
        
        tgt_depths = (depths[:, tgt_idx], None)
        ref_depths = (depths[:, ref_idx], None)
        
        pair_images_tgt = pair_images_tgt.reshape(B*(N-1), C, H, W)
        pair_images_ref = pair_images_ref.reshape(B*(N-1), C, H, W)
        
        poses     = self.pose_net(pair_images_tgt, pair_images_ref)
        poses_inv = self.pose_net(pair_images_ref, pair_images_tgt)
        
        poses, poses_inv = pose_vec2mat(poses), pose_vec2mat(poses_inv)
        
        poses     = poses.reshape(B, N-1, 3, 4)
        poses_inv = poses_inv.reshape(B, N-1, 3, 4)
        
        # Bottom row [0, 0, 0, 1]
        bottom_row = torch.tensor([0, 0, 0, 1], dtype=poses.dtype, device=poses.device)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(B, N-1, 1, 4)
        
        # Concatenate to make 4x4
        poses = torch.cat([poses, bottom_row], dim=2)        # [B, N-1, 4, 4]
        poses_inv = torch.cat([poses_inv, bottom_row], dim=2)  # [B, N-1, 4, 4]
        
        tgt_uncertainty = ref_uncertainty_inv = uncertainty_map[:, tgt_idx]
        ref_uncertainty = tgt_uncertainty_inv = uncertainty_map[:, ref_idx]
        
        tgt_intrinsics, ref_intrinsics = None, None
        
        poses = (poses,(tgt_uncertainty, ref_uncertainty))
        
        poses_inv = (poses_inv, (tgt_uncertainty_inv, ref_uncertainty_inv))
        
        return poses, poses_inv, tgt_depths, ref_depths, tgt_intrinsics, ref_intrinsics, tgt_idx, ref_idx, pair_images_tgt_out, pair_images_ref_out
        

    def get_prediction_multi_frames(self, images):
        
        aggregated_tokens_list, ps_idx = self.model.aggregator(images)
        
        with torch.autocast(device_type="cuda", enabled=False):        # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1] if self.cfg.model.vggt.enable_camera else self.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:], 
                                                                pose_encoding_type=self.cfg.model.camera_head.pose_encoding_type)
            
            # Predict Depth Maps
            if self.cfg.model.vggt.enable_depth:
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
            else:
                depth_map = self.depth_head(aggregated_tokens_list, images, ps_idx)
                    
            # get multi-frames uncertainties and poses
            poses, poses_inv, tgt_idx, ref_idx, pair_images_tgt, pair_images_ref= self.get_unty_and_pose_multi_frames(aggregated_tokens_list, ps_idx, extrinsic, images)
            
            tgt_depths = (depth_map[:, tgt_idx].permute(0, 1, 4, 2, 3), depth_conf[:, tgt_idx].unsqueeze(1) if self.cfg.model.vggt.enable_depth else None)
            tgt_intrinsics = intrinsic[:, tgt_idx] if intrinsic is not None else intrinsic
            
            ref_depths = (depth_map[:, ref_idx].permute(0, 1, 4, 2, 3), depth_conf[:, ref_idx].unsqueeze(1) if self.cfg.model.vggt.enable_depth else None)
            ref_intrinsics = intrinsic[:, ref_idx] if intrinsic is not None else intrinsic
            
        return poses, poses_inv, tgt_depths, ref_depths, tgt_intrinsics, ref_intrinsics, tgt_idx, ref_idx, pair_images_tgt, pair_images_ref
    
    def get_unty_and_pose_multi_frames(self, aggregated_tokens_list, ps_idx, extrinsics, images):
        
        B, N, _, _ = extrinsics.shape

        # Create the bottom row [0, 0, 0, 1] (in batch)
        bottom = torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype, device=extrinsics.device)
        bottom = bottom.view(1, 1, 1, 4).expand(B, N, 1, 4)
        
        # Concatenate to form [4, 4]
        extrinsics_homo = torch.cat([extrinsics, bottom], dim=2)    
        
        tgt_idx = [i for i in range(images.size(1)-1)]
        ref_idx = [i+1 for i in range(images.size(1)-1)]
        
        # Predict forward Uncertainty Maps    
        tgt_uncertainty, ref_uncertainty, pair_images_tgt = self.get_pairs_uncertainty(aggregated_tokens_list, 
                                                                images, 
                                                                tgt_idx,
                                                                ref_idx)
        tgt_uncertainty_inv, ref_uncertainty_inv, pair_images_ref = self.get_pairs_uncertainty(aggregated_tokens_list, 
                                                            images, 
                                                            ref_idx,
                                                            tgt_idx)
        
        poses = ((extrinsics_homo[:, ref_idx] @ safe_inverse(extrinsics_homo[:, tgt_idx]), 
                    (tgt_uncertainty.permute(0, 1, 4, 2, 3), ref_uncertainty)
                    ))
        poses_inv = ((extrinsics_homo[:, tgt_idx] @ safe_inverse(extrinsics_homo[:, ref_idx]),
                    (tgt_uncertainty_inv.permute(0, 1, 4, 2, 3), ref_uncertainty_inv)   
                    ))
        
        return poses, poses_inv, tgt_idx, ref_idx, pair_images_tgt, pair_images_ref
        
    
    def get_pairs_prediction(self, images):
        
        aggregated_tokens_list, ps_idx = self.model.aggregator(images)
        
        with torch.autocast(device_type="cuda", enabled=False):        # Predict Cameras
        
            # Predict Depth Maps
            if self.cfg.model.vggt.enable_depth:
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
            else:
                depth_map = self.depth_head(aggregated_tokens_list, images, ps_idx)
            
            
            # get position for pose estimation
            B, S, C_in, H, W = images.shape
            
            # we have S-1 pairs
            pos = None
            if self.model.aggregator.rope is not None:
                pos = self.model.aggregator.position_getter(B * (S-1), 
                                                            H // self.model.aggregator.patch_size, 
                                                            W // self.model.aggregator.patch_size, device=images.device)

            if self.model.aggregator.patch_start_idx > 0:
                # do not use position embedding for special tokens (camera and register tokens)
                # so set pos to 0 for the special tokens
                pos = pos + 1
                pos_special = torch.zeros(B * (S-1), self.model.aggregator.patch_start_idx, 2).to(images.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)
            
                
            
            tgt_idx = [i for i in range(images.size(1)-1)]
            ref_idx = [i+1 for i in range(images.size(1)-1)]
            
            # get depths
            tgt_depths = (depth_map[:, tgt_idx].permute(0, 1, 4, 2, 3), depth_conf[:, tgt_idx].unsqueeze(1) if self.cfg.model.vggt.enable_depth else None)
            tgt_intrinsics = None
            
            ref_depths = (depth_map[:, ref_idx].permute(0, 1, 4, 2, 3), depth_conf[:, ref_idx].unsqueeze(1) if self.cfg.model.vggt.enable_depth else None)
            ref_intrinsics = None # we don't predict intrinsics in this case, using none for adaption to other cases
            
            # predict pose
            last_aggregated_tokens = aggregated_tokens_list[-1]
            poses, poses_inv = self.get_pairs_pose(last_aggregated_tokens, images, tgt_idx, ref_idx, pos) 
            
            if self.cfg.loss.uncertainty.type is not None:
                if self.cfg.loss.uncertainty.type == 'pair-wise':
                    # Predict forward Uncertainty Maps    
                    tgt_uncertainty, ref_uncertainty, pair_images_tgt = self.get_pairs_uncertainty(aggregated_tokens_list, 
                                                                            images, 
                                                                            tgt_idx,
                                                                            ref_idx)
                    tgt_uncertainty_inv, ref_uncertainty_inv, pair_images_ref = self.get_pairs_uncertainty(aggregated_tokens_list, 
                                                                        images, 
                                                                        ref_idx,
                                                                        tgt_idx)
                elif self.cfg.loss.uncertainty.type == 'non-pair-wise':
                    uncertainty_map = self.uncertainty_head(aggregated_tokens_list, images, ps_idx)
                    tgt_uncertainty = ref_uncertainty_inv = uncertainty_map[:, tgt_idx]
                    ref_uncertainty = tgt_uncertainty_inv = uncertainty_map[:, ref_idx]
                    ref_uncertainty, ref_uncertainty_inv = ref_uncertainty.permute(0, 1, 4, 2, 3), ref_uncertainty_inv.permute(0, 1, 4, 2, 3)
                    pair_images_tgt, pair_images_ref = images[:, tgt_idx], images[:, ref_idx]
            else:
                tgt_uncertainty = ref_uncertainty = ref_uncertainty_inv = tgt_uncertainty_inv = torch.ones_like(depth_map[:, tgt_idx],
                                                                                                                device=depth_map.device)
                ref_uncertainty, ref_uncertainty_inv = ref_uncertainty.permute(0, 1, 4, 2, 3), ref_uncertainty_inv.permute(0, 1, 4, 2, 3)
                pair_images_tgt, pair_images_ref = images[:, tgt_idx], images[:, ref_idx]
                
            poses = (poses,(tgt_uncertainty.permute(0, 1, 4, 2, 3), ref_uncertainty))
            
            poses_inv = (poses_inv, (tgt_uncertainty_inv.permute(0, 1, 4, 2, 3), ref_uncertainty_inv))
        
        return poses, poses_inv, tgt_depths, ref_depths, tgt_intrinsics, ref_intrinsics, tgt_idx, ref_idx, pair_images_tgt, pair_images_ref
    
    def get_pairs_uncertainty(self, aggregated_tokens_list, images, tgt_idx, ref_idx):
        pair_tokens_list = [
            torch.cat([t[:, tgt_idx], t[:, ref_idx]], dim=-1)
            for t in aggregated_tokens_list
        ]

        pair_images_tgt = images[:, tgt_idx]
        
        tgt_uncertainty, ref_uncertainty = self.uncertainty_head(pair_tokens_list, 
                                                                pair_images_tgt, 
                                                                self.model.aggregator.patch_start_idx)
        
        del pair_tokens_list
        
        return tgt_uncertainty, ref_uncertainty, pair_images_tgt
        
    
    def get_pairs_pose(self, last_aggregated_tokens, images, tgt_idx, ref_idx, pos=None):
        
        pred_pose_enc_forward = self.camera_head(last_aggregated_tokens[:, tgt_idx],
                                                last_aggregated_tokens[:, ref_idx],
                                                self.model.aggregator.patch_start_idx, 
                                                images, 
                                                pos)
        
        pred_pose_enc_backward = self.camera_head(last_aggregated_tokens[:, ref_idx],
                                                last_aggregated_tokens[:, tgt_idx],
                                                self.model.aggregator.patch_start_idx, 
                                                images, 
                                                pos)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            poses, _ = pose_encoding_to_extri_intri(pred_pose_enc_forward, images.shape[-2:], pose_encoding_type=self.cfg.model.camera_head.pose_encoding_type)
            poses_inv, _ = pose_encoding_to_extri_intri(pred_pose_enc_backward, images.shape[-2:], pose_encoding_type=self.cfg.model.camera_head.pose_encoding_type)

        # poses: [B, N, 3, 4]

        # Get shape
        B, N, _, _ = poses.shape

        # Bottom row [0, 0, 0, 1]
        bottom_row = torch.tensor([0, 0, 0, 1], dtype=poses.dtype, device=poses.device)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(B, N, 1, 4)
        
        # Concatenate to make 4x4
        poses = torch.cat([poses, bottom_row], dim=2)        # [B, N, 4, 4]
        poses_inv = torch.cat([poses_inv, bottom_row], dim=2)  # [B, N, 4, 4]
        
        return poses, poses_inv
    
    
    
    
