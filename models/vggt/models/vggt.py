# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from models.vggt.models.aggregator import Aggregator, Aggregator_small, Aggregator_base, Aggregator_large
from models.vggt.heads.camera_head import CameraHead
from models.vggt.heads.dpt_head import DPTHead
from models.vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, model_type = "dinov2_vitl14_reg",
                 features = 256, out_channels = [256, 512, 1024, 1024], intermediate_layer_idx = [4, 11, 17, 23],
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        """adapted VGGT __init__, which accpet different model size

        Args:
            img_size (int, optional): Defaults to 518.
            patch_size (int, optional): Defaults to 14.
            patch embed type (str, optional): Default to "dinov2_vitl14_reg" follow VGGT.
            features (int, optional): Feature channels for intermediate representations in DPT head. Default is 256.
            out_channels (List[int], optional): Output channels for each intermediate layer for DPT head.
            intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
            enable_camera (bool, optional): Defaults to True.
            enable_point (bool, optional):  Defaults to True.
            enable_depth (bool, optional):  Defaults to True.
            enable_track (bool, optional):  Defaults to True.
        """
        super().__init__()
        Aggregators_type = {
                "dinov2_vitl14_reg": Aggregator_large,
                "dinov2_vitb14_reg": Aggregator_base,
                "dinov2_vits14_reg": Aggregator_small,
            }
        self.aggregator = Aggregators_type[model_type](img_size=img_size, patch_size=patch_size)

        self.camera_head = CameraHead(dim_in=2 * self.aggregator.patch_embed.embed_dim) if enable_camera else None
        
        self.point_head = DPTHead(dim_in=2 * self.aggregator.patch_embed.embed_dim, features=features, 
                                  out_channels=out_channels, output_dim=4, activation="inv_log", 
                                  conf_activation="expp1") if enable_point else None
        
        self.depth_head = DPTHead(dim_in=2 * self.aggregator.patch_embed.embed_dim, features=features, 
                                  out_channels=out_channels, output_dim=2, activation="exp", 
                                  conf_activation="expp1") if enable_depth else None
        
        self.track_head = TrackHead(dim_in=2 * self.aggregator.patch_embed.embed_dim, 
                                    patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

