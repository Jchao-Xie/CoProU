# code adapted from "https://github.com/ffrivera0/reloc3r/blob/main/reloc3r/pose_head.py" , 
# "https://github.com/yyfz/Pi3/blob/main/pi3/models/layers/camera_head.py"


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from pdb import set_trace as bb
from models.vggt.layers.block import Block
from torch.utils.checkpoint import checkpoint


# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # # change 1x1 convolution to linear
        # self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        # self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        # self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res


# parts of the code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L193'
class reloc3r_PoseHead(nn.Module):
    """ 
    pose regression head
    """
    def __init__(self, 
                 patch_size, 
                 dec_embed_dim, 
                 num_resconv_block=2,
                 rot_representation='9D'):
        super().__init__()
        self.patch_size = patch_size
        self.num_resconv_block = num_resconv_block
        self.rot_representation = rot_representation  

        output_dim = 4*self.patch_size**2

        self.proj = nn.Linear(dec_embed_dim, output_dim)
        self.res_conv = nn.ModuleList([copy.deepcopy(ResConvBlock(output_dim, output_dim)) 
            for _ in range(self.num_resconv_block)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        if self.rot_representation=='9D':
            self.fc_rot = nn.Linear(output_dim, 9)
        elif self.rot_representation=='6D':
            self.fc_rot = nn.Linear(output_dim, 6)
        else:
            self.fc_rot = nn.Linear(output_dim, 3)
        
    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r

    def rotation_6d_to_matrix(self, d6):  # code from pytorch3d
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    
    def rotation_euler_to_matrix(self, d3):  
        pass
    
    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        if self.rot_representation=='9D':
            out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        elif self.rot_representation=='6D':
            out_r = self.rotation_6d_to_matrix(out_r)
        else:
            out_r = self.rotation_euler_to_matrix(out_r)
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def forward(self, tokens, patch_h, patch_w):
        BN, hw, c = tokens.shape
        
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(BN, -1, patch_h, patch_w)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        out_t = self.fc_t(feat)  # [B,3]
        out_r = self.fc_rot(feat)  # [B,3]
        
        res = torch.concat([out_t, out_r], dim=-1) # [B, 6]

        return res
    
    
class PoseHead(nn.Module):
    """
    Predicting the relative poses between target images and reference images
    """
    
    def __init__(
        self,
        dim_in: int = 2048,
        depth: int = 4,
        pose_encoding_type: str = "absT_Euler",
        num_heads: int = 16,
        mlp_ratio: int = 1,
        init_values: float = 0.01,
        patch_size: int = 14, 
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope=None,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        
        self.patch_size = patch_size
        # decoder for the feature from AA 
        self.rope = rope
        self.depth = depth
        dim_dec = dim_in//4
        self.proj = nn.Linear(dim_in//2, dim_dec)
        self.token_norm = nn.LayerNorm(dim_dec*2)
        
        self.pose_decoder = nn.ModuleList(
            [
                Block(
                    dim=dim_dec*2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        
        self.pose_head = reloc3r_PoseHead(
                 patch_size, 
                 dim_dec*2, 
                 num_resconv_block=2,
                 rot_representation='3D' if pose_encoding_type == "absT_Euler" else '9D')
        
    def forward(self, 
                tgt_token: torch.Tensor, 
                ref_token: torch.Tensor, 
                ps_idx: int, images: torch.Tensor, 
                pos=None) -> torch.Tensor:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.

        Returns:
            predicted pose representation
        """
        
        _, _, _, H, W = images.shape
        
        assert tgt_token.shape[:-1] == ref_token.shape[:-1]
        pair_tokens = torch.cat([self.proj(tgt_token), self.proj(ref_token)], dim=-1)
        pair_tokens = self.token_norm(pair_tokens)
        
        B, S, P, C = pair_tokens.shape
        tokens = pair_tokens.view(B*S, P, C)

        for block_idx in range(self.depth):
            if self.training:
                tokens = checkpoint(self.pose_decoder[block_idx], tokens, pos, use_reentrant=False)
            else:
                tokens = self.pose_decoder[block_idx](tokens, pos=pos)
            
        # tokens = tokens.view(B, S, P, C)
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        tokens = tokens[:, ps_idx:]
        pred_pose_enc = self.pose_head(tokens, H // self.patch_size, W // self.patch_size)
        
        pred_pose_enc = pred_pose_enc.reshape(B, S, -1)
        return pred_pose_enc