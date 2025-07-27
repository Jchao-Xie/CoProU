import argparse

import datetime
from path import Path

import torch
from torch.utils.data import DataLoader


import custom_transforms
from utils import tensor2array
from datasets.sequence_folders import SequenceFolder
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
from depth_anything_v2.dpt import fine_tuning_DepthAnythingV2
import models

import lightning as L



parser = argparse.ArgumentParser(description='CoProU-VO on KITTI and nuScenes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--skip-frames', type=int, metavar='N', help='inter-frame interval', default=1)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adamw')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adamw')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nuscenes'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--dan', action='store_true', help='Enable depth anything mode')
parser.add_argument('--dinov2', action='store_true', help='Enable dinov2 mode')
parser.add_argument('--unfrozen-backbone', action='store_true', help='Unfreeze backbone')
parser.add_argument('--no-aug', action='store_true', help='Disable augmentation mode')
parser.add_argument('--resume-path', type=str, default=None, help='Path to checkpoint to resume training from (optional)')


# define the LightningModule
class CoProU(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)  # saves all args into self.hparams
        model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
        self.pose_net = models.PoseResNet(18, args.with_pretrain)
        self.disp_net = fine_tuning_DepthAnythingV2(**model_configs[args.encoder])
        
        if args.dan:
            weights = torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
            weights = {k: v for k, v in weights.items() if 'pretrained' in k}
            self.disp_net.load_state_dict(weights, strict=False)
            for param in self.disp_net.pretrained.parameters():
                param.requires_grad = False

        if args.dinov2:
            weights = torch.load(f"checkpoints/dinov2_{args.encoder}14_pretrain.pth")
            self.disp_net.pretrained.load_state_dict(weights)
            for param in self.disp_net.pretrained.parameters():
                param.requires_grad = False

        if args.pretrained_disp:
            print("=> using pre-trained weights for DispResNet")
            weights = torch.load(args.pretrained_disp)
            self.disp_net.load_state_dict(weights['state_dict'], strict=False)
            
        if args.pretrained_pose:
            print("=> using pre-trained weights for PoseResNet")
            weights = torch.load(args.pretrained_pose)
            self.pose_net.load_state_dict(weights['state_dict'], strict=False)
            
        if args.unfrozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, tgt_img, ref_imgs, input_size, res_tgt_img=None, res_ref_imgs=None):
            return self.model(tgt_img, ref_imgs, input_size, res_tgt_img, res_ref_imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.momentum, self.hparams.beta),
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    
    def training_step(self, batch, batch_idx):
        tgt_img, ref_imgs, intrinsics, intrinsics_inv, DAt_imgs = batch
        w1, w2, w3 = self.hparams.photo_loss_weight, self.hparams.smooth_loss_weight, self.hparams.geometry_consistency_weight

        DAt_tgt_img = DAt_imgs[:, 0]
        DAt_ref_imgs = [DAt_imgs[:, i+1] for i in range(DAt_imgs.shape[1]-1)]
        
        tgt_depth, ref_depths = self.compute_depth(DAt_tgt_img, DAt_ref_imgs, tgt_img.size()[-2:])
        poses, poses_inv = self.compute_pose_with_inv(tgt_img, ref_imgs)
        

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, self.hparams.num_scales, self.hparams.with_ssim,
                                                         self.hparams.with_mask, self.hparams.with_auto_mask, self.hparams.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        total_loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        self.log("train total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train photo_loss", loss_1, on_step=True, on_epoch=True)
        self.log("train smooth_loss", loss_2, on_step=True, on_epoch=True)
        self.log("train geometry_loss", loss_3, on_step=True, on_epoch=True)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        tgt_img, ref_imgs, intrinsics, intrinsics_inv, DAt_imgs = batch
        DAt_tgt_img = DAt_imgs[:, 0]
        DAt_ref_imgs = [DAt_imgs[:, i+1] for i in range(DAt_imgs.shape[1]-1)]

        tgt_depth, ref_depths = self.compute_depth(DAt_tgt_img, DAt_ref_imgs, tgt_img.size()[-2:])
        poses, poses_inv = self.compute_pose_with_inv(tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, self.hparams.num_scales, self.hparams.with_ssim,
                                                         self.hparams.with_mask, False, self.hparams.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        total_loss = loss_1

        self.log("val photo_loss", loss_1.item(), prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val smooth_loss", loss_2.item(), prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val geometry_loss", loss_3.item(), prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val total_loss", total_loss.item(), prog_bar=True, on_epoch=True, sync_dist=True)
        
        if batch_idx in self.trainer.datamodule.random_val_indices:
            tb = self.logger.experiment
            epoch = self.current_epoch

            if epoch == 0:
                tb.add_image('val Input', tensor2array(tgt_img[0]), epoch)

            tb.add_image('val Dispnet Output Normalized',
                        tensor2array(1.0 / tgt_depth[0][0][0], max_value=None, colormap='magma'),
                        epoch)

            tb.add_image('val Depth Output',
                        tensor2array(tgt_depth[0][0][0], max_value=10),
                        epoch)

            tb.add_image('val Uncertainty',
                        tensor2array(tgt_depth[0][1][0], max_value=None),
                        epoch)
    
    def compute_depth(self, tgt_img, ref_imgs, input_size):
        tgt_depth = [depth for depth in self.disp_net(tgt_img, input_size)]

        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [depth for depth in self.disp_net(ref_img, input_size)]
            ref_depths.append(ref_depth)

        return tgt_depth, ref_depths
    
    def compute_pose_with_inv(self, tgt_img, ref_imgs):
        poses = []
        poses_inv = []
        for ref_img in ref_imgs:
            poses.append(self.pose_net(tgt_img, ref_img))
            poses_inv.append(self.pose_net(ref_img, tgt_img))

        return poses, poses_inv
        
class VODataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.random_val_indices = []

    def setup(self, stage=None):
        args = self.args

        if args.no_aug:
            print("Augmentation disabled")
            train_transform = custom_transforms.Compose([])
        else:
            train_transform = custom_transforms.Compose([
                custom_transforms.RandomHorizontalFlip(),
                custom_transforms.RandomScaleCrop(),
            ])

        valid_transform = None

        self.train_dataset = SequenceFolder(
            args.data, transform=train_transform, seed=args.seed, train=True,
            sequence_length=args.sequence_length, skip_frames=args.skip_frames, dataset=args.dataset
        )

        self.val_dataset = SequenceFolder(
            args.data, transform=valid_transform, seed=args.seed, train=False,
            sequence_length=args.sequence_length, skip_frames=args.skip_frames, dataset=args.dataset
        )

        print(f"🔹 {len(self.train_dataset)} training samples")
        print(f"🔹 {len(self.val_dataset)} validation samples")
        
        self.random_val_indices = random.sample(range(len(self.val_dataloader()) // self.trainer.num_devices), 5)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            prefetch_factor=4,
            drop_last=False
        )

if __name__ == "__main__":
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    import random

    args = parser.parse_args()

    # Build model and data
    data_module = VODataModule(args)  
    model = CoProU(args)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path("checkpoints") / args.name / timestamp

    logger = TensorBoardLogger(
        save_dir=str(save_path.parent),  # e.g., "checkpoints/YourModel"
        name=save_path.name,             # e.g., timestamp like "07-24-11:00"
        version=""                       # ← disables automatic 'version_0' subfolder
    )
    print("Will log TensorBoard to:", save_path.absolute())

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,  # where to save checkpoints
        filename="checkpoint_{epoch:02d}",        # file naming pattern
        save_top_k=-1,                      # save ALL checkpoints, not just best
        every_n_epochs=1                    # save every epoch
    )

    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        # precision="16-mixed",
        max_epochs=args.epochs,
    )

    # Train!
    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.resume_path)