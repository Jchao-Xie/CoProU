import argparse
import time

import datetime
from path import Path

import torch
from torch.utils.data import dataloader
from hydra.utils import instantiate
from hydra import initialize, compose

from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_photo_and_geometry_loss_multi_frames

import math 
import models 

import lightning as L

from vggt_training.train_utils.freeze import freeze_modules
from models.vggt.heads.dpt_head import DPTHead
from models.PoseHead import PoseHead

from vggt_training.train_utils.optimizer import construct_optimizers
from vggt_training.train_utils.general import AverageMeter
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from coprou_training.train_util import load_pretrained, safe_inverse


from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import random
from models.vggt.models.vggt import VGGT

from vggt_training.data.datasets.ComposeDataset_module import MultiDataModule


parser = argparse.ArgumentParser(description='CoProU-VO on Waymo and nuScenes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', type=str, default="coprou_mf.yaml", help='Optimizer config')

from models.model_module import coprou

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    with initialize(version_base=None, config_path="vggt_training/config"):
        cfg = compose(config_name=args.config)
    
    # Build model and data
    data_module = MultiDataModule(cfg)  
    model = coprou(cfg)

    
    if cfg.train.load_weights_path is not None:
        ckpt = torch.load(cfg.train.load_weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"Loaded model weights from {cfg.train.load_weights_path}")

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path("checkpoints") / cfg.experiment.name / timestamp

    logger = TensorBoardLogger(
        save_dir=str(save_path.parent),  # e.g., "checkpoints/YourModel"
        name=save_path.name,             # e.g., timestamp like "07-24-11:00"
        version=""                       # ← disables automatic 'version_0' subfolder
    )
    print("Will log TensorBoard to:", save_path.absolute())

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,  # where to save checkpoints
        filename="checkpoint_{epoch:02d}",        # file naming pattern
        save_top_k=-1,                     
        # every_n_epochs=1,                   
        every_n_train_steps=cfg.train.ckpt_every_n_train_steps,     
    )

    trainer = Trainer(
        strategy="ddp" if cfg.optim.frozen_module_names is None and cfg.model.camera_head.type is not None else "ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        precision=cfg.optim.amp.precision,  
        max_epochs=-1,
        max_steps=cfg.train.max_steps,
        use_distributed_sampler=False,
        reload_dataloaders_every_n_epochs=1,
        limit_train_batches=cfg.num_batches_epoch_train,
        limit_val_batches=cfg.num_batches_epoch_val,
    )

    # Train!
    trainer.fit(model=model, 
                datamodule=data_module, 
                ckpt_path=cfg.train.resume_path,
                )