import argparse
import time
import csv
import datetime
from path import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

import models

import custom_transforms
from utils import tensor2array, save_checkpoint
from datasets.sequence_folders import SequenceFolder
from datasets.pair_folders import PairFolder
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from depth_anything_v2.dpt import fine_tuning_DepthAnythingV2, fine_tuning_DPTHead
from utils import depth_visualization, unnormalize_and_save
from inverse_warp import pose_vec2mat
from kitti_eval.kitti_odometry import KittiEvalOdom


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
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu', 'nuscenes'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-ph', action='store_true', help='use photometric error for validation.')
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--dan', action='store_true', help='Enable depth anything mode')
parser.add_argument('--dinov2', action='store_true', help='Enable dinov2 mode')
parser.add_argument('--no-aug', action='store_true', help='Disable augmentation mode')

best_error = 100
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# torch.autograd.set_detect_anomaly(True)  ### detect infinite values

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    print(args)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    if args.no_aug:
        print("disable augumentation")
        train_transform = custom_transforms.Compose([
        # custom_transforms.RandomHorizontalFlip(),
        # custom_transforms.RandomScaleCrop(),
        # custom_transforms.ArrayToTensor(),
        # normalize
    ])
    else:
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            # custom_transforms.ArrayToTensor(),
            # normalize
        ])

    valid_transform = None
    # custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            skip_frames=args.skip_frames,
            dataset=args.dataset
        )
    else:
        train_set = PairFolder(
            args.data,
            seed=args.seed,
            train=True,
            transform=train_transform
        )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        skip_frames=args.skip_frames,
        dataset=args.dataset
    )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, prefetch_factor=4, drop_last=False)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    if not args.debug and not args.no_aug:
        import random
        global RANDOM_VAL_INDICES
        RANDOM_VAL_INDICES = random.sample(range(len(val_loader)), len(output_writers))
    
    # create model
    print("=> creating model")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = fine_tuning_DepthAnythingV2(**model_configs[args.encoder])
    # depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    if args.dan:
        print("!!! Using depth anything pretrained model !!!")
        depth_anything.load_state_dict({k: v for k, v in torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu').items() if 'pretrained' in k}, strict=False)
        # depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'), strict=False)
        for param in depth_anything.pretrained.parameters():
            param.requires_grad = False
        # for param in depth_anything.parameters():
        #     param.requires_grad = False
        # for param in depth_anything.depth_head.scratch.output_conv1.parameters():
        #     param.requires_grad = True
        # for param in depth_anything.depth_head.scratch.output_conv2.parameters():
        #     param.requires_grad = True
        # for param in depth_anything.depth_head.scratch.uncertainty_conv1.parameters():
        #     param.requires_grad = True
        # for param in depth_anything.depth_head.scratch.uncertainty_conv2.parameters():
        #     param.requires_grad = True
    if args.dinov2:
        print("!!! Using dinov2 pretrained model !!!")
        depth_anything.pretrained.load_state_dict(torch.load("/home/stud/xiji/SC-Depth_Anything/checkpoints/dinov2_vits14_pretrain.pth"))
        for param in depth_anything.pretrained.parameters():
            param.requires_grad = False

    depth_anything = depth_anything.to(device)
    # disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    disp_net = depth_anything
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)
    
    # for name, module in disp_net.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.eval()  # Freeze running stats
    #         for param in module.parameters():
    #             param.requires_grad = False  # Freeze weights and bias
    #         print(f"Frozen BatchNorm2d in dispnet-> {name}: {module}")
            
    # for name, module in pose_net.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.eval()  # Freeze running stats
    #         for param in module.parameters():
    #             param.requires_grad = False  # Freeze weights and bias
    #         print(f"Frozen BatchNorm2d in posenet-> {name}: {module}")

    for name, param in pose_net.named_parameters():
        print(name, param.requires_grad)
    
    for name, param in disp_net.named_parameters():
        print(name, param.requires_grad)
        
    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    # for param in disp_net.pretrained.parameters():
    #     param.requires_grad = False
    # for name, param in disp_net.named_parameters():
    #     print(name, param.requires_grad)


    if torch.cuda.device_count() > 1:
        disp_net = torch.nn.DataParallel(disp_net)
        pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(optim_params,
                            betas=(args.momentum, args.beta),
                            weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()
    
    # if args.debug:
    #     for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, DAt_imgs) in enumerate(train_loader):
    #         pass

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        if args.debug is False:
            # train for one epoch
            logger.reset_train_bar()
            train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))


            # evaluate on validation set
            logger.reset_valid_bar()
            if args.with_ph:
                errors, error_names, VO_result = validate_with_photometric_error(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
            else:
                errors, error_names, VO_result = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
            logger.valid_writer.write(' * Avg {}'.format(error_string))

            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, epoch)
            
            if not args.with_ph:
                for k, vo_metric in enumerate(["translation error", "rotation error", "ATE"]):
                    training_writer.add_scalar(vo_metric, VO_result[k][0], epoch)


            # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            # decisive_error = errors[1]
            decisive_error = VO_result[2][0] if not args.with_ph else errors[1]
            if best_error < 0:
                best_error = decisive_error

            # remember lowest error and save checkpoint
            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
            save_checkpoint(
                args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': disp_net.module.state_dict() if isinstance(disp_net, torch.nn.DataParallel) else disp_net.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': pose_net.module.state_dict() if isinstance(pose_net, torch.nn.DataParallel) else pose_net.state_dict()
                },
                is_best)
            
            if epoch % 1 == 0:
                save_checkpoint(
                    args.save_path, {
                        'epoch': epoch + 1,
                        'state_dict': disp_net.module.state_dict() if isinstance(disp_net, torch.nn.DataParallel) else disp_net.state_dict()
                    }, {
                        'epoch': epoch + 1,
                        'state_dict': pose_net.module.state_dict() if isinstance(pose_net, torch.nn.DataParallel) else pose_net.state_dict()
                    },
                    False, filename=f'checkpoint_{epoch}.pth.tar')

            with open(args.save_path/args.log_summary, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([train_loss, decisive_error])
        else:
            logger.reset_train_bar()
            train_loss = overfitting_one_batch(args, train_loader, disp_net, pose_net, 
                                               depth_anything, optimizer, args.epoch_size, 
                                               logger, training_writer)
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

            if epoch % 1 == 0:
                save_checkpoint(
                    args.save_path, {
                        'epoch': epoch + 1,
                        'state_dict': disp_net.module.state_dict()
                    }, {
                        'epoch': epoch + 1,
                        'state_dict': pose_net.module.state_dict()
                    },
                    False)
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()
    # from torchsummary import summary
    # summary(pose_net, input_size=((3, 256, 832), (3, 256, 832)), device="cuda")
    # summary(disp_net, input_size=(3, 266, 826), device="cuda")

    # for m in disp_net.modules():    
    #     if isinstance(m, nn.BatchNorm2d):  
    #         m.eval()
    # for m in pose_net.modules():    
    #     if isinstance(m, torch.nn.BatchNorm2d):    
    #         print(m)   
    #         m.eval()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, DAt_imgs) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        
        DAt_tgt_img = DAt_imgs[:, 0].to(device)
        DAt_ref_imgs = [DAt_imgs[:, 1].to(device), DAt_imgs[:, 2].to(device)]
        
        inference_time_s = time.time()
        
        # DAt_imgs = DAt_imgs.to(device).flatten(start_dim=0, end_dim=1)
        # unnormalize_and_save(tgt_img[0], name="tgt_img.png")
        # unnormalize_and_save(DAt_imgs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], name="dat_img.png")
        # # depth_dat_imgs_end = time.time()
        # # print(f'depth_dat_imgs_time: {depth_dat_imgs_end - depth_dat_imgs_start}')
        # # depth_anything_start = time.time()
        # depth= depth_anything.infer_image(DAt_imgs, tgt_img.size())
        # depth_visualization(depth[0],depth[0],depth[0],'origin_depth')
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, DAt_tgt_img, DAt_ref_imgs, tgt_img.size()[-2:])
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if log_losses:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # inference_time_e = time.time()
        # print(f"inference time: {inference_time_e - inference_time_s}")
        
        # backpro_time_s = time.time()
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(f"update_time: {time.time() - backpro_time_s}")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    
    global_pose = np.eye(4)
    poses_list = [global_pose[0:3, :].reshape(1, 12)]
    
    # output_dir = Path("/home/stud/xiji/SC-Depth_Anything/tmp/")
    output_dir = os.path.join(args.save_path, "tmp/")
    output_dir.makedirs_p()
    
    eval_tool = KittiEvalOdom()
    gt_dir = "./kitti_eval/gt_poses/"

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, DAt_imgs) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

         
        DAt_tgt_img = DAt_imgs[:, 0].to(device)
        DAt_ref_imgs = [DAt_imgs[:, 1].to(device), DAt_imgs[:, 2].to(device)]
        
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, DAt_tgt_img, DAt_ref_imgs, tgt_img.size()[-2:])

        if log_outputs and i in RANDOM_VAL_INDICES:
            writer_idx = RANDOM_VAL_INDICES.index(i)
            if epoch == 0:
                output_writers[writer_idx].add_image('val Input', tensor2array(tgt_img[0]), 0)

            output_writers[writer_idx].add_image('val Dispnet Output Normalized',
                                        tensor2array(1 / tgt_depth[0][0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[writer_idx].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0][0], max_value=10),
                                        epoch)
            output_writers[writer_idx].add_image('uncertainty',
                                        tensor2array(tgt_depth[0][1][0], max_value=None),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()
        
        if i == 0:
            pose_mats = pose_vec2mat(poses_inv[0][:1]).detach().cpu().numpy()
            for k in range(pose_mats.shape[0]):
                pose_mat = np.vstack([pose_mats[k], np.array([0, 0, 0, 1])])
                global_pose = global_pose @  np.linalg.inv(pose_mat)
                poses_list.append(global_pose[0:3, :].reshape(1, 12))
        pose_mats = pose_vec2mat(poses[1]).detach().cpu().numpy()
        for k in range(pose_mats.shape[0]):
            pose_mat = np.vstack([pose_mats[k], np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)
            poses_list.append(global_pose[0:3, :].reshape(1, 12))

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    poses_list = np.concatenate(poses_list, axis=0)
    filename = Path(output_dir + "01" + ".txt")
    np.savetxt(filename, poses_list, delimiter=' ', fmt='%1.8e')
    VO_result = eval_tool.eval(
        "./kitti_eval/gt_poses/",
        output_dir,
        alignment='7dof',
        seqs=None,
    )
    
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss'], VO_result


@torch.no_grad()
def validate_with_photometric_error(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    
    # global_pose = np.eye(4)
    # poses_list = [global_pose[0:3, :].reshape(1, 12)]
    
    # # output_dir = Path("/home/stud/xiji/SC-Depth_Anything/tmp/")
    # output_dir = os.path.join(args.save_path, "tmp/")
    # output_dir.makedirs_p()
    
    # eval_tool = KittiEvalOdom()
    # gt_dir = "./kitti_eval/gt_poses/"
    # pred_rel_poses_list = [None] * len(val_loader.dataset)
    # gt_rel_poses_list = [None] * len(val_loader.dataset)
    # CURRENT = 0

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, DAt_imgs) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

         
        DAt_tgt_img = DAt_imgs[:, 0].to(device)
        DAt_ref_imgs = [DAt_imgs[:, 1].to(device), DAt_imgs[:, 2].to(device)]
        
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, DAt_tgt_img, DAt_ref_imgs, tgt_img.size()[-2:])

        if log_outputs and i in RANDOM_VAL_INDICES:
            writer_idx = RANDOM_VAL_INDICES.index(i)
            if epoch == 0:
                output_writers[writer_idx].add_image('val Input', tensor2array(tgt_img[0]), 0)

            output_writers[writer_idx].add_image('val Dispnet Output Normalized',
                                        tensor2array(1 / tgt_depth[0][0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[writer_idx].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0][0], max_value=10),
                                        epoch)
            output_writers[writer_idx].add_image('uncertainty',
                                        tensor2array(tgt_depth[0][1][0], max_value=None),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()
        
        # for i in range(poses[0].size(0)):
        #     gt_tgt_pose
        # if i == 0:
        #     pose_mats = pose_vec2mat(poses_inv[0][:1]).detach().cpu().numpy()
        #     for k in range(pose_mats.shape[0]):
        #         pose_mat = np.vstack([pose_mats[k], np.array([0, 0, 0, 1])])
        #         global_pose = global_pose @  np.linalg.inv(pose_mat)
        #         poses_list.append(global_pose[0:3, :].reshape(1, 12))
        # pose_mats = pose_vec2mat(poses[1]).detach().cpu().numpy()
        # for k in range(pose_mats.shape[0]):
        #     pose_mat = np.vstack([pose_mats[k], np.array([0, 0, 0, 1])])
        #     global_pose = global_pose @  np.linalg.inv(pose_mat)
        #     poses_list.append(global_pose[0:3, :].reshape(1, 12))
        

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    # poses_list = np.concatenate(poses_list, axis=0)
    # filename = Path(output_dir + "01" + ".txt")
    # np.savetxt(filename, poses_list, delimiter=' ', fmt='%1.8e')
    # VO_result = eval_tool.eval(
    #     "./kitti_eval/gt_poses/",
    #     output_dir,
    #     alignment='7dof',
    #     seqs=None,
    # )
    
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss'], None


def compute_depth(disp_net, tgt_img, ref_imgs, input_size):
    tgt_depth = [depth for depth in disp_net(tgt_img, input_size)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [depth for depth in disp_net(ref_img, input_size)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv

def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_RPE(gt, pred):
        """Compute RPE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            rpe_trans
            rpe_rot
        """
        trans_errors = []
        rot_errors = []
        for i in list(pred.keys())[:-1]:
            gt1 = gt[i]
            gt2 = gt[i+1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            pred2 = pred[i+1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            rel_err = np.linalg.inv(gt_rel) @ pred_rel

            trans_errors.append(translation_error(rel_err))
            rot_errors.append(rotation_error(rel_err))
        # rpe_trans = np.sqrt(np.mean(np.asarray(trans_errors) ** 2))
        # rpe_rot = np.sqrt(np.mean(np.asarray(rot_errors) ** 2))
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        return rpe_trans, rpe_rot

def overfitting_one_batch(args, train_loader, disp_net, pose_net, depth_anything, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()
    # from torchsummary import summary
    # summary(pose_net, input_size=((3, 256, 832), (3, 256, 832)), device="cuda")
    # summary(disp_net, input_size=(3, 266, 826), device="cuda")

    # for m in disp_net.modules():    
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         print(m)       
    #         m.eval()
    # for m in pose_net.modules():    
    #     if isinstance(m, torch.nn.BatchNorm2d):    
    #         print(m)   
    #         m.eval()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, DAt_imgs) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        
        DAt_tgt_img = DAt_imgs[:, 0].to(device)
        DAt_ref_imgs = [DAt_imgs[:, 1].to(device), DAt_imgs[:, 2].to(device)]
        
        inference_time_s = time.time()
        
        # DAt_imgs = DAt_imgs.to(device).flatten(start_dim=0, end_dim=1)
        # unnormalize_and_save(tgt_img[0], name="tgt_img.png")
        # unnormalize_and_save(DAt_imgs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], name="dat_img.png")
        # # depth_dat_imgs_end = time.time()
        # # print(f'depth_dat_imgs_time: {depth_dat_imgs_end - depth_dat_imgs_start}')
        # # depth_anything_start = time.time()
        # depth= depth_anything.infer_image(DAt_imgs, tgt_img.size())
        # depth_visualization(depth[0],depth[0],depth[0],'origin_depth')
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, DAt_tgt_img, DAt_ref_imgs, tgt_img.size()[-2:])
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if log_losses:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # inference_time_e = time.time()
        # print(f"inference time: {inference_time_e - inference_time_s}")
        
        # backpro_time_s = time.time()
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(f"update_time: {time.time() - backpro_time_s}")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10000 == 0:
                save_checkpoint(
                args.save_path, {
                    'epoch': i + 1,
                    'state_dict': disp_net.module.state_dict() if isinstance(disp_net, torch.nn.DataParallel) else disp_net.state_dict()
                }, {
                    'epoch': i + 1,
                    'state_dict': pose_net.module.state_dict() if isinstance(pose_net, torch.nn.DataParallel) else pose_net.state_dict()
                },
                False)
        

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]

if __name__ == '__main__':
    main()
