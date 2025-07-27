import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import custom_transforms
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from datasets.nuscenes_config.splits import train as training_list, val as validation_list
import random
import os
import cv2
from tqdm import tqdm


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.train = train
        if dataset == "kitti":
            scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
            self.scenes = sorted([self.root / folder.strip() for folder in open(scene_list_path)], key=lambda x: x.name)
        elif dataset == "nuscenes":
            if train:
                self.scenes = sorted([
                        os.path.join(self.root, folder.name)
                        for folder in self.root.iterdir()
                        if folder.is_dir() and folder.name.endswith("_0") and folder.name.split("_")[0] in training_list
                    ])
            else:
                self.scenes = sorted([
                        os.path.join(self.root, folder.name)
                        for folder in self.root.iterdir()
                        if folder.is_dir() and folder.name.endswith("_0") and folder.name.split("_")[0] in validation_list
                    ])[:70]
        self.transform = transform
        self.post_transform = custom_transforms.Compose([
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
        ])
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in tqdm(self.scenes):
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3)) if self.dataset == "kitti" else np.genfromtxt(scene/'intrinsics.txt').astype(np.float32)
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        self.samples = sequence_set
        
    def depth_anything_transform(self, raw_image, input_size=256):        
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
        
        image = raw_image / 255.0
        
        image = transform({'image': image})['image']

        return image
        

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        modified_img = []
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
            
        modified_img = [self.depth_anything_transform(img) for img in [tgt_img] + ref_imgs]
        modified_img = np.array(modified_img)
        
        if self.post_transform is not None:
            imgs, intrinsics = self.post_transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), modified_img

    def __len__(self):
        return len(self.samples)
