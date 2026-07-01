from __future__ import division
import torch
import random
import numpy as np
from PIL import Image

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


def normalize(images, intrinsics, mean, std):
    """
    Normalizes a list of image tensors (C x H x W) using channel-wise mean and std.
    """
    for tensor in images:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
    return images, intrinsics

def array_to_tensor(images, intrinsics):
    """
    Converts a list of numpy.ndarray (H x W x C) to a list of torch.FloatTensor (C x H x W),
    scaling pixel values to [0,1].
    """
    tensors = []
    for im in images:
        im = np.transpose(im, (2, 0, 1))  # HWC → CHW
        tensors.append(torch.from_numpy(im).float() / 255.0)
    return tensors, intrinsics
    
def random_horizontal_flip(images, intrinsics, p=0.5):
    """
    Randomly flips the given images horizontally with probability p.
    Adjusts intrinsics accordingly.
    """
    assert intrinsics is not None
    if random.random() < p:
        output_intrinsics = np.copy(intrinsics)
        output_images = [np.copy(np.fliplr(im)) for im in images]
        w = output_images[0].shape[1]
        output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
    else:
        output_images = images
        output_intrinsics = intrinsics
    return output_images, output_intrinsics



class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics



def random_scale_crop(images, intrinsics, scale_range=(1.0, 1.15)):
    """
    Randomly scales images up to 15% and crops back to original size.
    """
    assert intrinsics is not None
    output_intrinsics = np.copy(intrinsics)

    in_h, in_w, _ = images[0].shape
    x_scaling, y_scaling = np.random.uniform(*scale_range, 2)
    scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

    output_intrinsics[0] *= x_scaling
    output_intrinsics[1] *= y_scaling

    scaled_images = [
        np.array(
            Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))
        ).astype(np.float32)
        for im in images
    ]

    offset_y = np.random.randint(scaled_h - in_h + 1)
    offset_x = np.random.randint(scaled_w - in_w + 1)

    cropped_images = [
        im[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        for im in scaled_images
    ]

    output_intrinsics[0, 2] -= offset_x
    output_intrinsics[1, 2] -= offset_y

    return cropped_images, output_intrinsics


def ratio_crop(images, intrinsics, target_ratio):
    """
    Crops input images to match a specified target aspect ratio (width / height)
    while updating intrinsics accordingly.
    """
    assert len(images) > 0, "Expected a list of images."
    assert intrinsics is not None, "Intrinsics must be provided."

    in_h, in_w, _ = images[0].shape
    output_intrinsics = np.copy(intrinsics)

    orig_ratio = in_h / in_w   # height / width
    tgt_ratio  = float(target_ratio)
    
    if orig_ratio > tgt_ratio:
        # image too tall → crop height
        new_h = int(in_w * tgt_ratio)
        new_w = in_w
    else:
        # image too wide → crop width
        new_w = int(in_h / tgt_ratio)
        new_h = in_h

    cx_img, cy_img = in_w / 2, in_h / 2
    start_x = int(cx_img - new_w / 2)
    start_y = int(cy_img - new_h / 2)
    end_x = start_x + new_w
    end_y = start_y + new_h

    cropped_images = [
        np.array(Image.fromarray(im.astype(np.uint8)).crop((start_x, start_y, end_x, end_y))).astype(np.float32)
        for im in images
    ]

    output_intrinsics[0, 2] -= start_x
    output_intrinsics[1, 2] -= start_y
    return cropped_images, output_intrinsics
    
def resize_images(images, intrinsics, target_size):
    """
    Resizes images to (target_h, target_w) and scales intrinsics accordingly.
    """
    assert intrinsics is not None
    output_intrinsics = np.copy(intrinsics)

    in_h, in_w, _ = images[0].shape
    target_h, target_w = target_size
    
    assert target_h <= in_h and target_w <= in_w, (
        f"Target size ({target_h}, {target_w}) must be smaller than or equal to "
        f"original size ({in_h}, {in_w})."
    )

    x_scaling = target_w / in_w
    y_scaling = target_h / in_h

    output_intrinsics[0] *= x_scaling
    output_intrinsics[1] *= y_scaling

    resized_images = [
        np.array(
            Image.fromarray(im.astype(np.uint8)).resize(
                (target_w, target_h), resample=Image.LANCZOS
            )
        ).astype(np.float32)
        for im in images
    ]

    return resized_images, output_intrinsics
