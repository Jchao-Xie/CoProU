import math
from typing import List, Sequence, Tuple, Dict, Any
from torch.utils.data import Dataset
import torch
from models.vggt.utils.load_fn import load_and_preprocess_images
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

class SlidingWindowImageDataset(Dataset):
    def __init__(
        self,
        test_files: Sequence[str],
        window_length: int,
        sliding_step: int,
        img_size: int,
        patch_size: int,
        load_and_preprocess_images, 
        rescale_factor: float = 1.0,
        intrinsic: np.ndarray = None,
    ):
        # 1) Apply eval_interval subsampling here
        self.files = test_files
        self.n = len(self.files)

        self.window_length = min(self.n, window_length)
        self.sliding_step = sliding_step
        self.img_size = img_size
        self.patch_size = patch_size
        self.load_and_preprocess_images = load_and_preprocess_images
        self.rescale_factor = rescale_factor
        self.intrinsic = intrinsic

        # 2) Precompute all windows (list of list[int])
        self.windows = self._build_windows()

    def _build_windows(self) -> List[List[int]]:
        n = self.n
        L = self.window_length
        s = self.sliding_step

        # Your formula (kept) for number of windows
        sliding_times = (n - L + s) // s + (1 if ((n - L + s) % s != 0) else 0)
        self.sliding_times = sliding_times

        if n == 0:
            return []
        if n <= L:
            return [{"img_idx": list(range(n)),
                     "sliding_step": self.sliding_step}]

        windows = []
        # img_idx: List[int] = []
        for iter in range(sliding_times):
            # the step of last iteration is different
            if iter == sliding_times - 1 and ((n % self.sliding_step) != (self.window_length - self.sliding_step)) and sliding_times > 1:
                self.sliding_step = n - (self.sliding_step*(iter-1) + L)
                assert self.sliding_step < self.window_length
            
            if iter == 0:
                img_idx = [i + iter * self.sliding_step for i in range(self.window_length)]
            elif iter == sliding_times - 1:
                # img_idx = [i for i in range(n-(self.sliding_step +1), n)]
                img_idx = [i + self.sliding_step for i in img_idx]
            else:
                img_idx = [i + self.sliding_step for i in img_idx]
                
            windows.append({"img_idx":img_idx,
                            "sliding_step":self.sliding_step})
        
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> torch.Tensor:
        w = self.windows[i]
        idx = w["img_idx"]
        sliding_step = w["sliding_step"]
        paths = [self.files[j] for j in idx]        

        # IMPORTANT: return [T,C,H,W] on CPU; DataLoader workers do this in parallel
        imgs = self.load_and_preprocess_images(paths, target_size=self.img_size, patch_size=self.patch_size, rescale_factor=self.rescale_factor, intrinsic=self.intrinsic)  # [T,C,H,W] (ideally)
        return imgs, sliding_step, idx

def make_dataloader(test_files, window_length, sliding_step, img_size, patch_size=14, rescale_factor=1.0, intrinsic=None): 
    # test_files_path = Path(test_files_path)
    # test_files = sorted(test_files_path.glob("*.jpg"))
        
    dataset = SlidingWindowImageDataset(
                test_files=test_files,
                window_length=window_length,
                sliding_step=sliding_step,
                img_size=img_size,
                patch_size=patch_size,
                rescale_factor=rescale_factor,
                intrinsic=intrinsic,
                load_and_preprocess_images=load_and_preprocess_images,
            )
    loader = DataLoader(
        dataset,
        batch_size=1,          # keep 1 if your model expects one window at a time
        shuffle=False,
        num_workers=2,
        pin_memory=True,                # faster H2D copies
        persistent_workers=(2 > 0),
        prefetch_factor=4,              # each worker prefetches batches
        drop_last=False,
    )
    return loader

if __name__ == "__main__":
    test_files_path = Path('storage/waymo_original_size/waymo_original_size_val/segment-191862526745161106_1400_000_1420_000_with_camera_labels/images')
    test_files = sorted(test_files_path.glob("*.jpg"))
    loader = make_dataloader(test_files=test_files,
                             window_length=100,
                             sliding_step=50,
                             img_size=518)
    for windows in loader:
        # windows_imgs: [B, T, C, H, W] if your preprocess returns [T,C,H,W]
        windows_imgs, sliding_step, idx = windows
        windows_imgs = windows_imgs.to('cuda')
    