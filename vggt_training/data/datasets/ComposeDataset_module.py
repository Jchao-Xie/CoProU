import lightning as L
from torch.utils.data import DataLoader
import random
from hydra.utils import instantiate

class MultiDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.random_val_indices = []

    def setup(self, stage=None):
        self.train_dataset = instantiate(self.cfg.data.train, _recursive_=False)
        
        self.val_dataset = instantiate(self.cfg.data.val, _recursive_=False)
        
        self.random_val_indices = random.sample(range(self.cfg.num_batches_epoch_val), 20)
        self.random_train_indices = random.sample(range(self.cfg.num_batches_epoch_train), self.cfg.num_batches_epoch_train // 500)
    
    def train_dataloader(self):
        print(f"[DataModule] Epoch {self.trainer.current_epoch}, Rank {self.trainer.global_rank}, setting training loader")
        return self.train_dataset.get_loader(epoch=self.trainer.current_epoch + self.trainer.global_rank)

    def val_dataloader(self):
        print(f"[DataModule] Epoch {self.trainer.current_epoch}, Rank {self.trainer.global_rank}, setting validation loader")
        return self.val_dataset.get_loader(epoch=self.trainer.current_epoch + self.trainer.global_rank)