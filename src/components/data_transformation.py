import os
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from src.logger import logging
from src.exception import CustomException

class CIFAR10DataHandler:
    def __init__(self, data_dir, batch_size, num_workers=3, pin_memory=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def get_data_loaders(self):
        try:
            logging.info("Creating data loaders")
            train_tfms = tt.Compose([
                tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                tt.RandomHorizontalFlip(),
                # Additional transforms can be added here
                tt.ToTensor(),
                tt.Normalize(*self.stats, inplace=True)
            ])
            valid_tfms = tt.Compose([
                tt.ToTensor(),
                tt.Normalize(*self.stats)
            ])

            train_ds = ImageFolder(os.path.join(self.data_dir, 'train'), train_tfms)
            valid_ds = ImageFolder(os.path.join(self.data_dir, 'test'), valid_tfms)
            train_dl = DataLoader(train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)
            valid_dl = DataLoader(valid_ds, self.batch_size * 2, num_workers=self.num_workers,
                                  pin_memory=self.pin_memory)

            logging.info("Data loaders created successfully")
            return train_dl, valid_dl
        except Exception as e:
            logging.error("Error occurred while creating data loaders")
            raise CustomException(e, sys)
