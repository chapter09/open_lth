import numpy as np
import pandas as pd
import random
import os
from PIL import Image
import sys
import torch
import torchvision
import torch.utils.data 
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from ..datasets import base
from ..platforms.platform import get_platform


class CelebaDataset(VisionDataset):

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

        self.img_idx = df.index.values
        self.targets = df['Male'].values
        self.transform = transform

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = Image.open(os.path.join(self.root_dir, self.img_idx[idx]))
        target = self.targets[idx]
        
        if self.transform:
            X = self.transform(X)
        
        return X, target


class Dataset(base.ImageDataset):
    """The CelebA dataset."""
    
    @staticmethod
    def num_train_examples(): pass

    @staticmethod
    def num_test_examples(): pass

    #attr test, only 0/1
    @staticmethod
    def num_classes(): return 2

    @staticmethod
    def get_train_set(use_augmentation):
        #_image_transforms for celeba dataset
        csv_path = '/mnt/open_lth_datasets/CelebA/data/train/celeba-gender-train.csv'
        root_path = '/mnt/open_lth_datasets/CelebA/data/img_align_celeba/img_align_celeba'
        transform = transforms.Compose([
            transforms.CenterCrop((178, 178)),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])
        train_set = CelebaDataset(csv_path, root_path, transform=transform)
        data = []
        for i in range(len(train_set)):
            x,_ =train_set[i]

            data.append(x)

        data = np.vstack(data).reshape(-1, 3, 84, 84)
        data = data.transpose((0, 2, 3, 1))
        
        return Dataset(data.astype(np.uint8), np.asarray(train_set.targets))
    

    @staticmethod
    def get_test_set():
        pass
  

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example)



DataLoader = base.DataLoader