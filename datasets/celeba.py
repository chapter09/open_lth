import numpy as np
import random
import os
from PIL import Image
import sys
import torch
import torchvision

from ..datasets import base
from ..platforms.platform import get_platform


class Dataset(base.ImageDataset):
    """The CelebA dataset."""
    
    #total data size is 202599
    #don't use this in the registry
    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    #attr test, only 0/1
    @staticmethod
    def num_classes(): return 2

    @staticmethod
    def get_train_set(use_augmentation):
        #_image_transforms for celeba dataset
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = torchvision.datasets.celebA(train=True, split='train', target_type='attr', 
                                    root=os.path.join(get_platform().dataset_root, 'celeba'), download=True)

        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])
    

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.celebA(train=False, split='test', target_type='attr',
                                root=os.path.join(get_platform().dataset_root, 'celeba'), download=True)
        return Dataset(test_set.data, np.array(test_set.targets))

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def example_to_image(self, example):
        return Image.fromarray(example)



DataLoader = base.DataLoader