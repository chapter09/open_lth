# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import os
from PIL import Image
import sys
import torch
import torchvision

from datasets import base
from platforms.platform import get_platform


class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                super(CIFAR10, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples(client_num): return 50000 / client_num

    @staticmethod
    def num_test_examples(client_num): return 10000 / client_num

    @staticmethod
    def num_fl_train_examples(): return 500

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(cls, use_augmentation, client_num, bias_fraction):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        if bias_fraction:
            data, targets = Dataset.get_non_iid_set(train_set, client_num, bias_fraction)
        else:
            data = train_set.data
            targets = train_set.targets
        return Dataset(cls(np.array(data), np.array(targets)), augment if use_augmentation else [])
    
    

    @staticmethod
    def get_test_set():
        test_set = CIFAR10(train=False, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(test_set.data, np.array(test_set.targets))

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def example_to_image(self, example):
        return Image.fromarray(example)

    def non_iid_example_to_image(self, example):
        #has torch.tensor
        if torch.is_tensor(example):
            example = example.numpy()
        return Image.fromarray(example)


DataLoader = base.DataLoader

