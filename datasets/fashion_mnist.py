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

from torchvision import datasets
from ..datasets import base
from ..platforms.platform import get_platform


class Dataset(base.ImageDataset):

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000


    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        train_set = datasets.FashionMNIST(train=True, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
        return Dataset(train_set.data, train_set.targets)
    

    @staticmethod
    def get_test_set():
        test_set = datasets.FashionMNIST(train=False, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
        return Dataset(test_set.data, test_set.targets)

    def __init__(self,  examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')



DataLoader = base.DataLoader

