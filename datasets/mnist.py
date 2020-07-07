# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import random

from PIL import Image
import torchvision
import torch

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples(client_num): return 60000 / client_num

    @staticmethod
    def num_test_examples(client_num): return 10000 / client_num

    @staticmethod
    def num_classes(): return 10

    @classmethod
    def get_train_set(cls, use_augmentation, client_num, bias_fraction):
        # No augmentation for MNIST.
        train_set = torchvision.datasets.MNIST(
            train=True, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        if bias_fraction:
            data, targets = Dataset.get_non_iid_set(train_set, client_num, bias_fraction)
        else:
            data = train_set.data
            targets = train_set.targets

        return cls(np.array(data), np.array(targets))

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.MNIST(
            train=False, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        return Dataset(test_set.data, test_set.targets)


    def __init__(self,  examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')
    
    def non_iid_example_to_image(self, example):
        #has torch.tensor
        if torch.is_tensor(example):
            example = example.numpy()
        return Image.fromarray(example, mode='L')



DataLoader = base.DataLoader
