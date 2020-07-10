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

from ..datasets import base
from ..platforms.platform import get_platform


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
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_fl_train_examples(): return 500

    @staticmethod
    def num_fl_test_examples(): return 100

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])
    
    
    @classmethod
    def get_non_iid_train_set(cls, use_augmentation, bias_fraction, fl_test):

        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10_non_iid'), download=True)
        
        labels = list(train_set.classes)

        #create dict {label: [data]}
        grouped_data = {label: []
                        for label in labels}
        for dataitem in train_set:
            #tuple (<PIL.Image.Image image mode=L size=28x28 at 0x7FC0EA7A03D0>, 8)
            _, label = dataitem
            label = labels[label]
            grouped_data[label].append(dataitem)
        trainset = grouped_data

        #total data number:
        if fl_test:
            total_size = cls.num_fl_train_examples()
        else:
            total_size = cls.num_train_examples()
        
        if bias_fraction is None:
            dist = Dataset.uniform(total_size, cls.num_classes())

        else:
            # 1. 80% for one label, 20% for the rest
            majority = int(total_size * bias_fraction)
            minority = total_size - majority
            #distributed among all minor labels
            dist = Dataset.uniform(minority, (len(labels)-1))
            #randomly choose 1 label and get 480 sample
            pref = int(10*random.uniform(0, 1))
            dist.insert(pref, majority)

        #list of data with label
        partition = []
        used = {}
        for i, label in enumerate(labels):
            partition.extend(Dataset.extract(used, trainset, label, labels, dist[i]))
        np.random.shuffle(partition)

        data = []
        targets = []

        for d,t in partition:
            data.append(np.asarray(d))
            targets.append(t)

        return Dataset(np.asarray(data), np.asarray(targets), augment if use_augmentation else [])
  

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

