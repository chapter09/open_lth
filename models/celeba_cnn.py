# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from ..foundations import hparams
from ..lottery.desc import LotteryDesc
from ..models import base
from ..pruning import sparse_global

#image size is 218x178 resize to 

class Model(base.Model):
    """A cnn neural network designed for CelebA."""

    def __init__(self, plan, initializer, outputs):
        super(Model, self).__init__()
        
        #equivalent of tf sigmoid_cross_entropy_with_logits
        self.criterion = nn.BCEWithLogitsLoss()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.ReLU()
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.ReLU()
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.ReLU()
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
                nn.Linear(64*4*4, 512),
                nn.ReLU(),   
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
        )

        self.apply(initializer)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        
        logits = self.classifier(x.view(-1, 64*4*4))
        
        return F.softmax(logits, dim=1)

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('celeba'))

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs 

        plan = None

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='celeba_cnn',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='celeba',
            #need to change the params
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            lr=0.1,
            training_steps='160ep'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight'
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
