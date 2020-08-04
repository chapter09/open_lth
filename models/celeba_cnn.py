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

        self.apply(initializer)

    def forward(self, x):
        x = self.layers(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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

        #output is a binary, how to express 
        outputs = outputs 

        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

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
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight'
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
