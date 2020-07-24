# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
import os

from ..datasets import registry as datasets_registry
from ..foundations import desc
from ..foundations import hparams
from ..foundations.step import Step
from ..lottery.desc import LotteryDesc
from ..platforms.platform import get_platform


@dataclass
class TrainingDesc(desc.Desc):
    """The hyperparameters necessary to describe a training run."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    client_hparams: hparams.ClientHparams
    data_saved_folder =  None

    @staticmethod
    def name_prefix(): return 'train'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: LotteryDesc = None):
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        hparams.ClientHparams.add_args(parser, defaults=defaults.client_hparams if defaults else None)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        client_hparams = hparams.ClientHparams.create_from_args(args)
        return TrainingDesc(model_hparams, dataset_hparams, training_hparams, client_hparams)

    @property
    def end_step(self):
        iterations_per_epoch = datasets_registry.iterations_per_epoch(self.dataset_hparams)
        return Step.from_str(self.training_hparams.training_steps, iterations_per_epoch)

    @property
    def train_outputs(self):
        datasets_registry.num_classes(self.dataset_hparams)

    def run_path(self, replicate, experiment='main'):
        self.data_saved_folder = os.path.join(get_platform().root, self.current_time,
                                    str(self.client_hparams.round_num),
                                    str(self.client_hparams.client_id), self.hashname)
        return os.path.join(self.data_saved_folder, f'replicate_{replicate}', experiment)

    @property
    def display(self):
        return '\n'.join([self.dataset_hparams.display, self.model_hparams.display, 
                            self.training_hparams.display, self.client_hparams.display])
