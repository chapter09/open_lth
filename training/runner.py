# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from dataclasses import dataclass

from ..cli import shared_args
from ..foundations.runner import Runner
from .. import models
from ..platforms.platform import get_platform
from ..training import train
from ..training.desc import TrainingDesc


@dataclass
class TrainingRunner(Runner):
    replicate: int
    global_model_path: str
    desc: TrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(args.replicate, args.global_model_path, TrainingDesc.create_from_args(args),
                              not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate))

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print('='*82 + f'\nTraining a Model (Replicate {self.replicate})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.replicate)}' + '\n' + '='*82 + '\n')
        self.desc.save(self.desc.run_path(self.replicate))

        new_model = models.registry.get(self.desc.model_hparams)
        if self.global_model_path is not None:
            #on client, load the global model to start
            new_model.load_state_dict(torch.load(self.global_model_path))
        train.standard_train(
            new_model,self.desc.run_path(self.replicate),
            self.desc.dataset_hparams, self.desc.training_hparams, evaluate_every_epoch=self.evaluate_every_epoch)
