# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

from ..platforms import base


class Platform(base.Platform):
    
    _root: str = '/mnt/open_lth_data'

    @property
    def base(self):
        return '/mnt/open_lth_data'

    @property
    def root(self):        
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def dataset_root(self):
        return '/mnt/open_lth_datasets'

    @property
    def imagenet_root(self):
        raise NotImplementedError
    



