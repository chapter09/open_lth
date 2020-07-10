# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

<<<<<<< HEAD
from ..platforms import local
=======
from open_lth.platforms import local
>>>>>>> a12cc5228f2e75633f99017c1881525bf527f5e6


registered_platforms = {'local': local.Platform}


def get(name):
    return registered_platforms[name]
