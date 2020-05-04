from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from knlp.utils import Registry


MODELS = Registry('models')
LOSSES = Registry('losses')
OPTIMIZERS = Registry('optimizer')
SCHEDULERS = Registry('scheduler')

DATASETS = Registry('dataset')
#TRANSFORMS = Registry('transform')
HOOKS = Registry('hook')
