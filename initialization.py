from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pkgutil

import torch
import torch.nn as nn
import torch.optim as optim

# Put imports for pretrained models here
import knlp.models
import knlp.hooks
import knlp.losses

from knlp.registry import (
    MODELS,
    LOSSES,
    OPTIMIZERS,
    SCHEDULERS,
    DATASETS,
    HOOKS
)

def register_torch_modules():
    # register models
    for name, cls in knlp.models.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # register losses
    losses = [
        nn.CrossEntropyLoss,
        nn.BCELoss,
        nn.BCEWithLogitsLoss,
        #knlp.losses.CustomLoss1,
    ]

    for loss in losses:
        LOSSES.register(loss)

    # register optimizers
    optimizers = [
        optim.Adadelta,
        optim.Adagrad,
        optim.Adam,
        optim.AdamW,
        optim.SparseAdam,
        optim.Adamax,
        optim.ASGD,
        optim.LBFGS,
        optim.RMSprop,
        optim.Rprop,
        optim.SGD,
    ]
    for optimizer in optimizers:
        OPTIMIZERS.register(optimizer)

    # register schedulers
    schedulers = [
            optim.lr_scheduler.StepLR,
            optim.lr_scheduler.MultiStepLR,
            optim.lr_scheduler.ExponentialLR,
            optim.lr_scheduler.CosineAnnealingLR,
            optim.lr_scheduler.ReduceLROnPlateau,
            optim.lr_scheduler.CyclicLR,
            optim.lr_scheduler.OneCycleLR,
    ]
    for scheduler in schedulers:
        SCHEDULERS.register(scheduler)


def register_default_hooks():
    HOOKS.register(knlp.hooks.DefaultLossHook)
    HOOKS.register(knlp.hooks.DefaultForwardHook)
    HOOKS.register(knlp.hooks.DefaultPostForwardHook)
    HOOKS.register(knlp.hooks.DefaultMetricHook)
    HOOKS.register(knlp.hooks.DefaultModelBuilderHook)
    HOOKS.register(knlp.hooks.DefaultLoggerHook)
    HOOKS.register(knlp.hooks.DefaultWriteResultHook)


def initialize():
    register_torch_modules()
    register_default_hooks()
