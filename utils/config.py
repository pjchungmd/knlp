"""
Modified from: https://github.com/pudae/kaggle-understanding-clouds/blob/master/kvt/utils/config.py:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sacred import Experiment

ex = Experiment('knlp')

@ex.config
def default_config():
    # configuration for model
    backbone = {
        'name': 'roberta',
        'params': {
            'pretrained': True
        }
    }

    # configuration for loss 
    loss = {
        'name': 'CrossEntropyLoss',
        'params': {}
    }

    # configuration for optimizer
    optimizer = {
        'name': 'AdamW',
        'params': {
            'lr': 1e-4,
            'weight_decay': 0.001
        }
    }

    # configuration for scheduler
    scheduler = {
        'name': 'none',
        'params': {}
    }

    # configuration for training
    train = {
        'dir': 'train_logs/base',
        'batch_size': 32,
        'log_step': 2,
        'gradient_accumulation_step': None,
        'num_epochs': 3,
        'save_checkpoint_epoch': 1,
        'num_keep_checkpoint': 1,
    }

    # configuration for evaluation
    evaluation = {
        'batch_size': 64
    }
