import os
import random
import torch
import numpy as np
from torch import optim
import torch.nn as nn
import math

def set_seed(manualSeed=3):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

def make_optimizer(optimizer, lr, wd, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if optimizer == 'sgd':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0,
                'lr': lr,
                'weight_decay': wd
                # 'weight_decay': 1e-4
        }
    elif optimizer == 'sgd_momentum':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9,
                'lr': lr,
                'weight_decay': wd
                # 'weight_decay': 1e-4
        }
    elif optimizer == 'adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': lr,
            'weight_decay': wd
        }
    elif optimizer == 'rmsprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'alpha': 0.99,
            'eps': 1e-08,
            'lr': lr,
            'weight_decay': wd
        }
    
    return optimizer_function(trainable, **kwargs)
