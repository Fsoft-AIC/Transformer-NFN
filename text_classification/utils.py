import os
import random
import torch
import numpy as np
from torch import optim
import torch.nn as nn

def set_seed(manualSeed=3):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    wd_term = args.l2_regularize

    if args.optimizer == 'sgd':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0,
                'lr': args.lr,
                'weight_decay': wd_term
                # 'weight_decay': 1e-4
        }
    elif args.optimizer == 'sgd_momentum':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9,
                'lr': args.lr,
                'weight_decay': wd_term
                # 'weight_decay': 1e-4
        }
    elif args.optimizer == 'adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term
        }
    elif args.optimizer == 'rmsprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'alpha': 0.99,
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term
        }
    
    return optimizer_function(trainable, **kwargs)

def init_weights(std):
    def inner(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return inner