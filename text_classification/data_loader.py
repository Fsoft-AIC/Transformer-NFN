import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm
import numpy as np
from AGNewsTokenizedDataset import AGNewsTokenizedDataset


def get_loader(args):
    train = AGNewsTokenizedDataset(split='train', fraction=args.train_fraction)
    test = AGNewsTokenizedDataset(split='test')

    # Define a collate function for padding
    def collate_fn(batch):
        token_id_list, labels = zip(*batch)

        # Pad sequences to the same length
        padded_token_ids = pad_sequence(token_id_list, padding_value=0)

        # , torch.tensor(lengths)
        return padded_token_ids.T, torch.tensor(labels)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.n_workers,
                                                drop_last=True,
                                                collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.n_workers,
                                                drop_last=False,
                                                collate_fn=collate_fn)
   
    return train_loader, test_loader
