import argparse
import os

import torch
from solver import Solver
from utils import set_seed

import wandb


def main(args):
    # Create required directories if they don't exist
    os.makedirs(args.model_path,  exist_ok=True)

    solver = Solver(args)
    solver.train()               # Training function


# Update arguments
def update_args(args):
    args.model_path  = os.path.join(args.model_path, args.dataset)
    args.is_cuda     = torch.cuda.is_available()  # Check GPU availability

    if args.is_cuda:
        print("Using GPU")
    else:
        print("Cuda not available.")

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Vision Transformer from Scratch')

    # Training Arguments
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes in the dataset')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd", "sgd_momentum", "rmsprop"] ,help='choice of optimizer')
    parser.add_argument('--w_init_std', type=float, default=0.1, help="standard deviation of weight init")
    parser.add_argument('--lr', type=float, default=5e-4, help='peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--l2_regularize', type=float, default=1e-6, help="L2 weight regulization factor")

    # Data arguments
    parser.add_argument('--dataset', type=str, default='ag_news', choices=['ag_news'], help='dataset to use')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store downloaded dataset')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='fraction of training dataset used')

    # ViT Arguments
    parser.add_argument('--embed_dim', type=int, default=32, help="Embedding dimension")
    parser.add_argument('--n_layers', type=int, default=2, help="Number of layers in the model")
    parser.add_argument('--n_attention_heads', type=int, default=2, help="Number of attention heads")
    parser.add_argument('--forward_mul', type=int, default=2, help="Multiplication factor for the forward layer size")
    parser.add_argument('--max_len', type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout value')
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--wandb_project_name", type=str, default="", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="wandb entity")


    args = parser.parse_args()
    args = update_args(args)
    set_seed(args.seed)

    # wandb initialization
    run_name = f"opt_{args.optimizer}_lr_{args.lr}_dropout_{args.dropout}_wstd_{args.w_init_std}_tf_{args.train_fraction}_l2_{args.l2_regularize}"
    wandb.init(project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name)
    wandb.config.update(args)

    main(args)
