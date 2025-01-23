import argparse
import copy
import os

import pandas as pd
import torch
import tqdm

import wandb


def parse_file_name(file_name):
    # split the file name
    parts = os.path.splitext(file_name)[0].split('_')
    info = {}
    # get the model name
    info["run_id"] = parts[0]
    # get the epoch number
    info["epoch"] = parts[1]
    if len(parts) > 2:
        info["acc"] = parts[2]
    return info

def generate_metadata_from_folder(args):
    # loop for all files in the folder
    checkpoint_dir = os.path.join(args.model_dir, args.dataset)
    files = sorted(list(file for file in os.listdir(checkpoint_dir) if file.endswith('.pt')))

    # get metadata
    runs = wandb.Api().runs(f"{args.entity}/{args.project_name}")
    runs = {run.id: run for run in runs}

    model_records = []
    for i, file in tqdm.tqdm(enumerate(files), total=len(files)):
        # skip embedding file
        if file == "embedding.pt": continue
        # parse file name
        info = parse_file_name(file)
        run_id = info["run_id"]
        epoch = info["epoch"]
        # get metadata from config of wandb run with run_id
        try:
            run = runs[run_id]
        except KeyError:
            print(f"WARNING: Run {run_id} not found in wandb")
            # remove the file if the run is not found
            os.remove(os.path.join(checkpoint_dir, file))
            continue
        config = copy.deepcopy(run.config)
        config.pop('wandb_project_name', None)
        config.pop('wandb_entity', None)

        if 'best' in epoch:
            acc = run.summary['best_test_top1_accuracy']
            acc = str(int(acc * 10000))
            os.rename(os.path.join(checkpoint_dir, file), os.path.join(checkpoint_dir, f"{run_id}_{epoch}_{acc}.pt"))
            file = os.path.join(checkpoint_dir, f"{run_id}_{epoch}_{acc}.pt")
        else:
            try:
                acc = info['acc']
            except KeyError:
                print(f"WARNING: Run {run_id} does not have test_top1_accuracy in history")
                continue
            except IndexError:
                print(f"WARNING: Run {run_id} does not have epoch {epoch} in history. Value {run.history(keys=['test_top1_accuracy'], pandas=False)}")
                continue

        config['ckpt_epoch'] = epoch
        config['ckpt_file'] = os.path.join(args.dataset, file)
        config["test_top1_accuracy"] = int(acc) / 10000

        model_records.append(config)

    df = pd.DataFrame(model_records)
    # remove duplicated rows
    subset_col = df.columns.difference(["ckpt_epoch", "ckpt_file"])
    df = df.drop_duplicates(subset=subset_col, keep='last')
    print(len(df))
    save_path = os.path.join(args.model_dir, "metadata", f'{args.dataset}.csv')
    if not os.path.exists(os.path.join(args.model_dir, "metadata")):
        os.makedirs(os.path.join(args.model_dir, "metadata"))
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    # command line arguments
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model/')
    parser.add_argument('--entity', type=str, default='wandb')
    parser.add_argument('--project_name', type=str, default='name')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'fashionmnist', 'svhn'], default='mnist')
    args = parser.parse_args()
    generate_metadata_from_folder(args)
