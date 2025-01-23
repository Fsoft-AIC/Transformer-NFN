import argparse

import numpy as np
import torch
import torch.nn as nn
from common.utils import make_optimizer, set_seed
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model import InvariantNFN
from nfn_transformer.common.data_utils import SmallTransformerZooDataset
from nfn_transformer.common.weight_space import (
    AttentionWeightSpaceFeatures,
    LinearWeightSpaceFeatures,
    attention_network_spec_from_wsfeat,
    network_spec_from_wsfeat,
)


@torch.no_grad()
def evaluate(nfn_net, test_loader, loss_fn):
    nfn_net.eval()
    losses,err = [],[]
    pred,actual = [],[]
    for batch in tqdm(test_loader):
        # Move batch to cuda
        embedding, classifier, encoder = batch["embedding"], batch["classifier"], batch["encoder"]
        classifier, encoder = LinearWeightSpaceFeatures(classifier['weight'], classifier['bias']).to("cuda"), AttentionWeightSpaceFeatures(**encoder).to("cuda")
        true_acc = batch["accuracy"].cuda()

        # Forward step
        pred_acc = nfn_net(embedding, classifier, encoder)
        # Calculate loss
        err.append(torch.abs(pred_acc - true_acc).mean().item())
        loss = loss_fn(pred_acc, true_acc).item()
        losses.append(loss)
        pred.append(pred_acc.detach().cpu().numpy())
        actual.append(true_acc.cpu().numpy())
    avg_err, avg_loss = np.mean(err), np.mean(losses)
    actual, pred = np.concatenate(actual), np.concatenate(pred)
    rsq = r2_score(actual, pred)
    tau = kendalltau(actual, pred).correlation
    return avg_err, avg_loss, rsq, tau

def main(args):
    print("Start to load dataset")
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    train_set = SmallTransformerZooDataset(data_path=args.data_path, n_heads=args.n_heads, split="train", name = args.dataset, cut_off=args.cut_off, load_cache= args.load_cache)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, collate_fn=train_set.collate_fn)

    val_set = SmallTransformerZooDataset(data_path=args.data_path, n_heads=args.n_heads, split="val", name = args.dataset, cut_off=args.cut_off, load_cache= args.load_cache)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, collate_fn=train_set.collate_fn)

    test_set = SmallTransformerZooDataset(data_path=args.data_path, n_heads=args.n_heads, split="test", name = args.dataset, cut_off=args.cut_off, load_cache= args.load_cache)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, collate_fn=test_set.collate_fn)

    next_train_load = next(iter(train_loader))
    weight_classify = next_train_load['classifier']['weight']
    bias_classify = next_train_load['classifier']['bias']
    classifier_network_spec = network_spec_from_wsfeat(LinearWeightSpaceFeatures(weight_classify, bias_classify).to("cpu"), set_all_dims=True)

    weight_embedding = next_train_load['embedding']['weight']
    bias_embedding = next_train_load['embedding']['bias']
    embedding_network_spec = network_spec_from_wsfeat(LinearWeightSpaceFeatures(weight_embedding, bias_embedding).to("cpu"), set_all_dims=True)

    weight_encoder = next_train_load['encoder']

    encoder_network_spec = attention_network_spec_from_wsfeat(AttentionWeightSpaceFeatures(**weight_encoder).to("cpu"), set_all_dims=True)
    nfn_net = (InvariantNFN(embedding_network_spec=embedding_network_spec, classifier_network_spec=classifier_network_spec,
                            encoder_network_spec=encoder_network_spec,
                            classifier_nfn_channels=args.classifier_nfn_channels,
                            transformers_nfn_channels=args.transformers_nfn_channels,
                            num_out_classify=args.num_out_classify, num_out_embedding=args.num_out_embedding,
                            num_out_encoder=args.num_out_encoder, init_type=args.init_type,
                            enc_mode=args.enc_mode, cls_mode=args.cls_mode, emb_mode=args.emb_mode,
    ))
    print(nfn_net)
    num_params = sum(p.numel() for p in nfn_net.parameters() if p.requires_grad)
    print(f"Total params in NFN: {num_params}.")
    wandb.log({"model/total_parameters": num_params})

    nfn_net.cuda()
    optimizer = make_optimizer(optimizer=args.optimizer, lr=args.lr, wd=args.wd, model=nfn_net)

    # scheduler for linear warmup of lr and then cosine decay to 1e-5
    linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/args.warmup_epochs, end_factor=1.0, total_iters=args.warmup_epochs-1, last_epoch=-1, verbose=True)
    cos_decay     = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs-args.warmup_epochs, eta_min=1e-5, verbose=True)


    loss_fn = nn.BCELoss()
    best_rsq, best_tau = -float('inf'), -float('inf')

    for epoch in tqdm(range(args.epochs)):
        nfn_net.train()
        for batch in tqdm(train_loader):
            # Move batch to cuda
            embedding, classifier, encoder = batch["embedding"], batch["classifier"], batch["encoder"]
            optimizer.zero_grad()
            classifier, encoder = LinearWeightSpaceFeatures(classifier['weight'], classifier['bias']).to("cuda"), AttentionWeightSpaceFeatures(**encoder).to("cuda")
            true_acc = batch["accuracy"].cuda()
            # Forward step
            pred_acc = nfn_net(embedding, classifier, encoder)
            # Calculate loss
            try:
                loss = loss_fn(pred_acc, true_acc)  # NOTE: Placeholder
            except:
                print(pred_acc)
                print(true_acc)
            # Update the model
            loss.backward()
            optimizer.step()
        theoretical_loss = loss_fn(true_acc, true_acc)  # perfect loss
        wandb.log({
            "train/loss": loss.detach().cpu().item(),
            "train/rsq": r2_score(true_acc.cpu().numpy(), pred_acc.detach().cpu().numpy()),
            "train/theoretical_loss": theoretical_loss.detach().cpu().item(),
        })
        pass
        # train_err, train_loss, train_rsq, train_tau = evaluate(nfn_net, train_loader, loss_fn)
        # print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Tau: {train_tau}, Train err: {train_err}, Train rsq: {train_rsq}")
        val_err, val_loss, val_rsq, val_tau = evaluate(nfn_net, val_loader, loss_fn)
        print(f"Epoch: {epoch}, Val Loss: {val_loss}, Val Tau: {val_tau}, Val err: {val_err}, Val rsq: {val_rsq}")
        wandb.log({
            "val/l1_err": val_err,
            "val/loss": val_loss,
            "val/rsq": val_rsq,
            "val/kendall_tau": val_tau,
        })
        test_err, test_loss, test_rsq, test_tau = evaluate(nfn_net, test_loader, loss_fn)
        print(f"Epoch: {epoch}, Test Loss: {test_loss}, Test Tau: {test_tau}, Test err: {test_err}, Test rsq: {test_rsq}")
        wandb.log({
            "test/l1_err": test_err,
            "test/loss": test_loss,
            "test/rsq": test_rsq,
            "test/kendall_tau": test_tau,
        })

        # Update learning rate using schedulers
        if epoch < args.warmup_epochs:
            linear_warmup.step()
        else:
            cos_decay.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer nfn training from scratch')

    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--device', type=str, default=None)

    # Wandb arguments
    parser.add_argument('--wandb', type=str, default="True", help="Log run on wandb or not")
    parser.add_argument('--project', type=str, default=None, help='wandb project name')
    parser.add_argument('--entity', type=str, default=None, help='wandb entity name')

    # Model Arguments
    parser.add_argument("--model", type=str, default="transformer_nfn")
    parser.add_argument("--ws_dim", type=int, default=10, help="The number weight in the nfn network stacked on each other")
    parser.add_argument("--classifier_dim", type=int, default=1024, help="Dimension of the classification network")


    # Training Arguments
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd", "sgd_momentum", "rmsprop"] ,help='choice of optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='peak learning rate')
    parser.add_argument('--wd', type=float, default=0, help="Weight decay factor")
    parser.add_argument('--init_type', type=str, default="xavier_normal", choices=["pytorch_default", "kaiming_normal", "xavier_normal", "xavier_uniform","uniform"], help='Init mode of network parameters')

    # Model Arguments
    parser.add_argument("--classifier_nfn_channels", type=lambda value: [int(x) for x in value.split(',')], default=[10, 10], help="Channels for classifier NFN")
    parser.add_argument("--transformers_nfn_channels", type=lambda value: [int(x) for x in value.split(',')], default=[10], help="Channels for transformers NFN")
    parser.add_argument("--num_out_classify", type=int, default=10, help="Number of output classes for classifier")
    parser.add_argument("--num_out_embedding", type=int, default=10, help="Number of output classes for embedding")
    parser.add_argument("--num_out_encoder", type=int, default=10, help="Number of output classes for encoder")

    parser.add_argument('--enc_mode', default='inv', choices=['no', 'inv', 'statnn', 'mlp'])
    parser.add_argument('--cls_mode', default='mlp', choices=['no', 'hnps', 'statnn', 'mlp'])
    parser.add_argument('--emb_mode', default='mlp', choices=['no', 'mlp'])

    # Data arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist',"copy", "ag_news"], help='dataset to use')
    parser.add_argument('--cut_off', type=float, default=0.1, help='cut off rate for accuracy')
    parser.add_argument('--load_cache', type=bool, default=True, help='load cache')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of heads in transformer input network')
    parser.add_argument('--data_path', type=str, default='data/mnist', help='path to dataset')

    args = parser.parse_args()
    set_seed(manualSeed=args.seed)

    assert any(mode != 'no' for mode in [args.enc_mode, args.cls_mode, args.emb_mode])

    #Init wandb
    ablation = f'{args.enc_mode}_enc_{args.cls_mode}_cls_{args.emb_mode}_emb'
    name_wandb = f"dataset_{args.dataset}-model_{args.model}-ablation_{ablation}-ws_dim_{args.ws_dim}-optim_{args.optimizer}-seed_{args.seed}-lr_{args.lr}-cut_off_{args.cut_off}"

    if wandb.run is None:
        # Not in sweep mode, initialize wandb manually
        if args.wandb == "False":
            wandb.init(project=args.project, entity=args.entity, config={}, name=name_wandb, mode="disabled")
        else:
            wandb.init(project=args.project, entity=args.entity, config={}, name=name_wandb, mode="online")

    wandb.config.update(args)
    wandb.save("nfn_transformer/common/*.py")
    wandb.save("nfn_transformer/layers/*.py")
    wandb.save("nfn_transformer/main.py")
    wandb.save("nfn_transformer/model.py")
    main(args)
