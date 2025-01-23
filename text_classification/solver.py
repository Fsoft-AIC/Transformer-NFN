import os
import torch
import torch.nn as nn
from torch import optim
import wandb
from data_loader import get_loader
from model import TextClassificationTransformer
from sklearn.metrics import accuracy_score
from utils import make_optimizer, init_weights
from tqdm import tqdm


class Solver(object):
    def __init__(self, args):
        self.args = args
        self.wandb_id = wandb.run.id

        # Get data loaders
        self.train_loader, self.test_loader = get_loader(args)

        # Create Transformer object
        self.model = TextClassificationTransformer(embed_dim=self.args.embed_dim,
                                                    n_layers=self.args.n_layers,
                                                    n_attention_heads=self.args.n_attention_heads,
                                                    forward_mul=self.args.forward_mul,
                                                    max_len=self.args.max_len,
                                                    n_classes=self.args.n_classes,
                                                    dropout=self.args.dropout)
        
        self.model.apply(init_weights(std=args.w_init_std))
        
        if not os.path.exists(os.path.join(self.args.model_path, "embedding.pt")):
            embedding_state_dict = self.model.state_dict()
            for key in list(embedding_state_dict.keys()):
                if not key.startswith("embedding."):
                    del embedding_state_dict[key]
            torch.save(embedding_state_dict, os.path.join(self.args.model_path, "embedding.pt"))
        
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters is: {pytorch_total_params}")

        # Push to GPU
        if self.args.is_cuda:
            self.model = self.model.cuda()

        # Display Vision Transformer
        print('--------Network--------')
        print(self.model)       

        # Training loss function
        self.loss_fn = nn.CrossEntropyLoss()


    def test_dataset(self, loader):
        # Set Vision Transformer to evaluation mode
        self.model.eval()

        # Arrays to record all labels and logits
        all_labels = []
        all_logits = []

        # Testing loop
        for (x, y) in loader:
            if self.args.is_cuda:
                x = x.cuda()

            # Avoid capturing gradients in evaluation time for faster speed
            with torch.no_grad():
                logits = self.model(x)

            all_labels.append(y)
            all_logits.append(logits.cpu())

        # Convert all captured variables to torch
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        all_pred   = all_logits.max(1)[1]
        
        # Compute loss, accuracy and confusion matrix
        acc  = accuracy_score(y_true=all_labels, y_pred=all_pred)

        return acc

    def test(self):
        acc = self.test_dataset(self.test_loader)
        return acc

    def get_model_state_dict(self):
        model_state_dict = self.model.state_dict()
        # remove EmbedLayerText from the state dict
        for key in list(model_state_dict.keys()):
            if key.startswith("embedding."):
                del model_state_dict[key]
        return model_state_dict

    def train(self):
        # Define optimizer for training the model
        optimizer = make_optimizer(args=self.args, model=self.model)

        # scheduler for linear warmup of lr and then cosine decay to 1e-5
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/self.args.warmup_epochs, end_factor=1.0, total_iters=self.args.warmup_epochs-1, last_epoch=-1, verbose=True)
        cos_decay     = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs-self.args.warmup_epochs, eta_min=1e-5, verbose=True)

        # Variable to capture best test accuracy
        best_acc = 0

        # Training loop
        for epoch in range(self.args.epochs):

            # Set model to training mode
            self.model.train()

            # Loop on loader
            for i, (x, y) in enumerate(self.train_loader):

                # Push to GPU
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()

                # Get output logits from the model 
                logits = self.model(x)

                # Compute training loss
                loss = self.loss_fn(logits, y)

                # Updating the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Test the test set after every epoch
            test_acc = self.test() # Test training set every 25 epochs


            # Capture best test accuracy
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.get_model_state_dict(), os.path.join(self.args.model_path, f"{self.wandb_id}_best.pt"))
            # print(f"Best test acc: {best_acc:.2%}\n")

            # Save model
            if (epoch + 1) in [1, 50, 75, 100]:
                torch.save(self.get_model_state_dict(), os.path.join(self.args.model_path, f"{self.wandb_id}_{epoch+1}_{int(test_acc*10000)}.pt"))
            
            # Update learning rate using schedulers
            if epoch < self.args.warmup_epochs:
                linear_warmup.step()
            else:
                cos_decay.step()
        wandb.log({"best_test_accuracy": best_acc})
            