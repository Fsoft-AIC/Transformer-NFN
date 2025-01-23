import argparse
import os  # Import os to manage file paths

import lightgbm as lgb
import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.metrics import r2_score

import wandb
from nfn_transformer.common.data_utils import SmallTransformerZooDataset
from nfn_transformer.common.weight_space import (
    AttentionWeightSpaceFeatures,
    LinearWeightSpaceFeatures,
)


def flatten_input(embedding, classifier, encoder):
    """
    Flattens the embedding, classifier, and encoder into a 1D array for LightGBM.
    """
    classifier_features = []
    for weight, bias in classifier:
        classifier_features.append(np.array(weight).flatten())  # Convert to numpy and flatten
        classifier_features.append(np.array(bias).flatten())    # Convert to numpy and flatten
    classifier_flattened = np.concatenate(classifier_features)  # Concatenate all flattened classifier features

    encoder_features = []
    for W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B in encoder:
        encoder_features.extend([
            np.array(W_q).flatten(),
            np.array(W_k).flatten(),
            np.array(W_v).flatten(),
            np.array(W_o).flatten(),
            np.array(W_A).flatten(),
            np.array(W_B).flatten(),
            np.array(b_A).flatten(),
            np.array(b_B).flatten()
        ])
    encoder_flattened = np.concatenate(encoder_features)  # Concatenate all flattened encoder features
    if embedding != None:
        embedding_weight = np.concatenate([np.array(w).flatten() for w in embedding['weight']])
        embedding_bias = np.concatenate([np.array(b).flatten() for b in embedding['bias']])

        # Concatenate all features into a single 1D array
        features = np.concatenate([embedding_weight, embedding_bias, classifier_flattened, encoder_flattened])
    else:
        features = np.concatenate([classifier_flattened, encoder_flattened])

    return features


def prepare_data(dataset, data_name='ag_news'):
    """
    Prepares the data for LightGBM by flattening all input tensors without using a DataLoader.
    """
    X, y = [], []

    for i in range(len(dataset)):
        sample = dataset[i]

        embedding = sample["embedding"]
        classifier = LinearWeightSpaceFeatures(sample["classifier"]['weight'], sample["classifier"]['bias']).to("cpu")
        encoder = AttentionWeightSpaceFeatures(**sample["encoder"]).to("cpu")
        true_acc = sample["accuracy"]

        # Flatten and append
        if data_name == 'ag_news':
            X.append(flatten_input(None, classifier, encoder))
        else:
            X.append(flatten_input(embedding, classifier, encoder))

        y.append(true_acc[0])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y


def train_lgb(X_train, y_train, X_val, y_val, X_test, y_test, name_wandb, mode='rf', seed=3):
    """
    Trains a LightGBM model using the prepared dataset with early stopping using callbacks in lgb.train.
    """
    # Set up the LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Set LightGBM parameters
    if mode == 'gbdt':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 256,
            'max_depth': 10,
            'min_data_in_leaf': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'min_gain_to_split': 0.01,
            'max_bin': 255,
            'verbose': -1,
            'seed': seed,
            'device': 'gpu'
        }
    else:
        assert mode == 'rf'
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'rf',
            'learning_rate': 0.1,
            'num_leaves': 256,
            'max_depth': 10,
            'min_data_in_leaf': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'min_gain_to_split': 0.01,
            'max_bin': 255,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'feature_fraction': 0.8,
            'verbose': -1,
            'seed': seed,
            'device': 'gpu'
        }

    def log_metrics(env):
        """
        Custom callback to log train, validation, and test r2 and tau at each iteration.
        """
        y_pred_train = env.model.predict(X_train)
        y_pred_val = env.model.predict(X_val)
        y_pred_test = env.model.predict(X_test)

        # Calculate R2 and Kendall Tau for training set
        train_r2 = r2_score(y_train, y_pred_train)
        train_tau, _ = kendalltau(y_train, y_pred_train)

        # Calculate R2 and Kendall Tau for validation set
        val_r2 = r2_score(y_val, y_pred_val)
        val_tau, _ = kendalltau(y_val, y_pred_val)

        # Calculate R2 and Kendall Tau for test set
        test_r2 = r2_score(y_test, y_pred_test)
        test_tau, _ = kendalltau(y_test, y_pred_test)

        # Log metrics to console and wandb
        print(f"Iteration {env.iteration}: Train R2 = {train_r2:.4f}, Train Tau = {train_tau:.4f}, "
              f"Val R2 = {val_r2:.4f}, Val Tau = {val_tau:.4f}, Test R2 = {test_r2:.4f}, Test Tau = {test_tau:.4f}")

        wandb.log({
            "train/r2": train_r2,
            "train/tau": train_tau,
            "val/r2": val_r2,
            "val/tau": val_tau,
            "test/r2": test_r2,
            "test/tau": test_tau,
            "iteration": env.iteration
        })

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=10),
            log_metrics
        ]
    )

    # Save the model
    # model_path = f"data/{name_wandb}.txt"
    # gbm.save_model(model_path)
    # print(f"Model saved to {model_path}")

    return gbm


def main(args):
    print("Start to load dataset")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the entire dataset directly
    train_set = SmallTransformerZooDataset(data_path=args.data_path, n_heads=args.n_heads, split="train", name=args.dataset, cut_off=args.cut_off)
    val_set = SmallTransformerZooDataset(data_path=args.data_path, n_heads=args.n_heads, split="val", name=args.dataset, cut_off=args.cut_off)
    test_set = SmallTransformerZooDataset(data_path=args.data_path, n_heads=args.n_heads, split="test", name=args.dataset, cut_off=args.cut_off)

    # Prepare data for LightGBM
    print("Preparing data for LightGBM...")
    X_train, y_train = prepare_data(train_set, args.dataset)
    X_val, y_val = prepare_data(val_set, args.dataset)
    X_test, y_test = prepare_data(test_set, args.dataset)

    # Train LightGBM model
    print("Training LightGBM model...")
    lgb_model = train_lgb(X_train, y_train, X_val, y_val, X_test, y_test, name_wandb=args.name_wandb, mode=args.model, seed=args.seed)

    # y_pred = lgb_model.predict(X_test)
    # results = np.array([y_pred, y_test])
    # output_file = f'{args.model}_{args.dataset}.npy'
    # np.save(output_file, results)
    # print(f"Results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightGBM training from Transformer data')

    # Training Arguments
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--model', type=str, default="rf", choices=['gbdt', "rf"], help="model type")

    # Data arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', "copy", "ag_news"], help='dataset to use')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of heads in transformer input network')
    parser.add_argument('--data_path', type=str, default='./mnist_transformer', help='path to dataset')
    parser.add_argument('--cut_off', type=float, default=0.1, help='cut off rate for accuracy')

    # Wandb arguments
    parser.add_argument('--wandb', type=str, default="True", help="Log run on wandb or not")
    parser.add_argument('--project', type=str, default=None, help='wandb project name')
    parser.add_argument('--entity', type=str, default=None, help='wandb entity name')

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    name_wandb = f"dataset_{args.dataset}-model_{args.model}--seed_{args.seed}-cut_off_{args.cut_off}"
    args.name_wandb = name_wandb  # Store the name_wandb in args for use in main

    # Initialize wandb
    if wandb.run is None:
        if args.wandb == "False":
            wandb.init(project=args.project, entity=args.entity, config={}, name=name_wandb, mode="disabled")
        else:
            wandb.init(project=args.project, entity=args.entity, config={}, name=name_wandb, mode="online")

    wandb.config.update(args)

    main(args)
