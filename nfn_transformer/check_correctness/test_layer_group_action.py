import copy
import os
import random

import numpy as np
import torch
from model_vision import VisionTransformer
from torch import nn

from nfn_transformer.common.weight_space import (
    AttentionWeightSpaceFeatures,
    attention_network_spec_from_wsfeat,
)
from nfn_transformer.layers.layers import TransformersInv, TransformersLinear
from nfn_transformer.layers.misc_layers import TupleOpTransformer


def set_seed(manualSeed=3):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)


def sample_group_action(n_heads, D_k, D_v, D_A, D):
    S_h = torch.randperm(n_heads)
    # S_h = torch.range(0, n_heads-1, dtype=int)

    M = torch.rand(n_heads, D_k, D_k)
    M_T = torch.rand(n_heads, D_k, D_k).transpose(-1, -2)
    M_inv = torch.inverse(M)

    N = torch.rand(n_heads, D_v, D_v)
    N_inv = torch.inverse(N)

    P_pi_O = torch.eye(D)[torch.randperm(D)]
    P_pi_O_inv = P_pi_O.T
    P_pi_A = torch.eye(D_A)[torch.randperm(D_A)]
    P_pi_A_inv = P_pi_A.T

    return {"S_h": S_h, "M": M, "N": N, "P_pi_O": P_pi_O, "P_pi_A": P_pi_A}

def apply_group_action_to_wsfeat(wsfeat, group_actions):
    g_wsfeat = {}

    # Initialize empty lists for each transformed weight in the output
    for key in ["W_q", "W_k", "W_v", "W_o", "W_A", "W_B", "b_A", "b_B"]:
        g_wsfeat[key] = []

    # Iterate through each layer and group action
    for layer_idx, group_action in enumerate(group_actions):
        S_h, M, N, P_pi_O, P_pi_A = group_action["S_h"], group_action["M"], group_action["N"], group_action["P_pi_O"], group_action["P_pi_A"]

        # Process each key in wsfeat
        if "W_q" in wsfeat.__dict__:
            transformed_W_q = wsfeat.W_q[layer_idx][:,:,S_h] @ M[S_h].transpose(-1, -2)
            g_wsfeat["W_q"].append(transformed_W_q)

        if "W_k" in wsfeat.__dict__:
            transformed_W_k = wsfeat.W_k[layer_idx][:,:, S_h] @ torch.inverse(M[S_h])
            g_wsfeat["W_k"].append(transformed_W_k)

        if "W_v" in wsfeat.__dict__:
            transformed_W_v = wsfeat.W_v[layer_idx][:,:,S_h] @ N[S_h]
            g_wsfeat["W_v"].append(transformed_W_v)

        if "W_o" in wsfeat.__dict__:
            transformed_W_o = torch.inverse(N[S_h]) @ wsfeat.W_o[layer_idx][:,:,S_h] @ P_pi_O
            g_wsfeat["W_o"].append(transformed_W_o)

        if "W_A" in wsfeat.__dict__:
            transformed_W_A = torch.inverse(P_pi_O) @ wsfeat.W_A[layer_idx] @ P_pi_A
            g_wsfeat["W_A"].append(transformed_W_A)

        if "W_B" in wsfeat.__dict__:
            transformed_W_B = torch.inverse(P_pi_A) @ wsfeat.W_B[layer_idx]
            g_wsfeat["W_B"].append(transformed_W_B)

        if "b_A" in wsfeat.__dict__:
            transformed_b_A = wsfeat.b_A[layer_idx] @ P_pi_A
            g_wsfeat["b_A"].append(transformed_b_A)

        if "b_B" in wsfeat.__dict__:
            transformed_b_B = wsfeat.b_B[layer_idx]
            g_wsfeat["b_B"].append(transformed_b_B)

    # Convert lists back to tensors
    for key in g_wsfeat.keys():
        g_wsfeat[key] = torch.stack(g_wsfeat[key], dim=0)  # Shape: [n_layers, ...]

    return AttentionWeightSpaceFeatures(**g_wsfeat)


def check_params_eq(params1: AttentionWeightSpaceFeatures, params2: AttentionWeightSpaceFeatures):
    weight_keys = ["W_q", "W_k", "W_v", "W_o", "W_A", "W_B"]
    bias_keys = ["b_A", "b_B"]

    # Compare weights
    equal = True
    for key in weight_keys:
        weight1 = getattr(params1, key)
        weight2 = getattr(params2, key)

        for w1, w2 in zip(weight1, weight2):
            if not torch.allclose(w1, w2, atol=1e-2, rtol=1e-2, equal_nan=True):
                print(f"Mismatch found in {key}, diff {torch.abs(w1-w2).max()}")
                equal = False

    # Compare biases
    for key in bias_keys:
        bias1 = getattr(params1, key)
        bias2 = getattr(params2, key)

        for b1, b2 in zip(bias1, bias2):
            if not torch.allclose(b1, b2, atol=1e-2, rtol=1e-2, equal_nan=True):
                print(f"Mismatch found in {key}, diff {torch.abs(w1-w2).max()}")
                equal = False

    return equal


def test_layer_equivariance_group_action():
    # Define model and data parameters
    embed_dim = 32
    n_layers = 2
    n_heads = 4
    forward_mul = 1
    image_size = 28
    n_channels = 1
    patch_size = 4
    bsz = 2  # The batch size (number of models)
    D_k = embed_dim // n_heads
    D_v = embed_dim // n_heads
    D_A = embed_dim * forward_mul
    D = embed_dim

    channel_in = 2
    channel_out = 3

    ws_dict = {
        "W_q": torch.randn(n_layers, bsz, channel_in, n_heads, embed_dim, D_k),  # Shape: [n_layers, b, channel_in, h, D, D_q]
        "W_k": torch.randn(n_layers, bsz, channel_in, n_heads, embed_dim, D_k),  # Shape: [n_layers, b, channel_in, h, D, D_k]
        "W_v": torch.randn(n_layers, bsz, channel_in, n_heads, embed_dim, D_v),  # Shape: [n_layers, b, channel_in, h, D, D_v]
        "W_o": torch.randn(n_layers, bsz, channel_in, n_heads, D_v, embed_dim),  # Shape: [n_layers, b, channel_in, h, D_v, D]
        "W_A": torch.randn(n_layers, bsz, channel_in, embed_dim, D_A),           # Shape: [n_layers, b, channel_in, D, D_A]
        "W_B": torch.randn(n_layers, bsz, channel_in, D_A, embed_dim),           # Shape: [n_layers, b, channel_in, D_A, D]
        "b_A": torch.randn(n_layers, bsz, channel_in, D_A),                      # Shape: [n_layers, b, channel_in, D_A]
        "b_B": torch.randn(n_layers, bsz, channel_in, embed_dim)                 # Shape: [n_layers, b, channel_in, D]
    }

    # Display the shapes to verify the transformation
    for key, tensor in ws_dict.items():
        print(f"{key} shape: {tensor.shape}")

    wsfeat = AttentionWeightSpaceFeatures(**ws_dict)

    encoder_weight_spec =  attention_network_spec_from_wsfeat(wsfeat)

    actual_D, actual_D_q, actual_D_k, actual_D_v, actual_D_A, _ = encoder_weight_spec.get_all_dims()

    # Compare actual dimensions to expected dimensions
    assert actual_D == D, f"Expected D={D}, but got {actual_D}"
    assert actual_D_q == D_k, f"Expected D_q={D_k}, but got {actual_D_k}"
    assert actual_D_k == D_k, f"Expected D_k={D_k}, but got {actual_D_k}"
    assert actual_D_v == D_v, f"Expected D_v={D_v}, but got {actual_D_v}"
    assert actual_D_A == D_A, f"Expected D_A={D_A}, but got {actual_D_A}"

    nfn = nn.Sequential(
        # TupleOpTransformer(nn.ReLU()),
        # TupleOpTransformer(nn.ReLU(), masked_features=['W_q, W_k', 'W_v', 'W_o']),
        TransformersLinear(encoder_weight_spec, channel_in, channel_out),
        TupleOpTransformer(nn.ReLU(), masked_features=['W_q', 'W_k', 'W_v', 'W_o']),
        # TupleOpTransformer(nn.ReLU()),
        TransformersLinear(encoder_weight_spec, channel_out, channel_out),
        # TupleOpTransformer(nn.ReLU()),
        # TupleOpTransformer(nn.ReLU(), masked_features=['W_q, W_k', 'W_v', 'W_o']),
    )
    out = nfn(wsfeat)

    for _ in range(10):
        group_actions = [sample_group_action(n_heads, D_k, D_v, D_A, D) for _ in range(n_layers)]

        wsfeat1 = apply_group_action_to_wsfeat(wsfeat, group_actions)
        print(check_params_eq(nfn(wsfeat1), apply_group_action_to_wsfeat(out, group_actions)))


def test_layer_invariant_group_action():
    # Define model and data parameters
    embed_dim = 32
    n_layers = 2
    n_heads = 4
    forward_mul = 1
    image_size = 28
    n_channels = 1
    patch_size = 4
    bsz = 5  # The batch size (number of models)
    D_k = embed_dim // n_heads
    D_v = embed_dim // n_heads
    D_A = embed_dim * forward_mul
    D = embed_dim

    channel_in = 2
    channel_out = 3


    # print(encoder_weight_spec.get_all_dims())


    n_test = 100
    cnt = 0
    for _ in range(n_test):
        ws_dict = {
            "W_q": torch.randn(n_layers, bsz, channel_in, n_heads, embed_dim, D_k),  # Shape: [n_layers, b, channel_in, h, D, D_q]
            "W_k": torch.randn(n_layers, bsz, channel_in, n_heads, embed_dim, D_k),  # Shape: [n_layers, b, channel_in, h, D, D_k]
            "W_v": torch.randn(n_layers, bsz, channel_in, n_heads, embed_dim, D_v),  # Shape: [n_layers, b, channel_in, h, D, D_v]
            "W_o": torch.randn(n_layers, bsz, channel_in, n_heads, D_v, embed_dim),  # Shape: [n_layers, b, channel_in, h, D_v, D]
            "W_A": torch.randn(n_layers, bsz, channel_in, embed_dim, D_A),           # Shape: [n_layers, b, channel_in, D, D_A]
            "W_B": torch.randn(n_layers, bsz, channel_in, D_A, embed_dim),           # Shape: [n_layers, b, channel_in, D_A, D]
            "b_A": torch.randn(n_layers, bsz, channel_in, D_A),                      # Shape: [n_layers, b, channel_in, D_A]
            "b_B": torch.randn(n_layers, bsz, channel_in, embed_dim)                 # Shape: [n_layers, b, channel_in, D]
        }

        wsfeat = AttentionWeightSpaceFeatures(**ws_dict)

        encoder_weight_spec = attention_network_spec_from_wsfeat(wsfeat)

        nfn = nn.Sequential(
            TransformersLinear(encoder_weight_spec, channel_in, channel_out),
            TupleOpTransformer(nn.ReLU(), masked_features=['W_q', 'W_k', 'W_v', 'W_o']),
            # TransformersLinear(encoder_weight_spec, channel_out, channel_out),
            # TupleOpTransformer(nn.ReLU()),
            # TransformersInv(encoder_weight_spec, channel_in, channel_in),
            TransformersInv(encoder_weight_spec, channel_out, channel_out),
            # nn.Linear(n_layers * channel_out, 1)
        )

        out = nfn(wsfeat)

        group_actions = [sample_group_action(n_heads, D_k, D_v, D_A, D) for _ in range(n_layers)]

        wsfeat1 = apply_group_action_to_wsfeat(wsfeat, group_actions)
        out1 = nfn(wsfeat1)


        inv = torch.allclose(out, out1, atol=1e-3, rtol=1e-3, equal_nan=True)
        # print((out - out1).abs())
        # print("out", out.detach().numpy())
        # print("out1", out1.detach().numpy())
        cnt += inv
        print(f"NFN is invariant: {inv}.")
    print(f"Success Rate: {cnt}/{n_test}")



if __name__ == "__main__":
    set_seed(6)
    test_layer_equivariance_group_action()
    test_layer_invariant_group_action()
