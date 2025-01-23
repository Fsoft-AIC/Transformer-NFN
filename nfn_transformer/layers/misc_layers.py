import torch
import torch.nn as nn

from nfn_transformer.common.weight_space import (
    AttentionNetworkSpec,
    AttentionWeightSpaceFeatures,
    LinearWeightSpaceFeatures,
    NetworkSpec,
)
from nfn_transformer.layers.layer_utils import shape_wsfeat_symmetry


class TupleOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, wsfeat: LinearWeightSpaceFeatures) -> LinearWeightSpaceFeatures:
        out_weights = [self.op(w) for w in wsfeat.weights]
        out_bias = [self.op(b) for b in wsfeat.biases]
        return LinearWeightSpaceFeatures(out_weights, out_bias)

    def __repr__(self):
        return f"TupleOp({self.op})"

class StatFeaturizer(nn.Module):
    def forward(self, wsfeat: LinearWeightSpaceFeatures) -> torch.Tensor:
        out = []
        for (weight, bias) in wsfeat:
            out.append(self.compute_stats(weight))
            out.append(self.compute_stats(bias))
        return torch.cat(out, dim=-1)

    def compute_stats(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the statistics of the given tensor."""
        tensor = torch.flatten(tensor, start_dim=2) # (B, C, H*W)
        mean = tensor.mean(-1) # (B, C)
        var = tensor.var(-1) # (B, C)
        q = torch.tensor([0., 0.25, 0.5, 0.75, 1.]).to(tensor.device)
        quantiles = torch.quantile(tensor, q, dim=-1) # (5, B, C)
        return torch.stack([mean, var, *quantiles], dim=-1) # (B, C, 7)

    @staticmethod
    def get_num_outs(network_spec: NetworkSpec):
        """Returns the number of outputs of the StatFeaturizer layer."""
        return 2 * len(network_spec) * 7


class FlattenWeights(nn.Module):
    def __init__(self, network_spec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, wsfeat):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        outs = []
        for i in range(len(self.network_spec)):
            w, b = wsfeat[i]
            outs.append(torch.flatten(w, start_dim=2).transpose(1, 2))
            outs.append(b.transpose(1, 2))
        return torch.cat(outs, dim=1)  # (B, N, C)


class TupleOpTransformer(nn.Module):
    def __init__(self, op, masked_features=None):
        super().__init__()
        self.op = op
        self.mask_features = masked_features if masked_features else []

    def forward(self, wsfeat: AttentionWeightSpaceFeatures) -> AttentionWeightSpaceFeatures:
        keys = ["W_q", "W_k", "W_v", "W_o", "W_A", "W_B", "b_A", "b_B"]
        out_dict = {}
        for key in keys:
            if key in self.mask_features:
                out_dict[key] = getattr(wsfeat, key)
            else:
                out_dict[key] = [self.op(w) for w in getattr(wsfeat, key)]

        return AttentionWeightSpaceFeatures(**out_dict)

    def __repr__(self):
        return f"TupleOpTransformer({self.op})"


class FlattenWeightsTransformer(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec):
        super().__init__()
        self.encoder_weight_spec = encoder_weight_spec

    def forward(self, wsfeat):
        out = []
        L = len(wsfeat)  # Number of layers

        for i in range(L):
            W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B = wsfeat[i]
            out.append(torch.flatten(W_k, start_dim=2).transpose(1, 2))
            out.append(torch.flatten(W_q, start_dim=2).transpose(1, 2))
            out.append(torch.flatten(W_v, start_dim=2).transpose(1, 2))
            out.append(torch.flatten(W_o, start_dim=2).transpose(1, 2))
            out.append(torch.flatten(W_A, start_dim=2).transpose(1, 2))
            out.append(torch.flatten(W_B, start_dim=2).transpose(1, 2))
            out.append(b_A.transpose(1, 2))
            out.append(b_B.transpose(1, 2))
        return torch.cat(out, dim=1)  # (B, N, C)


class StatFeaturizerTransformer(nn.Module):
    def forward(self, wsfeat: AttentionWeightSpaceFeatures) -> torch.Tensor:
        out = []
        L = len(wsfeat)  # Number of layers

        for i in range(L):
            W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B = wsfeat[i]
            out.append(self.compute_stats(W_q))
            out.append(self.compute_stats(W_k))
            out.append(self.compute_stats(W_v))
            out.append(self.compute_stats(W_o))
            out.append(self.compute_stats(W_A))
            out.append(self.compute_stats(W_B))
            out.append(self.compute_stats(b_A))
            out.append(self.compute_stats(b_B))

        return torch.cat(out, dim=-1)

    def compute_stats(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the statistics of the given tensor."""
        tensor = torch.flatten(tensor, start_dim=2) # (B, C, H*W)
        mean = tensor.mean(-1) # (B, C)
        var = tensor.var(-1) # (B, C)
        q = torch.tensor([0., 0.25, 0.5, 0.75, 1.]).to(tensor.device)
        quantiles = torch.quantile(tensor, q, dim=-1) # (5, B, C)
        return torch.stack([mean, var, *quantiles], dim=-1) # (B, C, 7)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the StatFeaturizerTransformer layer."""
        return 8 * len(network_spec) * 7
