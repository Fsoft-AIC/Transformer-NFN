import collections
import collections.abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from einops import rearrange


@dataclass(frozen=True)
class ArraySpec:
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class NetworkSpec:
    weight_spec: List[ArraySpec]
    bias_spec: List[ArraySpec]
    layer_weight_shapes: List[ArraySpec]

    def get_io(self):
        # n_in, n_out
        return self.weight_spec[0].shape[1], self.weight_spec[-1].shape[0]

    def get_matrices_shape(self):
        return self.layer_weight_shapes

    def get_num_params(self):
        """Returns the number of parameters in the network."""
        num_params = 0
        for w, b in zip(self.weight_spec, self.bias_spec):
            num_weights = 1
            for dim in w.shape:
                assert dim != -1
                num_weights *= dim
            num_biases = 1
            for dim in b.shape:
                assert dim != -1
                num_biases *= dim
            num_params += num_weights+num_biases
        return num_params

    def __len__(self):
        return len(self.weight_spec)


class LinearWeightSpaceFeatures(collections.abc.Sequence):
    def __init__(self, weights, biases):
        # No mutability
        self.weights = weights
        self.biases = biases

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        return zip(self.weights, self.biases)

    def __getitem__(self, index):
        return (self.weights[index], self.biases[index])

    def to(self, device):
        """Moves all tensors to device."""
        return LinearWeightSpaceFeatures(tuple(w.to(device, non_blocking=True) for w in self.weights), tuple(b.to(device, non_blocking=True) for b in self.biases))


class AttentionWeightSpaceFeatures(collections.abc.Sequence):
    def __init__(self, W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B) -> None:
        self.W_q, self.W_k, self.W_v, self.W_o, self.W_A, self.W_B, self.b_A, self.b_B = W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B

    def __len__(self):
        return len(self.W_q)

    def __iter__(self):
        return zip(self.W_q, self.W_k, self.W_v, self.W_o, self.W_A, self.W_B, self.b_A, self.b_B)

    def __getitem__(self, index):
        return self.W_q[index], self.W_k[index], self.W_v[index], self.W_o[index], self.W_A[index], self.W_B[index], self.b_A[index], self.b_B[index]

    def to(self, device):
        """Moves all tensors to device."""
        return AttentionWeightSpaceFeatures(tuple(wq.to(device, non_blocking=True) for wq in self.W_q), \
            tuple(wk.to(device, non_blocking=True) for wk in self.W_k),\
            tuple(wv.to(device, non_blocking=True) for wv in self.W_v),\
            tuple(wo.to(device, non_blocking=True) for wo in self.W_o),\
            tuple(wa.to(device, non_blocking=True) for wa in self.W_A),\
            tuple(wb.to(device, non_blocking=True) for wb in self.W_B),\
            tuple(ba.to(device, non_blocking=True) for ba in self.b_A),\
            tuple(bb.to(device, non_blocking=True) for bb in self.b_B))

@dataclass(frozen=True)
class AttentionNetworkSpec:
    weight_spec: Dict[str,List[ArraySpec]]
    bias_spec: Dict[str,List[ArraySpec]]
    num_params: int
    def get_io(self):
        # n_in, n_out
        return self.weight_spec['W_q'][-1].shape[-2], self.weight_spec['W_q'][0].shape[-1]

    def get_all_dims(self):
        #return dim D, D_q, D_k, D_v, D_A, h
        return self.weight_spec['W_q'][-1].shape[-2],  self.weight_spec['W_q'][0].shape[-1], self.weight_spec['W_k'][0].shape[-1], self.weight_spec['W_v'][0].shape[-1], self.weight_spec['W_A'][0].shape[-1], self.weight_spec['h'][0].shape
    
    def get_num_params(self):
        """Returns the number of parameters in the network."""
        return self.num_params
    
    def __len__(self):
        return len(self.weight_spec['W_q'])




def network_spec_from_wsfeat(wsfeat: LinearWeightSpaceFeatures, set_all_dims=False) -> NetworkSpec:
    assert len(wsfeat.weights) == len(wsfeat.biases)
    weight_specs = []
    bias_specs = []
    layer_weight_shapes = []
    L = len(wsfeat.weights)

    for i, (weight, bias) in enumerate(zip(wsfeat.weights, wsfeat.biases)):
        # Determine the shape of each layer's weights
        if weight.dim() == 4:
            layer_weight_shape = weight.shape
        elif weight.dim() == 6:
            layer_weight_shape = rearrange(weight, "b c o i h w -> b (c h w) o i").shape
        else:
            raise ValueError(f"Unsupported weight dim: {weight.dim()}")

        layer_weight_shapes.append(layer_weight_shape)

        # Define weight shape with symmetry considerations
        if weight.dim() == 4:
            weight_shape = [-1, -1]
        elif weight.dim() == 6:
            weight_shape = [-1, -1, weight.shape[-2], weight.shape[-1]]

        if i == 0 or set_all_dims:
            weight_shape[1] = layer_weight_shape[3]
        if i == L - 1 or set_all_dims:
            weight_shape[0] = layer_weight_shape[2]
        weight_specs.append(ArraySpec(tuple(weight_shape)))

        # Define bias shape
        bias_shape = (-1,)
        if i == L - 1 or set_all_dims:
            bias_shape = (bias.shape[-1],)
        bias_specs.append(ArraySpec(bias_shape))

    return NetworkSpec(weight_specs, bias_specs, layer_weight_shapes)


def attention_network_spec_from_wsfeat(wsfeat: AttentionNetworkSpec, set_all_dims=False) -> AttentionNetworkSpec:
    assert len(wsfeat.W_q) == len(wsfeat.W_k) == len(wsfeat.W_v) == len(wsfeat.W_o) == len(wsfeat.W_A) == \
        len(wsfeat.W_B) == len(wsfeat.b_A) == len(wsfeat.b_B)
    weight_specs = {'W_q':[], 'W_k':[], 'W_v':[], 'W_o':[], 'W_A':[], 'W_B':[], "h":[]}
    weight_dict = {'W_q': wsfeat.W_q, 'W_k': wsfeat.W_k, 'W_v': wsfeat.W_v, 'W_o': wsfeat.W_o, 'W_A': wsfeat.W_A, 'W_B': wsfeat.W_B}
    bias_specs = {'b_A':[], 'b_B':[]}
    bias_dict = {'b_A': wsfeat.b_A, 'b_B': wsfeat.b_B}
    L = len(wsfeat.W_q)
    num_params = 0
    for i in range(L):
        # Define weight shape with symmetry considerations
        for weight_key in ['W_q', 'W_k', 'W_v', 'W_o','W_A', 'W_B']:
            weight = weight_dict[weight_key][i]
            num_weights = 1
            for dim in weight.shape[1:]:
                assert dim != -1
                num_weights *= dim
            num_params += num_weights

            weight_shape = [-1, -1]
            if i == 0 or set_all_dims:
                weight_shape[1] = weight.shape[-1]
            if i == L - 1 or set_all_dims:
                weight_shape[0] = weight.shape[-2]
            weight_specs[weight_key].append(ArraySpec(tuple(weight_shape)))


        for bias_key in ['b_A', 'b_B']:
            bias = bias_dict[bias_key][i]
            num_biases = 1
            for dim in bias.shape[1:]:
                assert dim != -1
                num_biases *= dim
            num_params += num_biases

            bias_shape = (-1,)
            if i == L - 1 or set_all_dims:
                bias_shape = (bias.shape[-1],)
            bias_specs[bias_key].append(ArraySpec(bias_shape))
        
        weight_specs["h"].append(ArraySpec(weight_dict["W_q"][i].shape[2]))

    return AttentionNetworkSpec(weight_specs, bias_specs, num_params)