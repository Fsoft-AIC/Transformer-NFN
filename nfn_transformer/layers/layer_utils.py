import math

from einops import rearrange
from torch import nn

from nfn_transformer.common.weight_space import LinearWeightSpaceFeatures


def set_init_(*layers, init_type="pytorch_default"):
    in_chan = 0
    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            in_chan += layer.in_channels
        elif isinstance(layer, nn.Linear):
            in_chan += layer.in_features
        else:
            raise NotImplementedError(f"Unknown layer type {type(layer)}")
    if init_type == "pytorch_default":
        bd = math.sqrt(1 / in_chan)
        for layer in layers:
            nn.init.uniform_(layer.weight, -bd, bd)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -bd, bd)
    elif init_type == "kaiming_normal":
        std = math.sqrt(2 / in_chan)
        for layer in layers:
            nn.init.normal_(layer.weight, 0, std)
            layer.bias.data.zero_()
    else:
        raise NotImplementedError(f"Unknown init type {init_type}.")

def set_init_einsum_(*layers, init_type="pytorch_default",scale_degree=1):
    in_chan = 0
    out_chan=0
    for layer in layers:
        in_chan += layer.fan_in
        out_chan += layer.fan_out
    if init_type == "pytorch_default":
        bd = math.sqrt(1 / in_chan) ** scale_degree
        for layer in layers:
            nn.init.uniform_(layer.weight, -bd, bd)

    elif init_type == "kaiming_normal":
        std = math.sqrt(2 / in_chan) ** scale_degree
        for layer in layers:
            nn.init.normal_(layer.weight, 0, std)

    elif init_type == "xavier_normal":
        std = math.sqrt(2 / (in_chan+out_chan)) ** scale_degree
        for layer in layers:
            nn.init.normal_(layer.weight, 0, std)
    elif init_type == "xavier_uniform":
        bd = math.sqrt(6 / (in_chan+out_chan))** scale_degree
        for layer in layers:
            nn.init.uniform_(layer.weight, -bd, bd)
    elif init_type == "uniform":
        for layer in layers:
            nn.init.uniform_(layer.weight, -1, 1)
    else:
        raise NotImplementedError(f"Unknown init type {init_type}.")


def shape_wsfeat_symmetry(params, network_spec: LinearWeightSpaceFeatures):
    """Reshape so last 2 dims have symmetry, channel dims have all nonsymmetry.
    E.g., for conv weights we reshape (B, C, out, in, h, w) -> (B, C * h * w, out, in)
    """
    weights, bias = params.weights, params.biases
    reshaped_weights = []
    for weight, weight_spec in zip(weights, network_spec.weight_spec):
        if len(weight_spec.shape) == 2:  # mlp weight matrix:
            reshaped_weights.append(weight)
        else:
            reshaped_weights.append(rearrange(weight, "b c o i h w -> b (c h w) o i"))
    return LinearWeightSpaceFeatures(reshaped_weights, bias)
