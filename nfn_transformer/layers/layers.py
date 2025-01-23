import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from nfn_transformer.common.weight_space import AttentionNetworkSpec, AttentionWeightSpaceFeatures, NetworkSpec, LinearWeightSpaceFeatures
from nfn_transformer.layers.layer_utils import (
    set_init_,
    set_init_einsum_,
)
class HNPSLinear(nn.Module):
    def __init__(self, in_network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.in_network_spec = in_network_spec

        layer_weight_shapes = in_network_spec.get_matrices_shape()
        layer_in_weight_shape, layer_out_weight_shape = layer_weight_shapes[0], layer_weight_shapes[-1]
        self.L = len(in_network_spec)
        in_filter_facs = [int(np.prod(spec.shape[2:])) for spec in in_network_spec.weight_spec]
        out_filter_facs = in_filter_facs
        
        for i in range(self.L):
            in_filter_fac = in_filter_facs[i]
            out_filter_fac = out_filter_facs[i]
            if i == 0:
                self.layer_0_Y_W = EinsumLayer(equation="mnqk, bnjq -> bmjk", 
                                              weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels, 
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_Y_b = EinsumLayer(equation="mnk, bnj -> bmjk",
                                               weight_shape=[out_filter_fac * out_channels, in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask= [0, 1, 0])
                self.layer_0_z_W = EinsumLayer(equation="mnq, bnjq -> bmj",
                                               weight_shape=[out_channels, in_filter_fac * in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_z_b = EinsumLayer(equation="mn, bnj -> bmj",
                                               weight_shape=[out_channels, in_channels],
                                               fan_in_mask=[0, 1])

                set_init_einsum_(
                    self.layer_0_Y_W,
                    self.layer_0_Y_b,
                    init_type=init_type,
                )
                set_init_einsum_(
                    self.layer_0_z_W,
                    self.layer_0_z_b,
                    init_type=init_type,
                )

            elif i == self.L-1:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mnpj, bnpk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mnpj, bnp -> bmj",
                                            weight_shape=[out_channels, in_filter_fac * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_tau",
                                EinsumLayer(equation="mj, b-> bmj",
                                            weight_shape=[out_channels, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 0])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                attributes = []
                attributes.extend([getattr(self, f"layer_{i}_z_b")])
                attributes.extend([getattr(self, f"layer_{i}_z_tau")])

                set_init_einsum_(*attributes,
                    init_type=init_type,
                )                
            else:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mn, bnjk -> bmjk",
                                            weight_shape=[out_filter_fac * out_channels, in_filter_fac * in_channels],
                                            fan_in_mask=[0, 1])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mn, bnj -> bmj",
                                            weight_shape=[out_channels, in_channels],
                                            fan_in_mask=[0, 1])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    init_type=init_type,
                )

    
    def forward(self, wsfeat: LinearWeightSpaceFeatures) -> LinearWeightSpaceFeatures:
        out_weights, out_biases = [], []
        for i in range(self.L):
            # weight, bias = wsfeat['weight'][i], wsfeat['bias'][i]
            # weight, bias = weight.cuda(), bias.cuda()
            
            weight, bias = wsfeat[i]
            if  i == 0:
                Y_W = self.layer_0_Y_W(weight)
                Y_b = self.layer_0_Y_b(bias)
                #make a random tensor for Y_b with the same shape as Y_W
                #Y_b = torch.randn_like(Y_W)
                out_weights.append(Y_W + Y_b)
                
                z_W = self.layer_0_z_W(weight)
                z_b = self.layer_0_z_b(bias)
                out_biases.append(z_W + z_b)

            elif i == self.L-1:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                z_tau = getattr(self, f"layer_{i}_z_tau")(torch.tensor([1], device=weight.device))
                out_biases.append(z_b + z_tau)
            
            else:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                out_biases.append(z_b)
        #return {'weight': out_weights, 'bias': out_biases}
        return LinearWeightSpaceFeatures(out_weights, out_biases)

class ElementwiseParamNormalize(nn.Module):
    def __init__(self, hidden, mode_normalize) -> None:
        super().__init__()
        self.hidden = hidden
        self.mode_normalize = mode_normalize
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.ones(hidden))
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)


    def forward(self, input):
        if self.mode_normalize == "param_mul_L2":
            if input.dim() == 6: #C NN
                    input_shape = input.shape
                    input = rearrange(input, 'b c i j k l -> b i j (c k l)')
                    input_regularized = F.normalize(input, p=2.0, dim=-1)
                    input_regularized = self.weight * input_regularized + self.bias
                    input_regularized = rearrange(input_regularized, 'b i j (c k l) -> b c i j k l',
                                                    c = input_shape[1], k = input_shape[-2],
                                                    l = input_shape[-1])
            elif input.dim() == 4: # MLP
                input = rearrange(input, 'b c i j-> b i j c')
                input_regularized = F.normalize(input, p=2.0, dim=-1)
                input_regularized = self.weight * input_regularized + self.bias
                input_regularized = rearrange(input_regularized, 'b i j c -> b c i j')

            elif input.dim() == 3: #bias
                input = rearrange(input, 'b c j-> b j c')
                input_regularized = F.normalize(input, p=2.0, dim=-1)
                input_regularized = self.weight * input_regularized + self.bias
                input_regularized = rearrange(input_regularized, 'b j c -> b c j')
        return input_regularized

class HNPSPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec, nfn_channels, mode_pooling="param_mul_L2"):
        super().__init__()
        self.network_spec = network_spec
        self.mode_pooling = mode_pooling
        self.nfn_channels = nfn_channels
        if self.mode_pooling == "param_mul_L2":
            for i in range(len(network_spec)):
                if len(network_spec.weight_spec[i].shape) == 4: #CNN
                    self.add_module(f"regularize_W_{i}",
                                    ElementwiseParamNormalize(nfn_channels *
                                                math.prod(network_spec.weight_spec[i].shape[-2:]),
                                                mode_normalize=mode_pooling)
                                    )
                else:
                    self.add_module(f"regularize_W_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_pooling))
                self.add_module(f"regularize_b_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_pooling))


    def forward(self, wsfeat: LinearWeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            if self.mode_pooling == "param_mul_L2":
                regularizer_w = getattr(self, f"regularize_W_{i}")
                regularizer_b = getattr(self, f"regularize_b_{i}")
            else:
                regularizer_w = self.regularize_without_param
                regularizer_b = self.regularize_without_param


            if i == 0:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=2))  # average over rows

            elif i == len(wsfeat) - 1:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=3))  # average over cols

            else:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=(2,3)).unsqueeze(-1))

            if i == len(wsfeat) - 1:
                out.append(bias)
            else:
                # bias_regularized = F.normalize(bias, dim=1, p=2.0)
                bias_regularized = regularizer_b(bias)
                out.append(bias_regularized.mean(dim=-1).unsqueeze(-1))

        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    def regularize_without_param(self, weight):
        if self.mode_pooling == "L1":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=1.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_pooling == "L2":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_pooling == "L2_square":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2],
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2

        return weight_regularized

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_in, n_out = network_spec.get_io()
        num_outs = 0
        for i, fac in enumerate(filter_facs):
            if i == 0:
                num_outs += n_in * fac + 1
            elif i == len(filter_facs) - 1:
                num_outs += n_out * fac + n_out
            else:
                num_outs += fac + 1
        return num_outs



class EinsumLayer(nn.Module):
    def __init__(self, equation="", weight_shape=None, input_shape=None, fan_in_mask=None, unsqueeze_dims=None) -> None:
        super().__init__()
        self.equation = equation
        if len(self.equation) == 0:
            return

        self.weight_shape_list = weight_shape
        self.weight_shape_tensor = torch.tensor(weight_shape, dtype=torch.int)
        if input_shape is not None:
            self.input_shape_tensor = torch.tensor(input_shape, dtype=torch.int)
        else:
            self.input_shape_tensor = self.weight_shape_tensor

        # Get fan_in and fan_out
        self.fan_in_mask = torch.tensor(fan_in_mask).ge(0.5)
        if torch.all(self.fan_in_mask == False):
            self.fan_in = 0
        else:
            # self.fan_in = torch.prod(self.weight_shape_tensor[self.fan_in_mask])
            self.fan_in = torch.prod(self.input_shape_tensor[self.fan_in_mask])
        self.fan_out_mask = torch.tensor(fan_in_mask).lt(0.5)
        if torch.all(self.fan_out_mask == False):
            self.fan_out = 0
        else:
            # self.fan_out = torch.prod(self.weight_shape_tensor[self.fan_out_mask])
            self.fan_out = torch.abs(torch.prod(self.input_shape_tensor[self.fan_out_mask]))

        # Setup equation
        self.equation = equation
        self.weight = nn.Parameter(torch.empty(self.weight_shape_list))
        #self.weight = nn.Parameter(torch.ones(self.weight_shape_list))
        self.unsqueeze_dims = unsqueeze_dims if unsqueeze_dims is not None else []
        self.input_parts = self.equation.split('->')[0].split(',')
        self.num_inputs = len(self.input_parts)

    def forward(self, input=None):
        if self.num_inputs == 1:
            result = torch.einsum(self.equation, self.weight)

        elif self.num_inputs == 2:
            if 'b' in self.input_parts[0]:  # The first part is the input tensor
                result = torch.einsum(self.equation, input, self.weight)
            elif 'b' in self.input_parts[1]:  # The second part is the input tensor
                result = torch.einsum(self.equation, self.weight, input)
            else:
                raise ValueError("No batch dimension 'b' found in the einsum equation input parts.")

        else:
            raise ValueError(f"Unexpected number of input parts ({self.num_inputs}) in einsum equation: {self.equation}")
        
        for dim in self.unsqueeze_dims:
            result = result.unsqueeze(dim)

        return result

        
#TODO: Add Transformer Layer to process wsfeat of the encoder

class TransformersLinear(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec, in_channels, out_channels, init_type="pytorch_default", scale_degree = 2):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec

        D, D_q, D_k, D_v, D_a, h =  encoder_weight_spec.get_all_dims() #not yet implemented
        
        self.L = len(encoder_weight_spec)
        for i in range(self.L):
            # -----------------------------------
            #            W_Q Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_Q",EinsumLayer(equation="bdhpk, edjp -> behjk",
                                        weight_shape=[self.e, self.d, D, D],
                                        input_shape=[-1, self.d, h, D, D_q],
                                        # fan_in_mask=[0, 1, 0, 1])
                                        fan_in_mask=[0, 1, 0, 0, 1]
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_Q"), init_type=init_type)
            # -----------------------------------
            #            W_K Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_K",EinsumLayer(equation="bdhpk, edjp -> behjk",
                                        weight_shape=[self.e, self.d, D, D],
                                        input_shape=[-1, self.d, h, D, D_k],
                                        # fan_in_mask=[0, 1, 0, 1])
                                        fan_in_mask=[0, 1, 0, 0, 1]
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_K"), init_type=init_type)
            # -----------------------------------
            #            W_V Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_V",EinsumLayer(equation="bdhpk, edjp -> behjk",
                                        weight_shape=[self.e, self.d, D, D],
                                        input_shape=[-1, self.d, h, D, D_v],
                                        # fan_in_mask=[0, 1, 0, 1])
                                        fan_in_mask=[0, 1, 0, 0, 1],
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_V"), init_type=init_type)
            # -----------------------------------
            #            W_O Terms
            # -----------------------------------
            self.add_module(f"layer_{i}_W_O_1", EinsumLayer(equation="bdhjk, ed -> behj",
                                        weight_shape=[self.e, self.d],
                                        input_shape=[-1, self.d, h, D_v, D],
                                        # fan_in_mask=[0, 1],
                                        fan_in_mask=[0, 1, 0, 0, 1],
                                        unsqueeze_dims=[-1]
                                        ))
            
            self.add_module(f"layer_{i}_W_O_2", EinsumLayer(equation="bdhjk, ed -> behjk",
                                        weight_shape=[self.e, self.d],
                                        input_shape=[-1, self.d, h, D_v, D],
                                        # fan_in_mask=[0, 1]
                                        fan_in_mask=[0, 1, 0, 0, 0]
                                        ))
            set_init_einsum_(getattr(self, f"layer_{i}_W_O_1"), getattr(self, f"layer_{i}_W_O_2"), init_type=init_type)

            # -----------------------------------
            #            W_A Terms
            # -----------------------------------
            # 1st Term
            self.add_module(f"layer_{i}_W_A_W_QK", EinsumLayer(equation="bdhpq, edpq -> be",
                                            weight_shape=[self.e, self.d, D, D],
                                            input_shape=[-1, self.d, h, D, D],
                                            # fan_in_mask=[0, 1, 1, 1],
                                            fan_in_mask=[0, 1, 1, 1, 1],
                                            unsqueeze_dims=[-1, -1]))
            
            # 2nd Term
            self.add_module(f"layer_{i}_W_A_W_VO_1", EinsumLayer(equation="bdhpq, edp -> be",
                                                weight_shape=[self.e, self.d, D],
                                                input_shape=[-1, self.d, h, D, D],
                                                fan_in_mask=[0, 1, 1, 1, 1],
                                                # fan_in_mask=[0, 1, 1],
                                                unsqueeze_dims=[-1, -1]))
            
            # 3rd Term
            self.add_module(f"layer_{i}_W_A_W_VO_2", EinsumLayer(equation="bdhpj, edp -> bej",
                                                weight_shape=[self.e, self.d, D],
                                                input_shape=[-1, self.d, h, D, D],
                                                fan_in_mask=[0, 1, 1, 1, 0],
                                                # fan_in_mask=[0, 1, 1],
                                                unsqueeze_dims=[-1]))

            # 4th Term
            self.add_module(f"layer_{i}_W_A_W_A_1", EinsumLayer(equation="bdpq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            # fan_in_mask=[0, 1],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 1],
                                            unsqueeze_dims=[-1, -1]))

            # 5th Term
            self.add_module(f"layer_{i}_W_A_W_A_2", EinsumLayer(equation="bdjq, ed -> bej",                                                               
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 0, 1],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-1]))
            
            # 6th Term
            self.add_module(f"layer_{i}_W_A_W_A_3", EinsumLayer(equation="bdpk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 0],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-2]))

            # 7th Term
            self.add_module(f"layer_{i}_W_A_W_A_4", EinsumLayer(equation="bdjk, ed -> bejk",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 0, 0],
                                            # fan_in_mask=[0, 1]
                                            ))
            
            # 8th Term
            self.add_module(f"layer_{i}_W_A_W_B_1", EinsumLayer(equation="bdpq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1, -1]))
            
            # 9th Term
            self.add_module(f"layer_{i}_W_A_W_B_2", EinsumLayer(equation="bdkq, edq -> bek",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 0, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-2]))
            
            # 10th Term
            self.add_module(f"layer_{i}_W_A_b_A_1", EinsumLayer(equation="bdq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 1],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-1, -1]))
            
            # 11th Term
            self.add_module(f"layer_{i}_W_A_b_A_2", EinsumLayer(equation="bdk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 0],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-2]))

            # 12th Term
            self.add_module(f"layer_{i}_W_A_b_B", EinsumLayer(equation="bdq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1, -1]))

            # 13th Term
            self.add_module(f"layer_{i}_W_A_bias", EinsumLayer(equation="e -> e",
                                        weight_shape=[self.e],
                                        input_shape=[self.e],
                                        fan_in_mask=[0],
                                        # fan_in_mask=[0],
                                        unsqueeze_dims=[0, -1, -1]))
            
            set_init_einsum_(
                            getattr(self, f"layer_{i}_W_A_b_A_2"),
                            getattr(self, f"layer_{i}_W_A_b_B"),
                            getattr(self, f"layer_{i}_W_A_bias"),
                            init_type=init_type)
            set_init_einsum_(getattr(self, f"layer_{i}_W_A_W_QK"),
                            getattr(self, f"layer_{i}_W_A_W_VO_1"),
                            getattr(self, f"layer_{i}_W_A_W_VO_2"),
init_type=init_type, scale_degree=scale_degree)
            set_init_einsum_(getattr(self, f"layer_{i}_W_A_b_A_1"),
                            getattr(self, f"layer_{i}_W_A_W_B_1"),
                            getattr(self, f"layer_{i}_W_A_W_B_2"),
                            getattr(self, f"layer_{i}_W_A_W_A_1"),
                            getattr(self, f"layer_{i}_W_A_W_A_2"),
                            getattr(self, f"layer_{i}_W_A_W_A_3"),
                            getattr(self, f"layer_{i}_W_A_W_A_4"),

init_type=init_type, scale_degree=scale_degree)

            # -----------------------------------
            #            b_A Terms
            # -----------------------------------
            # 1st Term
            self.add_module(f"layer_{i}_b_A_QK", EinsumLayer(equation="bdhpq, edpq -> be",
                                            weight_shape=[self.e, self.d, D, D],
                                            input_shape=[-1, self.d, h, D, D],
                                            fan_in_mask=[0, 1, 1, 1, 1],
                                            # fan_in_mask=[0, 1, 1, 1],
                                            unsqueeze_dims=[-1]))
            
            # 2nd Term
            self.add_module(f"layer_{i}_b_A_VO", EinsumLayer(equation="bdhpq, edp -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, h, D, D],
                                            fan_in_mask=[0, 1, 1, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]))
            
            # 3rd Term
            self.add_module(f"layer_{i}_b_A_W_A_1",  EinsumLayer(equation="bdpq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 1],
                                            # fan_in_mask=[0, 1],
                                            unsqueeze_dims=[-1]))

            # 4th Term
            self.add_module(f"layer_{i}_b_A_W_A_2", EinsumLayer(equation="bdpk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D, D_a],
                                            fan_in_mask=[0, 1, 1, 0],
                                            # fan_in_mask=[0, 1]
                                            ))

            # 5th Term
            self.add_module(f"layer_{i}_b_A_W_B_1", EinsumLayer(equation="bdpq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 1 ,1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]))

            # 6th Term
            self.add_module(f"layer_{i}_b_A_W_B_2", EinsumLayer(equation="bdkq, edq -> bek",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D_a, D],
                                            fan_in_mask=[0, 1, 0, 1]
                                            # fan_in_mask=[0, 1, 1]
                                            ))

            # 7th Term
            self.add_module(f"layer_{i}_b_A_b_A_1",  EinsumLayer(equation="bdq, ed -> be",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]
                                            # fan_in_mask=[0, 1, 1, 0]
                                            ))

            # 8th Term
            self.add_module(f"layer_{i}_b_A_b_A_2", EinsumLayer(equation="bdk, ed -> bek",
                                            weight_shape=[self.e, self.d],
                                            input_shape=[-1, self.d, D_a],
                                            fan_in_mask=[0, 1, 0],
                                            # fan_in_mask=[0, 1, 0]
                                            ))

            # 9th Term
            self.add_module(f"layer_{i}_b_A_b_B",  EinsumLayer(equation="bdq, edq -> be",
                                            weight_shape=[self.e, self.d, D],
                                            input_shape=[-1, self.d, D],
                                            fan_in_mask=[0, 1, 1],
                                            # fan_in_mask=[0, 1, 1],
                                            unsqueeze_dims=[-1]))

            # 10th Term
            self.add_module(f"layer_{i}_b_A_bias", EinsumLayer(equation="e -> e",
                                            weight_shape=[self.e],
                                            input_shape=[self.e],
                                            fan_in_mask=[0],
                                            # fan_in_mask=[0],
                                            unsqueeze_dims=[0, -1]))
            
            set_init_einsum_(
                
                            getattr(self, f"layer_{i}_b_A_b_A_2"),
                            getattr(self, f"layer_{i}_b_A_b_B"),
                            getattr(self, f"layer_{i}_b_A_bias"),

                            init_type=init_type)
            set_init_einsum_(getattr(self, f"layer_{i}_b_A_QK"),
                            getattr(self, f"layer_{i}_b_A_VO"),
                            init_type=init_type, scale_degree=scale_degree)
            set_init_einsum_(getattr(self, f"layer_{i}_b_A_W_B_1"),
                            getattr(self, f"layer_{i}_b_A_W_B_2"), getattr(self, f"layer_{i}_b_A_W_A_1"),
                            getattr(self, f"layer_{i}_b_A_W_A_2"),getattr(self, f"layer_{i}_b_A_b_A_1"),  #bAbA1
                            init_type=init_type, scale_degree=scale_degree)

            # -----------------------------------
            #            W_B Terms
            # -----------------------------------
            
            # 1st Term
            self.add_module(f"layer_{i}_W_B_QK_1", EinsumLayer(
                equation="bdhpq, edkpq -> bek",
                weight_shape=[self.e, self.d, D, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1],
                # fan_in_mask=[0, 1, 0, 1, 1],
                unsqueeze_dims=[-2]
            ))

            # 2nd Term
            self.add_module(f"layer_{i}_W_B_VO", EinsumLayer(
                equation="bdhpq, edkp -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1],
                # fan_in_mask=[0, 1, 0, 1],
                unsqueeze_dims=[-2]
            ))

            # 3rd Term
            self.add_module(f"layer_{i}_W_B_W_A_1", EinsumLayer(
                equation="bdpq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D, D_a],
                fan_in_mask=[0, 1, 1, 1],
                # fan_in_mask=[0, 1, 0],
                unsqueeze_dims=[-2]
            ))

            # 4th Term
            self.add_module(f"layer_{i}_W_B_W_A_2", EinsumLayer(
                equation="bdpj, edk -> bejk",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D, D_a],
                fan_in_mask=[0, 1, 1, 0],
                # fan_in_mask=[0, 1, 0]
            ))

            # 5th Term
            self.add_module(f"layer_{i}_W_B_W_B_1", EinsumLayer(
                equation="bdpq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D_a, D],
                fan_in_mask=[0, 1, 1, 1],
                # fan_in_mask=[0, 1, 0, 1],
                unsqueeze_dims=[-2]
            ))

            # 6th Term
            self.add_module(f"layer_{i}_W_B_W_B_2",  EinsumLayer(
                equation="bdjq, edkq -> bejk",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D_a, D],
                fan_in_mask=[0, 1, 0, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 7th Term
            self.add_module(f"layer_{i}_W_B_b_A_1", EinsumLayer(
                equation="bdq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D_a],
                fan_in_mask=[0, 1, 1],
                # fan_in_mask=[0, 1, 0],
                unsqueeze_dims=[-2]
            ))

            # 8th Term
            self.add_module(f"layer_{i}_W_B_b_A_2", EinsumLayer(
                equation="bdj, edk -> bejk",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D_a],
                fan_in_mask=[0, 1, 0]
                # fan_in_mask=[0, 1, 0]
            ))

            # 9th Term
            self.add_module(f"layer_{i}_W_B_b_B", EinsumLayer(
                equation="bdq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D],
                fan_in_mask=[0, 1, 1],
                # fan_in_mask=[0, 1, 0, 1],
                unsqueeze_dims=[-2]
            ))

            # 10th Term (Bias Term)
            self.add_module(f"layer_{i}_W_B_bias", EinsumLayer(
                equation="ek -> ek",
                weight_shape=[self.e, D],
                input_shape=[self.e, D],
                fan_in_mask=[0, 0],
                # fan_in_mask=[0, 0],
                unsqueeze_dims=[0, -2]
            ))
            
            set_init_einsum_(
                            getattr(self, f"layer_{i}_W_B_b_A_2"),
                            getattr(self, f"layer_{i}_W_B_b_B"),
                            getattr(self, f"layer_{i}_W_B_bias"),
                            

                            init_type=init_type)
            set_init_einsum_(getattr(self, f"layer_{i}_W_B_QK_1"),
                            getattr(self, f"layer_{i}_W_B_VO"), init_type=init_type, scale_degree=scale_degree)
            set_init_einsum_(getattr(self, f"layer_{i}_W_B_W_B_1"),
                            getattr(self, f"layer_{i}_W_B_b_A_1"),getattr(self, f"layer_{i}_W_B_W_A_1"),
                            getattr(self, f"layer_{i}_W_B_W_A_2"), getattr(self, f"layer_{i}_W_B_W_B_2"), init_type=init_type, scale_degree=scale_degree)

            # -----------------------------------
            #            b_B Terms
            # -----------------------------------                                   
            # 1st Term
            self.add_module(f"layer_{i}_b_B_QK", EinsumLayer(
                equation="bdhpq, edkpq -> bek",
                weight_shape=[self.e, self.d, D, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1]
                # fan_in_mask=[0, 1, 0, 1, 1]
            ))

            # 2nd Term
            self.add_module(f"layer_{i}_b_B_VO", EinsumLayer(
                equation="bdhpq, edkp -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, h, D, D],
                fan_in_mask=[0, 1, 1, 1, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 3rd Term
            self.add_module(f"layer_{i}_b_B_W_A", EinsumLayer(
                equation="bdpq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D, D_a],
                fan_in_mask=[0, 1, 1, 1]
                # fan_in_mask=[0, 1, 0]
            ))

            # 4th Term
            self.add_module(f"layer_{i}_b_B_W_B", EinsumLayer(
                equation="bdpq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D_a, D],
                fan_in_mask=[0, 1, 1, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 5th Term
            self.add_module(f"layer_{i}_b_B_b_A", EinsumLayer(
                equation="bdq, edk -> bek",
                weight_shape=[self.e, self.d, D],
                input_shape=[-1, self.d, D_a],
                fan_in_mask=[0, 1, 1]
                # fan_in_mask=[0, 1, 0]
            ))

            # 6th Term
            self.add_module(f"layer_{i}_b_B_b_B", EinsumLayer(
                equation="bdq, edkq -> bek",
                weight_shape=[self.e, self.d, D, D],
                input_shape=[-1, self.d, D],
                fan_in_mask=[0, 1, 1]
                # fan_in_mask=[0, 1, 0, 1]
            ))

            # 7th Term: Bias Term as EinsumLayer
            self.add_module(f"layer_{i}_b_B_bias", EinsumLayer(
                equation="ed -> ed",
                weight_shape=[self.e, D],
                input_shape=[self.e, D],
                fan_in_mask=[0, 0],
                # fan_in_mask=[0, 0],
                unsqueeze_dims=[0]
            ))
            set_init_einsum_(                        
                        getattr(self, f"layer_{i}_b_B_b_A"),
                        getattr(self, f"layer_{i}_b_B_b_B"),
                        getattr(self, f"layer_{i}_b_B_bias"),
                        
            init_type=init_type)
            set_init_einsum_(
                        getattr(self, f"layer_{i}_b_B_QK"),
                        getattr(self, f"layer_{i}_b_B_VO"),init_type=init_type,scale_degree=scale_degree)
            set_init_einsum_(
                        getattr(self, f"layer_{i}_b_B_W_B"),
                        getattr(self, f"layer_{i}_b_B_b_A"), 
                        getattr(self, f"layer_{i}_b_B_W_B"),getattr(self, f"layer_{i}_b_B_W_A"),init_type=init_type,scale_degree=scale_degree)


    def forward(self, wsfeat: AttentionWeightSpaceFeatures):
        out_dict = {
            "W_q": [], "W_k": [], "W_v": [], "W_o": [],
            "W_A": [], "W_B": [], "b_A": [], "b_B": []
        }
        
        L = len(wsfeat)  # Number of layers
    
        # Loop over each layer's weights and biases
        for i in range(L):
            W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B = wsfeat[i]

            # Compute intermediate products using einsum equations
            WW_qk = torch.einsum('bdhpk, bdhqk -> bdhpq', W_q, W_k)
            WW_vo = torch.einsum('bdhpk, bdhkq -> bdhpq', W_v, W_o)

            # Apply your EinsumLayers for W_q, W_k, W_v, and W_o
            layer_W_q = getattr(self, f"layer_{i}_W_Q")(W_q)
            layer_W_k = getattr(self, f"layer_{i}_W_K")(W_k)
            layer_W_v = getattr(self, f"layer_{i}_W_V")(W_v)
            layer_W_o_1 = getattr(self, f"layer_{i}_W_O_1")(W_o)
            layer_W_o_2 = getattr(self, f"layer_{i}_W_O_2")(W_o)

            # Apply your EinsumLayers for W_A
            layer_W_A_W_QK = getattr(self, f"layer_{i}_W_A_W_QK")(WW_qk)
            layer_W_A_W_VO_1 = getattr(self, f"layer_{i}_W_A_W_VO_1")(WW_vo)
            layer_W_A_W_VO_2 = getattr(self, f"layer_{i}_W_A_W_VO_2")(WW_vo)
            layer_W_A_W_A_1 = getattr(self, f"layer_{i}_W_A_W_A_1")(W_A)
            layer_W_A_W_A_2 = getattr(self, f"layer_{i}_W_A_W_A_2")(W_A)
            layer_W_A_W_A_3 = getattr(self, f"layer_{i}_W_A_W_A_3")(W_A)
            layer_W_A_W_A_4 = getattr(self, f"layer_{i}_W_A_W_A_4")(W_A)
            layer_W_A_W_B_1 = getattr(self, f"layer_{i}_W_A_W_B_1")(W_B)
            layer_W_A_W_B_2 = getattr(self, f"layer_{i}_W_A_W_B_2")(W_B)
            layer_W_A_b_A_1 = getattr(self, f"layer_{i}_W_A_b_A_1")(b_A)
            layer_W_A_b_A_2 = getattr(self, f"layer_{i}_W_A_b_A_2")(b_A)
            layer_W_A_b_B = getattr(self, f"layer_{i}_W_A_b_B")(b_B)
            layer_W_A_bias = getattr(self, f"layer_{i}_W_A_bias")()

            # Apply your EinsumLayers for b_A
            layer_b_A_QK = getattr(self, f"layer_{i}_b_A_QK")(WW_qk)
            layer_b_A_VO = getattr(self, f"layer_{i}_b_A_VO")(WW_vo)
            layer_b_A_W_A_1 = getattr(self, f"layer_{i}_b_A_W_A_1")(W_A)
            layer_b_A_W_A_2 = getattr(self, f"layer_{i}_b_A_W_A_2")(W_A)
            layer_b_A_W_B_1 = getattr(self, f"layer_{i}_b_A_W_B_1")(W_B)
            layer_b_A_W_B_2 = getattr(self, f"layer_{i}_b_A_W_B_2")(W_B)
            layer_b_A_b_A_1 = getattr(self, f"layer_{i}_b_A_b_A_1")(b_A)
            layer_b_A_b_A_2 = getattr(self, f"layer_{i}_b_A_b_A_2")(b_A)
            layer_b_A_b_B = getattr(self, f"layer_{i}_b_A_b_B")(b_B)
            layer_b_A_bias = getattr(self, f"layer_{i}_b_A_bias")()

            # Apply your EinsumLayers for W_B
            layer_W_B_QK_1 = getattr(self, f"layer_{i}_W_B_QK_1")(WW_qk)
            layer_W_B_VO = getattr(self, f"layer_{i}_W_B_VO")(WW_vo)
            layer_W_B_W_A_1 = getattr(self, f"layer_{i}_W_B_W_A_1")(W_A)
            layer_W_B_W_A_2 = getattr(self, f"layer_{i}_W_B_W_A_2")(W_A)
            layer_W_B_W_B_1 = getattr(self, f"layer_{i}_W_B_W_B_1")(W_B)
            layer_W_B_W_B_2 = getattr(self, f"layer_{i}_W_B_W_B_2")(W_B)
            layer_W_B_b_A_1 = getattr(self, f"layer_{i}_W_B_b_A_1")(b_A)
            layer_W_B_b_A_2 = getattr(self, f"layer_{i}_W_B_b_A_2")(b_A)
            layer_W_B_b_B = getattr(self, f"layer_{i}_W_B_b_B")(b_B)
            layer_W_B_bias = getattr(self, f"layer_{i}_W_B_bias")()

            # Apply your EinsumLayers for b_B
            layer_b_B_QK = getattr(self, f"layer_{i}_b_B_QK")(WW_qk)
            layer_b_B_VO = getattr(self, f"layer_{i}_b_B_VO")(WW_vo)
            layer_b_B_W_A = getattr(self, f"layer_{i}_b_B_W_A")(W_A)
            layer_b_B_W_B = getattr(self, f"layer_{i}_b_B_W_B")(W_B)
            layer_b_B_b_A = getattr(self, f"layer_{i}_b_B_b_A")(b_A)
            layer_b_B_b_B = getattr(self, f"layer_{i}_b_B_b_B")(b_B)
            layer_b_B_bias = getattr(self, f"layer_{i}_b_B_bias")()

            out_dict["W_q"].append(layer_W_q )
            out_dict["W_k"].append(layer_W_k)
            out_dict["W_v"].append(layer_W_v)
            out_dict["W_o"].append(layer_W_o_1 + layer_W_o_2)
            
            out_dict["W_A"].append(
                (layer_W_A_W_QK + layer_W_A_W_VO_1 + layer_W_A_W_VO_2 +
                layer_W_A_W_A_1 + layer_W_A_W_A_2 + layer_W_A_W_A_3 +
                layer_W_A_W_A_4 + layer_W_A_W_B_1 + layer_W_A_W_B_2 +
                layer_W_A_b_A_1 + layer_W_A_b_A_2 + layer_W_A_b_B +
                layer_W_A_bias)
            )
            out_dict["W_B"].append(
                (layer_W_B_QK_1 + layer_W_B_VO + layer_W_B_W_A_1 +
                layer_W_B_W_A_2 + layer_W_B_W_B_1 + layer_W_B_W_B_2 +
                layer_W_B_b_A_1 + layer_W_B_b_A_2 + layer_W_B_b_B +
                layer_W_B_bias)
            )

            out_dict["b_A"].append(
                (layer_b_A_QK + layer_b_A_VO + layer_b_A_W_A_1 +
                layer_b_A_W_A_2 + layer_b_A_W_B_1 + layer_b_A_W_B_2 +
                layer_b_A_b_A_1 + layer_b_A_b_A_2 + layer_b_A_b_B +
                layer_b_A_bias)
            )

            out_dict["b_B"].append(
                (layer_b_B_QK + layer_b_B_VO + layer_b_B_W_A +
                layer_b_B_W_B + layer_b_B_b_A + layer_b_B_b_B +
                layer_b_B_bias)
            )

        return AttentionWeightSpaceFeatures(**out_dict)


class TransformersInv(nn.Module):
    def __init__(self, encoder_weight_spec: AttentionNetworkSpec, in_channels, out_channels, init_type="pytorch_default", scale_degree = 2, layer_norm= True):
        super().__init__()
        self.d, self.e = in_channels, out_channels
        self.encoder_weight_spec = encoder_weight_spec

        D, D_q, D_k, D_v, D_a, h = encoder_weight_spec.get_all_dims()
        self.L = len(encoder_weight_spec)

        for i in range(self.L):
            # Term 1:
            self.add_module(f"layer_{i}_QK",
                            EinsumLayer(
                                equation="bdhpq, edpq -> be",
                                weight_shape=[self.e, self.d, D, D],
                                input_shape=[-1, self.d, h, D, D],
                                fan_in_mask=[0, 1, 1, 1, 1]
                                # fan_in_mask=[0, 1, 1, 1]
                            ))
            
            # Term 2:
            self.add_module(f"layer_{i}_VO",
                            EinsumLayer(
                                equation="bdhpq, edp -> be",
                                weight_shape=[self.e, self.d, D],
                                input_shape=[-1, self.d, h, D, D],
                                fan_in_mask=[0, 1, 1, 1, 1],
                                # fan_in_mask=[0, 1, 1]
                            ))
            
            # Term 3:
            self.add_module(f"layer_{i}_W_A",
                            EinsumLayer(
                                equation="bdpq, ed -> be",
                                weight_shape=[self.e, self.d],
                                input_shape=[-1, self.d, D, D_a],
                                fan_in_mask=[0, 1, 1, 1]
                                # fan_in_mask=[0, 1]
                            ))
            
            # Term 4:
            self.add_module(f"layer_{i}_W_B",
                            EinsumLayer(
                                equation="bdpq, edq -> be",
                                weight_shape=[self.e, self.d, D],
                                input_shape=[-1, self.d, D_a, D],
                                fan_in_mask=[0, 1, 1, 1]
                                # fan_in_mask=[0, 1, 1]
                            ))
            
            # Term 5:
            self.add_module(f"layer_{i}_b_A",
                            EinsumLayer(
                                equation="bdq, ed -> be",
                                weight_shape=[self.e, self.d],
                                input_shape=[-1, self.d, D_a],
                                fan_in_mask=[0, 1, 1]
                                # fan_in_mask=[0, 1]
                            ))
            
            # Term 6:
            self.add_module(f"layer_{i}_b_B",
                            EinsumLayer(
                                equation="bdq, edq -> be",
                                weight_shape=[self.e, self.d, D],
                                input_shape=[-1, 1, 1],
                                fan_in_mask=[0, 1, 1],
                                # fan_in_mask=[0, 1, 1]
                            ))
            
            # Term 7: Bias 
            self.add_module(f"layer_{i}_bias", 
                            EinsumLayer(
                                equation="e -> e",
                                weight_shape=[self.e],
                                input_shape=[self.e],
                                fan_in_mask=[0],
                                # fan_in_mask=[0],
                                unsqueeze_dims=[0]
                            ))
    
            set_init_einsum_(
                getattr(self, f"layer_{i}_b_A"),
                getattr(self, f"layer_{i}_b_B"),
                getattr(self, f"layer_{i}_bias")
                , init_type=init_type)
            set_init_einsum_( getattr(self, f"layer_{i}_QK"),
                getattr(self, f"layer_{i}_VO"), init_type=init_type, scale_degree=scale_degree)
            set_init_einsum_(
                 getattr(self, f"layer_{i}_W_A"),
                getattr(self, f"layer_{i}_W_B"), init_type=init_type, scale_degree=scale_degree)
            
            #TODO: apply for L layer
            self.layer_norm= layer_norm
            if layer_norm:
                self.layer_norm1 = nn.LayerNorm(out_channels)
                self.layer_norm2 = nn.LayerNorm(out_channels)


    def forward(self, wsfeat: AttentionWeightSpaceFeatures):
        out = []

        for i in range(self.L):
            W_q, W_k, W_v, W_o, W_A, W_B, b_A, b_B = wsfeat[i]
            
            WW_qk = torch.einsum('bdhpk, bdhqk -> bdhpq', W_q, W_k)
            WW_vo = torch.einsum('bdhpk, bdhkq -> bdhpq', W_v, W_o)

            layer_QK = getattr(self, f"layer_{i}_QK")(WW_qk)  # Output: [b, e]
            layer_VO = getattr(self, f"layer_{i}_VO")(WW_vo)  # Output: [b, e]
            layer_W_A = getattr(self, f"layer_{i}_W_A")(W_A)  # Output: [b, e]
            layer_W_B = getattr(self, f"layer_{i}_W_B")(W_B)  # Output: [b, e]
            layer_b_A = getattr(self, f"layer_{i}_b_A")(b_A)  # Output: [b, e]
            layer_b_B = getattr(self, f"layer_{i}_b_B")(b_B)  # Output: [b, e]
            layer_bias = getattr(self, f"layer_{i}_bias")()      # Output: [e]

            # Sum all terms to get I(U) for the current layer
            I_U = (
                layer_QK +
                layer_VO +
                layer_W_A +
                layer_W_B +
                layer_b_A +
                layer_b_B +
                layer_bias
            )  # Shape: [b, e]

            out.append(I_U)
        if self.layer_norm:
            out1 = self.layer_norm1(out[0])
            out2 =self.layer_norm2(out[1])
            out = [out1,out2]        
        return rearrange(out, 'L b e -> b (L e)')