from model_vision import VisionTransformer
import copy
import torch
import random
import numpy as np
import os

def set_seed(manualSeed=3):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

class Checkpoint_WeightSpace_Converter():
    def __init__(self) -> None:
        self.statedict_to_weightspace_names = {"queries": {"weight": "W_q"}, 
                            "keys": {"weight": "W_k"}, 
                            "values": {"weight": "W_v"}, 
                            "out_projection": {"weight": "W_o"},
                            
                            "fc1": {"weight": "W_A", "bias": "b_A"},
                            "fc2": {"weight": "W_B", "bias": "b_B"}
                            }

    def convert_state_dict_to_weight_space(self, checkpoint, n_encoder_layer, n_heads):
        encoder = [{} for i in range(n_encoder_layer)]
        all_model_keys = [k.split(".") for k in checkpoint.keys()]

        for key in all_model_keys:
            if key[0] == "encoder" and key[-1] in ["weight", "bias"]:
                layer_shape = checkpoint[".".join(key)].shape
                if key[-2] in ["queries", "keys", "values"]:
                    encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]] = checkpoint[".".join(key)].reshape(n_heads, layer_shape[0]//n_heads, layer_shape[1]).transpose(-1, -2)
                elif key[-2] == "out_projection":
                    encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]] = checkpoint[".".join(key)].transpose(0,1).reshape(n_heads, layer_shape[1]//n_heads, layer_shape[0])
                else:
                    if key[-1] == "weight":
                        encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]] = checkpoint[".".join(key)].transpose(0,1)
                    else:
                        encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]] = checkpoint[".".join(key)]
        return encoder
    
    def convert_weight_space_to_state_dict(self, encoder, checkpoint):
        all_model_keys = [k.split(".") for k in checkpoint.keys()]
        
        for key in all_model_keys:
            if key[0] == "encoder" and key[-1] in ["weight", "bias"]:
                layer_shape = checkpoint[".".join(key)].shape
                if key[-2] in ["queries", "keys", "values"]:
                    checkpoint[".".join(key)] = encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]].transpose(-1, -2).reshape(layer_shape[0], layer_shape[1])
                elif key[-2] == "out_projection":
                    checkpoint[".".join(key)] = encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]].reshape(layer_shape[1], layer_shape[0]).transpose(-1, -2)
                else:
                    if key[-1] == "weight":
                        checkpoint[".".join(key)] = encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]].transpose(0,1)
                    else:
                        checkpoint[".".join(key)] = encoder[int(key[1])][self.statedict_to_weightspace_names[key[-2]][key[-1]]]
        
        return checkpoint

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

def apply_group_action(encoders, group_actions):
    g_encoders = []
    for encoder, group_action in zip(encoders, group_actions):
        S_h, M, N, P_pi_O, P_pi_A = group_action["S_h"], group_action["M"], group_action["N"], group_action["P_pi_O"], group_action["P_pi_A"]
        g_encoder = {}
        for key in encoder.keys():
            if key == "W_q":
                g_encoder["W_q"] = encoder["W_q"][S_h] @ M[S_h].transpose(-1, -2)
            elif key == "W_k":
                g_encoder["W_k"] = encoder["W_k"][S_h] @ torch.inverse(M[S_h])
            elif key == "W_v":
                g_encoder["W_v"] = encoder["W_v"][S_h] @ N[S_h]
            elif key == "W_o":
                g_encoder["W_o"] = torch.inverse(N[S_h]) @ encoder["W_o"][S_h] @ P_pi_O
            elif key == "W_A":
                g_encoder["W_A"] = torch.inverse(P_pi_O) @ encoder["W_A"] @ P_pi_A
            elif key == "W_B":
                g_encoder["W_B"] = torch.inverse(P_pi_A) @ encoder["W_B"]
            elif key == "b_A":
                g_encoder["b_A"] = encoder["b_A"] @ P_pi_A
            elif key == "b_B":
                g_encoder["b_B"] = encoder["b_B"]
            # else:
            #     g_encoder[key] = encoder[key]
        g_encoders.append(g_encoder)
    return g_encoders

def check_group_action():
    embed_dim = 32
    n_layers = 2
    n_heads = 2
    forward_mul = 2
    image_size = 28
    n_channels = 1
    patch_size = 4
    bsz = 1024
    D_k = embed_dim // n_heads
    D_v = embed_dim // n_heads
    D_A = embed_dim * forward_mul
    D = embed_dim


    model_1 = VisionTransformer(n_channels=n_channels, embed_dim=embed_dim, n_layers=n_layers, n_attention_heads=n_heads, 
                                forward_mul=forward_mul, image_size=image_size,
                              patch_size=patch_size, n_classes=10, dropout=0.0)
    checkpoint_1 = model_1.state_dict()

    converter = Checkpoint_WeightSpace_Converter()
    encoders_ws = converter.convert_state_dict_to_weight_space(checkpoint_1, n_layers, n_heads)

    checkpoint_2 = copy.deepcopy(model_1.state_dict())
    group_actions = [sample_group_action(n_heads, D_k, D_v, D_A, D) for _ in range(n_layers)]
    g_encoders_ws = apply_group_action(encoders_ws, group_actions)

    checkpoint_2 = converter.convert_weight_space_to_state_dict(g_encoders_ws, checkpoint_2)
    model_2 = VisionTransformer(n_channels=n_channels, embed_dim=embed_dim, n_layers=n_layers, n_attention_heads=n_heads, 
                                forward_mul=forward_mul, image_size=image_size,
                              patch_size=patch_size, n_classes=10, dropout=0.0)
    model_2.load_state_dict(checkpoint_2)

    batch = torch.rand([bsz, n_channels, image_size, image_size])

    result_1 = model_1(batch)
    result_2 = model_2(batch)
    num_correct = sum([torch.allclose(result_1[idx], result_2[idx], rtol=1e-4) for idx in range(result_1.shape[0])])
    print(f"Group action correct for: {num_correct}/{result_1.shape[0]} samples")

if __name__ == "__main__":
    set_seed(6)
    check_group_action()
