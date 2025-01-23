import os
from random import shuffle

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle


class SmallTransformerZooDataset(Dataset):
    def __init__(self, data_path="vision_data", n_heads=2, split="train", name = 'mnist', load_cache=True, cut_off = 0.1) -> None:
        super().__init__()
        self.split = split
        self.n_heads = n_heads
        self.name = name
        self.cut_off = cut_off
        if load_cache and os.path.exists(os.path.join(data_path, f"{name}_{split}_{self.cut_off}.pt")):
            print("Cache file found! Loading from cache...")
            self.load_from_cache(data_path, name)
        else:
            if load_cache and not os.path.exists(os.path.join(data_path, f"{name}_{split}_{self.cut_off}.pt")):
                print(f"Cache file not found!")
            print(f"Loading from csv files...")
            self.load_from_csv(data_path, name)
            print("Cache file generated! Loading from cache...")
            self.load_from_cache(data_path, name)
        self.len_data = len(self.accuracy)
        print(f"Done loading {self.split} data !")

    def load_from_cache(self, data_path, name):
        data = torch.load(os.path.join(data_path, f"{name}_{self.split}_{self.cut_off}.pt"), weights_only=True)
        self.embedding = data["embedding"]
        self.encoder = data["encoder"]
        self.classifier = data["classifier"]
        self.accuracy = data["accuracy"]
        self.len_data = self.accuracy.shape[0]

    def load_from_csv(self, data_path, name):
        # Load dataset
        data_folder = os.path.join(data_path, name)

        #load accuracy from csv file
        meta_folder = os.path.join(data_path, "metadata")
        model_info =  pd.read_csv(os.path.join(meta_folder, f"{name}.csv"))

        #keep only the model trained for 75 epochs
        check1 = model_info['ckpt_epoch']=='75'
        # model_info = model_info[check1 & check2] #TODO: uncomment this
        if self.cut_off >0:
            if self.name =='mnist':
                check2= model_info['test_top1_accuracy'] >= self.cut_off
            else:
                check2= model_info['test_accuracy'] >= self.cut_off
            check1 = check1&check2

        model_info = model_info[check1]

        file_names_all = model_info['ckpt_file']
        file_names_all = [file_name[len(name)+1:] for file_name in file_names_all]
        len_all = len(file_names_all)
        num_train = int(float(len_all * 0.7))
        num_val = int(float(len_all * 0.15))
        num_test = len_all - num_val - num_train
        # Shuffle before splitting
        shuffle(file_names_all)
        file_names_splitted = {"train": file_names_all[:num_train], "val": file_names_all[num_train:num_train + num_val], "test": file_names_all[num_train + num_val:]}
        splits = ["train", "val", "test"]

        encoder_names = {"queries": {"weight": "W_q"},
                         "keys": {"weight": "W_k"},
                         "values": {"weight": "W_v"},
                         "out_projection": {"weight": "W_o"},

                         "fc1": {"weight": "W_A", "bias": "b_A"},
                         "fc2": {"weight": "W_B", "bias": "b_B"}
                         }
        # Load data into data structures
        for split in splits:

            # Initialize data structures
            embedding = {"weight": [], "bias": []}
            encoder = []
            classifier = {"weight": [], "bias": []}

            accuracy = []

            file_names = file_names_splitted[split]
            len_data = len(file_names)
            all_model_keys = None

            for idx, file in enumerate(file_names):
                if file.split(".")[-1] == "pt":
                    try:
                        checkpoint = torch.load(os.path.join(data_folder, file), map_location=torch.device('cpu'), weights_only=True)
                        # Init empty memory to load weights
                        if all_model_keys is None:
                            all_model_keys = [k.split(".") for k in checkpoint.keys()]

                            for key in all_model_keys:
                                if key[0] in ["embedding", "classifier"] and key[-1] in ["weight", "bias"]:
                                    eval(key[0])[key[-1]].append(torch.empty([len_data, *checkpoint[".".join(key)].shape], device="cpu").unsqueeze(1))
                                elif key[0] == "encoder" and key[-1] in ["weight", "bias"]:
                                    # If encoder layer is not inside the list, then initiate it
                                    if int(key[1]) > len(encoder) - 1:
                                        layer_shape = checkpoint[".".join(key)].shape
                                        if key[-2] in ["queries", "keys", "values"]:
                                            encoder.append({encoder_names[key[-2]][key[-1]]:
                                                                torch.empty([len_data, *checkpoint[".".join(key)].reshape(self.n_heads, layer_shape[0]//self.n_heads, layer_shape[1]).transpose(-1, -2).shape], device="cpu").unsqueeze(1)})
                                        elif key[-2] == "out_projection":
                                            encoder.append({encoder_names[key[-2]][key[-1]]:
                                                                torch.empty([len_data, *checkpoint[".".join(key)].transpose(-1, -2).reshape(self.n_heads, layer_shape[1]//self.n_heads, layer_shape[0]).shape], device="cpu").unsqueeze(1)}).unsqueeze(1)
                                        else:
                                            if key[-1] == "weight":
                                                encoder.append({encoder_names[key[-2]][key[-1]]:
                                                                    torch.empty([len_data, *checkpoint[".".join(key)].transpose(0,1).shape], device="cpu").unsqueeze(1)})
                                            else:
                                                encoder.append({encoder_names[key[-2]][key[-1]]:
                                                                    torch.empty([len_data, *checkpoint[".".join(key)].shape], device="cpu").unsqueeze(1)})
                                    else:
                                        layer_shape = checkpoint[".".join(key)].shape
                                        if key[-2] in ["queries", "keys", "values"]:
                                            encoder[int(key[1])][encoder_names[key[-2]][key[-1]]] = torch.empty([len_data, *checkpoint[".".join(key)].reshape(self.n_heads, layer_shape[0]//self.n_heads, layer_shape[1]).transpose(-1, -2).shape], device="cpu").unsqueeze(1)
                                        elif key[-2] == "out_projection":
                                            encoder[int(key[1])][encoder_names[key[-2]][key[-1]]] = torch.empty([len_data, *checkpoint[".".join(key)].transpose(-1, -2).reshape(self.n_heads, layer_shape[1]//self.n_heads, layer_shape[0]).shape], device="cpu").unsqueeze(1)
                                        else:
                                            if key[-1] == "weight":
                                                encoder[int(key[1])][encoder_names[key[-2]][key[-1]]] = torch.empty([len_data, *checkpoint[".".join(key)].transpose(0,1).shape], device="cpu").unsqueeze(1)
                                            else:
                                                encoder[int(key[1])][encoder_names[key[-2]][key[-1]]] = torch.empty([len_data, *checkpoint[".".join(key)].shape], device="cpu").unsqueeze(1)

                        # Load weight into the correct slide of the data
                        idx_layer_embedding = 0
                        idx_layer_classifier = 0
                        for key in all_model_keys:
                            if key[0] in ["embedding", "classifier", "encoder"] and key[-1] in ["weight", "bias"]:
                                if key[0] == "embedding":
                                    eval(key[0])[key[-1]][idx_layer_embedding // 2][idx] = checkpoint[".".join(key)].unsqueeze(0)
                                    idx_layer_embedding += 1
                                elif key[0] == "classifier":
                                    eval(key[0])[key[-1]][idx_layer_classifier // 2][idx] = checkpoint[".".join(key)].unsqueeze(0)
                                    idx_layer_classifier += 1
                                elif key[0] == "encoder":
                                    layer_shape = checkpoint[".".join(key)].shape
                                    if key[-2] in ["queries", "keys", "values"]:
                                        encoder[int(key[1])][encoder_names[key[-2]][key[-1]]][idx] = checkpoint[".".join(key)].reshape(self.n_heads, layer_shape[0]//self.n_heads, layer_shape[1]).transpose(-1, -2).unsqueeze(0)
                                    elif key[-2] == "out_projection":
                                        encoder[int(key[1])][encoder_names[key[-2]][key[-1]]][idx] = checkpoint[".".join(key)].transpose(0,1).reshape(self.n_heads, layer_shape[1]//self.n_heads, layer_shape[0]).unsqueeze(0)
                                    else:
                                        if key[-1] == "weight":
                                            encoder[int(key[1])][encoder_names[key[-2]][key[-1]]][idx] = checkpoint[".".join(key)].transpose(0,1).unsqueeze(0)
                                        else:
                                            encoder[int(key[1])][encoder_names[key[-2]][key[-1]]][idx] = checkpoint[".".join(key)].unsqueeze(0)
                        #get accuracy with corresponding file name
                        if self.name == "mnist":
                            accuracy.append(model_info[model_info['ckpt_file']==f"{name}/{file}"]['test_top1_accuracy'].values[0])
                        else:
                            accuracy.append(model_info[model_info['ckpt_file']==f"{name}/{file}"]['test_accuracy'].values[0])


                    except Exception as e:
                        raise(e)
                        print(f"Error loading {file} !")
                        continue
            encoder = {k: [block[k] for block in encoder] for k in encoder[0].keys()}
            #make accuracy a tensor of shape [len_data,1]
            accuracy = torch.tensor(accuracy).unsqueeze(1).float()
            torch.save({"embedding": embedding, "encoder": encoder, "classifier": classifier, "accuracy": accuracy},
                   os.path.join(data_path, f"{name}_{split}_{self.cut_off}.pt"))
        print("done")
        # W_q, W_k, W_v, W_o is already splitted by the head, i.e. [batch_size, n_heads, dim_in, dim_out]
        # All weight matrices in the encoder are transpose, because right weight multiply is use, i.e., XW instead of WX !!!

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        embedding = {k: [self.embedding[k][layer][index] for layer in range(len(self.embedding[k]))] for k in self.embedding.keys()}
        classifier = {k: [self.classifier[k][layer][index] for layer in range(len(self.classifier[k]))] for k in self.classifier.keys()}
        # encoder = [{k: block[k][index] for k in block.keys()} for block in self.encoder]
        encoder = {k: [key_block[index] for key_block in self.encoder[k]] for k in self.encoder.keys()}
        accuracy = self.accuracy[index]
        return {"embedding": embedding, "encoder": encoder, "classifier": classifier, "accuracy": accuracy}

    @staticmethod
    def collate_fn(batch):
        embedding = {k: [torch.stack([sample["embedding"][k][layer] for sample in batch], dim=0) for layer in range(len(batch[0]["embedding"][k]))] for k in batch[0]["embedding"].keys()}
        classifier = {k: [torch.stack([sample["classifier"][k][layer] for sample in batch], dim=0) for layer in range(len(batch[0]["classifier"][k]))] for k in batch[0]["classifier"].keys()}
        # encoder = [{k: torch.stack([sample["encoder"][block_idx][k] for sample in batch], dim=0) for k in batch[0]["encoder"][block_idx].keys()} for block_idx in range(len(batch[0]["encoder"]))]
        encoder = {k: [torch.stack([sample["encoder"][k][layer] for sample in batch], dim=0) for layer in range(len(batch[0]["encoder"][k]))] for k in batch[0]["encoder"].keys()}
        accuracy = torch.stack([sample["accuracy"] for sample in batch])
        return {"embedding": embedding, "classifier": classifier, "encoder": encoder, "accuracy": accuracy}

    @staticmethod
    def to_device(batch, device):
        embedding = {k: [batch["embedding"][k][layer].to(device) for layer in range(len(batch["embedding"][k]))] for k in batch["embedding"].keys()}
        classifier = {k: [batch["classifier"][k][layer].to(device) for layer in range(len(batch["classifier"][k]))] for k in batch["classifier"].keys()}
        encoder = {k: [batch["encoder"][k][layer].to(device) for layer in range(len(batch["encoder"][k]))] for k in batch["encoder"].keys()}
        accuracy = batch["accuracy"].to(device)
        return {"embedding": embedding, "encoder": encoder, "classifier": classifier, "accuracy": accuracy}

class SmallTransformerZooDatasetAugmented(SmallTransformerZooDataset):
    def __init__(self, data_path="vision_data", n_heads=2, D_k=16, D_v=16, D_A=64, D=32, 
                 n_encoder_layer=2, split="train", name = 'mnist', load_cache=True, cut_off = 0.1, 
                 augment_factor: int = 3, scale=1.0) -> None:
        super().__init__(data_path=data_path, n_heads=n_heads, split=split, name =name, load_cache=load_cache, cut_off = cut_off)
        self.n_heads = n_heads
        self.D_k = D_k
        self.D_v = D_v
        self.D_A = D_A
        self.D = D
        self.augment_factor = augment_factor

        if augment_factor > 1:

            augment_path = os.path.join(data_path, f"{name}_augment_{split}_{self.cut_off}_augment_factor_{self.augment_factor}_low_{scale_low}_high_{scale_high}.pt")
            print(f"Augment path is: {augment_path}")

            if load_cache == True and os.path.exists(augment_path):
                with open(augment_path, 'rb') as handle:
                    augmented_data = pickle.load(handle)
                print(f"Loaded Augmented data at {augment_path}")
            else:
                print(f"Starting augment data with augment factor (int): {self.augment_factor}")
                
                embedding_augmented = {"weight": [[] for _ in range(len(self.embedding["weight"]))], 
                                    "bias": [[] for _ in range(len(self.embedding["weight"]))]}
                encoder_augmented = {
                                    "W_q": [[] for _ in range(n_encoder_layer)], 
                                    "W_k": [[] for _ in range(n_encoder_layer)], 
                                    "W_v": [[] for _ in range(n_encoder_layer)], 
                                    "W_o": [[] for _ in range(n_encoder_layer)], 
                                    "W_A": [[] for _ in range(n_encoder_layer)],
                                    "b_A": [[] for _ in range(n_encoder_layer)], 
                                    "W_B": [[] for _ in range(n_encoder_layer)],
                                    "b_B": [[] for _ in range(n_encoder_layer)]
                                    }
                classifier_augmented = {"weight": [[] for _ in range(len(self.classifier["weight"]))], 
                            "bias": [[] for _ in range(len(self.classifier["bias"]))]}
                accuracy_augmented = []
                
                # Generate augmented samples
                for idx in range(self.len_data):
                    for i in range(self.augment_factor-1):
                        # Augment encoder
                        g = [self.sample_group_action(scale) for _ in range(n_encoder_layer)]
                        g_encoder_idx = self.apply_group_action_to_wsfeat(self.encoder, idx, g)
                        for layer in range(n_encoder_layer):
                            for key in encoder_augmented.keys():
                                encoder_augmented[key][layer].append(g_encoder_idx[key][layer])
                        # Duplicate embedding
                        if len(self.embedding["weight"]) > 0:
                            for layer in range(len(embedding_augmented["weight"])):
                                for key in embedding_augmented.keys():
                                    embedding_augmented[key][layer].append(self.embedding[key][layer][idx])
                        else:
                            pass
                        # Duplicate classifier
                        if len(self.classifier["weight"]) > 0:
                            for layer in range(len(classifier_augmented["weight"])):
                                for key in classifier_augmented.keys():
                                    classifier_augmented[key][layer].append(self.classifier[key][layer][idx])
                        else:
                            pass
                        # Duplicate accuracy
                        accuracy_augmented.append(self.accuracy[idx])
                        
                
                # Stack augmented encoders
                encoder_augmented = {key: [torch.stack(encoder_augmented[key][layer], dim=0) 
                                        for layer in range(n_encoder_layer)] 
                                        for key in encoder_augmented.keys()}
                # Stack augmented embeddings
                embedding_augmented = {key: [torch.stack(embedding_augmented[key][layer], dim=0) 
                                        for layer in range(len(self.embedding["weight"]))] 
                                        for key in embedding_augmented.keys()}
                # Stack augmented classifiers
                classifier_augmented = {key: [torch.stack(classifier_augmented[key][layer], dim=0) 
                                        for layer in range(len(self.classifier["weight"]))] 
                                        for key in classifier_augmented.keys()}
                # Stack augmented accuracies
                accuracy_augmented = torch.stack(accuracy_augmented, dim=0)
                
                # Stack original and augmented encoders
                encoder_augmented = {key: [torch.concat([self.encoder[key][layer], encoder_augmented[key][layer]], dim=0) 
                                            for layer in range(n_encoder_layer)] 
                                            for key in encoder_augmented.keys()}
                
                # Stack original and augmented embeddings
                if len(self.embedding["weight"]) > 0:
                    embedding_augmented = {key: [torch.concat([self.embedding[key][layer], embedding_augmented[key][layer]], dim=0) 
                                                for layer in range(len(self.embedding["weight"]))] 
                                                for key in embedding_augmented.keys()}
                
                # Stack original and augmented classifiers
                if len(self.classifier["weight"]) > 0:
                    classifier_augmented = {key: [torch.concat([self.classifier[key][layer], classifier_augmented[key][layer]], dim=0) 
                                                for layer in range(len(self.classifier["weight"]))] 
                                                for key in classifier_augmented.keys()}
                
                # Stack original and augmented accuracies
                accuracy_augmented = torch.concat([self.accuracy, accuracy_augmented], dim=0)
                
                print(f"Augment data completed")

                augmented_data = {"encoder": encoder_augmented,
                                "embedding": embedding_augmented,
                                "classifier": classifier_augmented,
                                "accuracy": accuracy_augmented}
                
                with open(augment_path, 'wb') as handle:
                    pickle.dump(augmented_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Augmented data saved at {augment_path}")

            self.encoder = augmented_data["encoder"]
            self.embedding = augmented_data["embedding"]
            self.classifier = augmented_data["classifier"]
            self.accuracy = augmented_data["accuracy"]
            self.len_data = augment_factor * self.len_data
        else:
            print("No augmentation done")

    @staticmethod
    def encoder_dict_to_list(encoder):
        encoder = [{k: encoder[k][layer] for k in encoder.keys()} for layer in range(len(encoder["W_q"]))]
        return encoder

    def sample_group_action(self, scale):
        S_h = torch.randperm(self.n_heads)

        M = scale * (torch.rand(self.n_heads, self.D_k, self.D_k) - 0.5)*2

        N = scale * (torch.rand(self.n_heads, self.D_v, self.D_v) - 0.5)*2

        P_pi_O = torch.eye(self.D)[torch.randperm(self.D)]
        P_pi_A = torch.eye(self.D_A)[torch.randperm(self.D_A)]

        return {"S_h": S_h, "M": M, "N": N, "P_pi_O": P_pi_O, "P_pi_A": P_pi_A}

    @staticmethod
    def apply_group_action_to_wsfeat(encoder_dict, idx, group_actions):
        g_encoder_dict = {}

        # Initialize empty lists for each transformed weight in the output
        for key in ["W_q", "W_k", "W_v", "W_o", "W_A", "W_B", "b_A", "b_B"]:
            g_encoder_dict[key] = []

        # Iterate through each layer and group action
        for layer_idx, group_action in enumerate(group_actions):
            S_h, M, N, P_pi_O, P_pi_A = group_action["S_h"], group_action["M"], group_action["N"], group_action["P_pi_O"], group_action["P_pi_A"]

            # Process each key
            # if key == "W_q":
            g_encoder_dict["W_q"].append(encoder_dict["W_q"][layer_idx][idx][:, S_h] @ M[S_h].transpose(-1, -2))
            # elif key == "W_k":
            g_encoder_dict["W_k"].append(encoder_dict["W_k"][layer_idx][idx][:, S_h] @ torch.inverse(M[S_h]))
            # elif key == "W_v":
            g_encoder_dict["W_v"].append(encoder_dict["W_v"][layer_idx][idx][:, S_h] @ N[S_h])
            # elif key == "W_o":
            g_encoder_dict["W_o"].append(torch.inverse(N[S_h]) @ encoder_dict["W_o"][layer_idx][idx][:, S_h] @ P_pi_O)
            # elif key == "W_A":
            g_encoder_dict["W_A"].append(torch.inverse(P_pi_O) @ encoder_dict["W_A"][layer_idx][idx] @ P_pi_A)
            # elif key == "W_B":
            g_encoder_dict["W_B"].append(torch.inverse(P_pi_A) @ encoder_dict["W_B"][layer_idx][idx])
            # elif key == "b_A":
            g_encoder_dict["b_A"].append(encoder_dict["b_A"][layer_idx][idx] @ P_pi_A)
            # elif key == "b_B":
            g_encoder_dict["b_B"].append(encoder_dict["b_B"][layer_idx][idx])
        return g_encoder_dict




if __name__ == "__main__":

    dataset = SmallTransformerZooDatasetAugmented(data_path="nfn_transformer/dataset/MNIST-Transformers", split="test", name='mnist', load_cache=True, cut_off=0)

    loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)

    for idx, batch in enumerate(loader):
        batch = dataset.to_device(batch, "cpu")
        embedding, classifier, encoder, accuracy = batch["embedding"], batch["classifier"], batch["encoder"], batch["accuracy"]
        # print("test")
