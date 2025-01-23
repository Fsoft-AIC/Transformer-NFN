import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle
from embedding_utils import word2vec_as_vocab
from tqdm import tqdm

def tokenize_dataset(dataset, vocab):
    # Tokenization function
    def tokenizer(text):
        return text.split()  # A simple whitespace-based tokenizer

    tokenized_dataset = []
    for sentence in dataset:
        tokens = tokenizer(sentence)
        indices = [vocab.get(token, 0) for token in tokens]  # Use 0 if token is not in vocab
        tokenized_dataset.append(torch.tensor(indices, dtype=torch.long))
    return tokenized_dataset

def read_agnews(split):
    if split == 'train':
        df = pd.read_csv(os.path.join('data', 'ag_news', 'train.csv'))
    elif split == 'test':
        df = pd.read_csv(os.path.join('data', 'ag_news', 'test.csv'))
    else:
        raise ValueError(f"Unknown split: {split}")
    df['Class Index'] = df['Class Index'].astype(int)
    return df


class AGNewsTokenizedDataset(Dataset):
    def __init__(self, split='train', fraction=1.0):
        datafile_name = os.path.join('data', 'ag_news', f'{split}_tokenized.pkl')
        if not os.path.isfile(datafile_name):
            data = read_agnews(split=split)
            vocab = word2vec_as_vocab('reduced_w2v_32d.bin')

            # Tokenize and convert to token IDs
            print(f'Tokenizing {split} data')
            # self.tokenized_data = [(torch.tensor([vocab.get(token, 0)
            #                                         for token in self.tokenizer(text)],
            #                                         dtype=torch.int64).cuda(), label-1)
            #                         for label, text in tqdm(data)]
            # create tokenized data by df.apply
            self.tokenized_data = data.apply(
                lambda x: (
                    torch.tensor(
                        [vocab.get(token, 0) for token in self.tokenizer(x['Description'])], 
                        dtype=torch.int64
                    ), 
                    x['Class Index'] - 1
                ), 
                axis=1
            ).tolist()
            # Save to a file
            with open(datafile_name, 'wb') as f:
                pickle.dump(self.tokenized_data, f)
        else:
            with open(datafile_name, 'rb') as f:
                self.tokenized_data = pickle.load(f)
        
        self.tokenized_data = self.tokenized_data[:int(fraction * len(self.tokenized_data))]

    def tokenizer(self, text):
        return text.split()  # A simple whitespace-based tokenizer

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]
