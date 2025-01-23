import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import os
import gzip
import shutil
import gdown


def save_word2vec_text_format(words, embeddings, filename):
    with open(filename, 'w') as f:
        # Write the number of words and dimensions
        f.write(f"{len(words)} {embeddings.shape[1]}\n")
        
        for word, embedding in zip(words, embeddings):
            # Write the word and its embedding
            f.write(f"{word} {' '.join(map(str, embedding))}\n")

def create_small_embeddings(name, dim=32):
    print(f'Creating {dim}d embedding')
    if not os.path.exists(os.path.join('data', 'word2vec')):
        os.mkdir(os.path.join('data', 'word2vec'))

    w2v_file_path = os.path.join('data', 'word2vec', 'GoogleNews-vectors-negative300.bin')
    if not os.path.isfile(w2v_file_path):
        ggdrive_fileid = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'
        url = f'https://drive.google.com/uc?id={ggdrive_fileid}'
        # Download the file
        w2v_file_path_gz = w2v_file_path.replace('.bin', '.bin.gz')
        gdown.download(url, w2v_file_path_gz, quiet=False)
        with gzip.open(w2v_file_path_gz, 'rb') as f_in:
            with open(w2v_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(w2v_file_path_gz)
    model = KeyedVectors.load_word2vec_format(w2v_file_path, binary=True)
    # Retrieve words and embeddings
    words = list(model.key_to_index.keys())
    embeddings = np.array([model[word] for word in words])

    # Apply PCA
    n_components = dim  # Number of dimensions you want
    pca = PCA(n_components=n_components, random_state=27)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Save the reduced embeddings
    save_word2vec_text_format(words, reduced_embeddings, name)


def word2vec_as_vocab(word2vec_name):
    # Load Word2Vec using Gensim
    w2v_path = os.path.join('data', 'word2vec', word2vec_name)
    if not os.path.isfile(w2v_path):
        create_small_embeddings(w2v_path)
    word2vec = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

    # Prepare a vocabulary mapping from word to index
    vocab = {word: idx + 1 for idx, word in enumerate(word2vec.index_to_key)}  # +1 to reserve 0 for padding
    return vocab

# Function to load gensim Word2Vec and create torch embedding layer
def load_word2vec_as_embedding_layer(word2vec_path):
    # Load Word2Vec using Gensim
    if not os.path.isfile(word2vec_path):
        word2vecpath_bin = word2vec_path.replace('.pt', '.bin')
        create_small_embeddings(word2vecpath_bin)
        word2vec = KeyedVectors.load_word2vec_format(word2vecpath_bin, binary=False)
        
        # Get dimensions of embeddings
        embedding_dim = word2vec.vector_size
        
        # Prepare a vocabulary mapping from word to index
        vocab = {word: idx + 1 for idx, word in enumerate(word2vec.index_to_key)}  # +1 to reserve 0 for padding
        
        # Initialize embeddings matrix (with an extra row for padding token)
        embeddings_matrix = np.zeros((len(vocab) + 1, embedding_dim))
        
        for word, idx in vocab.items():
            embeddings_matrix[idx] = word2vec[word]
        
        # Create a PyTorch embedding layer
        embeddings_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings_matrix), freeze=True)
        
        torch.save(embeddings_layer, word2vec_path)
    else:
        # load the embeddings layer 
        embeddings_layer = torch.load(word2vec_path, weights_only=True)
        # freeze the embeddings
        embeddings_layer.weight.requires_grad = False        
    
    return embeddings_layer
