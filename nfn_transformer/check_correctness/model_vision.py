import torch
import torch.nn as nn
import math


# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# N -> Number of Patches = IH/P * IW/P
# S -> Sequence Length   = IH/P * IW/P + 1 or N + 1 (extra 1 is of Classification Token)
# Q -> Query Sequence length (equal to S for self-attention)
# K -> Key Sequence length   (equal to S for self-attention)
# V -> Value Sequence length (equal to S for self-attention)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H

def getPositionalEncoding(embed_dim: int, len: int = 5000):
    position = torch.arange(len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(len, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class EmbedLayerVision(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1         = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)                       # Patch Encoding
        # self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True)      # Learnable Positional Embedding
        self.pos_embedding = nn.Parameter(getPositionalEncoding(embed_dim=embed_dim, len=(image_size // patch_size) ** 2), 
                                          requires_grad=False)
        # self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)                                    # Classification Token
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)                                                         # B, C, IH, IW     -> B, E, IH/P, IW/P                Split image into the patches and embed patches
        x = x.reshape([B, x.shape[1], -1])                                        # B, E, IH/P, IW/P -> B, E, (IH/P*IW/P) -> B, E, N    Flattening the patches
        x = x.permute(0, 2, 1)                                                    # B, E, N          -> B, N, E                         Rearrange to put sequence dimension in the middle
        x = x + self.pos_embedding                                                # B, N, E          -> B, N, E                         Add positional embedding
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim          = embed_dim
        self.n_attention_heads  = n_attention_heads
        self.head_embed_dim     = embed_dim // n_attention_heads

        self.queries            = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=False)   # Queries projection
        self.keys               = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=False)   # Keys projection
        self.values             = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=False)   # Values projection
        self.out_projection     = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim, bias=False)   # Out projection

    def forward(self, x):
        b, s, e = x.shape  # Note: In case of self-attention Q, K and V are all equal to S

        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, Q, E      ->  B, Q, (H*HE)  ->  B, Q, H, HE
        xq = xq.permute(0, 2, 1, 3)                                                         # B, Q, H, HE  ->  B, H, Q, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, K, E      ->  B, K, (H*HE)  ->  B, K, H, HE
        xk = xk.permute(0, 2, 1, 3)                                                         # B, K, H, HE  ->  B, H, K, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, V, E      ->  B, V, (H*HE)  ->  B, V, H, HE
        xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  ->  B, H, V, HE


        # Compute Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2)                                                         # B, H, K, HE  ->  B, H, HE, K
        x_attention = torch.matmul(xq, xk)                                                  # B, H, Q, HE  *   B, H, HE, K   ->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  ->  A, B, C, F   if D==E)

        x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

        x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

        x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K  *  B, H, V, HE  ->  B, H, Q, HE     Compute Attention product with Values

        # Format the output
        x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(b, s, e)                                                              # B, Q, H, HE -> B, Q, (H*HE)

        x = self.out_projection(x)                                                          # B, Q,(H*HE) -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attention  = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1   = nn.Dropout(dropout)
        
        self.norm2      = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.fc1        = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.ReLU()
        self.fc2        = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x):
        # x = x + self.dropout1(self.attention(self.norm1(x)))                                # Skip connections
        # x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))           # Skip connections
        x = self.dropout1(self.attention(self.norm1(x)))                                # Skip connections
        x = self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))           # Skip connections
        return x


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # New architectures skip fc1 and activations and directly apply fc2.
        self.fc1        = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()
        self.fc2        = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # x = x[:, 0, :]              # Get CLS token
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        super().__init__()
        self.embedding  = EmbedLayerVision(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder    = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm       = nn.LayerNorm(embed_dim, elementwise_affine=False)                                       # Final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x
