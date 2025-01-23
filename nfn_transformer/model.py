import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from nfn_transformer.common.weight_space import (
    AttentionWeightSpaceFeatures,
    LinearWeightSpaceFeatures,
    NetworkSpec,
)
from nfn_transformer.layers import HNPSLinear, HNPSPool
from nfn_transformer.layers.layers import TransformersInv, TransformersLinear
from nfn_transformer.layers.misc_layers import (
    FlattenWeights,
    FlattenWeightsTransformer,
    StatFeaturizer,
    StatFeaturizerTransformer,
    TupleOp,
    TupleOpTransformer,
)

MODE2LAYER = {
    "HNPS":HNPSLinear
}

# LN_DICT = {
#     "param": ParamLayerNorm,
#     "simple": SimpleLayerNorm,
# }

POOL_DICT = {"HNPS_L1": HNPSPool,
             "HNPS_L2": HNPSPool,
             "HNPS_L2_square": HNPSPool,
             "HNPS_param_mul_L2":HNPSPool,
}

class NormalizingModule(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def set_stats(self, mean_std_stats):
        if self.normalize:
            print("Setting stats")
            weight_stats, bias_stats = mean_std_stats
            for i, (w, b) in enumerate(zip(weight_stats, bias_stats)):
                mean_weights, std_weights = w
                mean_bias, std_bias = b
                # wherever std_weights < 1e-5, set to 1
                std_weights = torch.where(std_weights < 1e-5, torch.ones_like(std_weights), std_weights)
                std_bias = torch.where(std_bias < 1e-5, torch.ones_like(std_bias), std_bias)
                self.register_buffer(f"mean_weights_{i}", mean_weights)
                self.register_buffer(f"std_weights_{i}", std_weights)
                self.register_buffer(f"mean_bias_{i}", mean_bias)
                self.register_buffer(f"std_bias_{i}", std_bias)

    def _normalize(self, params):
        out_weights, out_bias = [], []
        for i, (w, b) in enumerate(params):
            mean_weights_i, std_weights_i = getattr(self, f"mean_weights_{i}"), getattr(self, f"std_weights_{i}")
            mean_bias_i, std_bias_i = getattr(self, f"mean_bias_{i}"), getattr(self, f"std_bias_{i}")
            out_weights.append((w - mean_weights_i) / std_weights_i)
            out_bias.append((b - mean_bias_i) / std_bias_i)
        return LinearWeightSpaceFeatures(out_weights, out_bias)


    def preprocess(self, params):
        if self.normalize:
            params = self._normalize(params)
        return params


class MlpHead(nn.Module):
    def __init__(
        self,
        network_spec,
        in_channels,
        pool_mode = "HNPS_param_mul_L2",
        num_out=1,
        h_size=10,
        dropout=0.0,
        lnorm=False,
        sigmoid=True,
        n_layers=2
    ):
        super().__init__()
        self.sigmoid = sigmoid
        head_layers = []
        if pool_mode in POOL_DICT.keys():
            pool_cls = POOL_DICT[pool_mode]
            if pool_mode.startswith("HNPS"):
                head_layers.extend([pool_cls(network_spec, in_channels,mode_pooling = pool_mode[5:]), nn.Flatten(start_dim=-2)])
            num_pooled_outs = in_channels * pool_cls.get_num_outs(network_spec)
        else:
            num_pooled_outs = in_channels
        head_layers.append(nn.Linear(num_pooled_outs, h_size))
        for i in range(2):
            if lnorm:
                head_layers.append(nn.LayerNorm(h_size))
            head_layers.append(nn.ReLU())
            if dropout > 0:
                head_layers.append(nn.Dropout(p=dropout))
            head_layers.append(nn.Linear(h_size, h_size if i == 0 else num_out))
        if sigmoid:
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.head(x)


class InvariantNFN(NormalizingModule):
    """Invariant hypernetwork. Outputs a scalar."""
    def __init__(
        self,
        embedding_network_spec,
        classifier_network_spec,
        encoder_network_spec,
        classifier_nfn_channels = [10,10],
        transformers_nfn_channels = [10,10],
        classifier_model_mode="HNPS",
        feature_dropout=0,
        normalize=False,
        in_channels=1,
        init_type="xavier_normal",
        classifier_pool_mode="HNPS_param_mul_L2",
        num_out_classify = 10,
        num_out_embedding = 10,
        num_out_encoder = 5,
        hnps_init = "xavier_normal",
        enc_mode = 'inv',
        cls_mode = 'hnps',
        emb_mode = 'mlp'
    ):
        super().__init__(normalize=normalize)
        self.classifier_layers = []
        self.encoder_layers = []
        self.embedding_layers = []
        self.final_classifier_head = []
        self.enc_mode = enc_mode
        self.cls_mode = cls_mode
        self.emb_mode = emb_mode

        final_mlp_inp = 0
        #Inv-nfn/StatNN/MLP for encoder
        if self.enc_mode != 'no':
            if self.enc_mode == "statnn":
                self.encoder_layers.append(StatFeaturizerTransformer())
                self.encoder_layers.append(nn.Flatten(start_dim=-2))

                prev_channels = StatFeaturizerTransformer.get_num_outs(encoder_network_spec)
                for num_channels in transformers_nfn_channels:
                    self.encoder_layers.append(nn.Linear(prev_channels, num_channels))
                    self.encoder_layers.append(nn.ReLU())
                    prev_channels = num_channels

                self.encoder_layers.append(nn.Linear(prev_channels, num_out_encoder))
                self.encoder_layers.append(nn.ReLU())

            elif self.enc_mode == "mlp":
                self.encoder_layers.append(FlattenWeightsTransformer(encoder_network_spec))
                self.encoder_layers.append(nn.Flatten(start_dim=-2))

                prev_channels = encoder_network_spec.get_num_params()
                for num_channels in transformers_nfn_channels:
                    self.encoder_layers.append(nn.Linear(prev_channels, num_channels))
                    self.encoder_layers.append(nn.ReLU())
                    prev_channels = num_channels

                self.encoder_layers.append(nn.Linear(prev_channels, num_out_encoder))
                self.encoder_layers.append(nn.ReLU())

            elif self.enc_mode == 'inv':
                prev_channels = in_channels
                for num_channels in transformers_nfn_channels:
                    self.encoder_layers.append(TransformersLinear(encoder_network_spec, in_channels=prev_channels, out_channels=num_channels, init_type=init_type))
                    self.encoder_layers.append(TupleOpTransformer(nn.ReLU(), masked_features=['W_q', 'W_k', 'W_v', 'W_o']))
                    prev_channels = num_channels
                self.encoder_layers.append(TransformersInv(encoder_network_spec, in_channels=prev_channels, out_channels=prev_channels, init_type=init_type))
                self.encoder_layers.append(nn.LayerNorm(prev_channels*len(encoder_network_spec)))
                self.encoder_layers.append(MlpHead(network_spec=encoder_network_spec, in_channels=prev_channels*len(encoder_network_spec), num_out=num_out_encoder, pool_mode=None, sigmoid=False, lnorm=True))
            else:
                assert False

            self.encoder_layers = nn.Sequential(*self.encoder_layers)
            final_mlp_inp += num_out_encoder

        #Inv nfn for classifier
        if self.cls_mode != 'no':
            if self.cls_mode == "mlp":
                self.classifier_layers.append(FlattenWeights(classifier_network_spec))
                self.classifier_layers.append(nn.Flatten(start_dim=-2))

                prev_channels = classifier_network_spec.get_num_params()
                for num_channels in classifier_nfn_channels:
                    self.classifier_layers.append(nn.Linear(prev_channels, num_channels))
                    self.classifier_layers.append(nn.ReLU())
                    prev_channels = num_channels

                self.classifier_layers.append(nn.Linear(prev_channels, num_out_classify))
                self.classifier_layers.append(nn.ReLU())


            elif self.cls_mode == "statnn":
                self.classifier_layers.append(StatFeaturizer())
                self.classifier_layers.append(nn.Flatten(start_dim=-2))

                prev_channels = StatFeaturizer.get_num_outs(classifier_network_spec)
                for num_channels in transformers_nfn_channels:
                    self.classifier_layers.append(nn.Linear(prev_channels, num_channels))
                    self.classifier_layers.append(nn.ReLU())
                    prev_channels = num_channels

                self.classifier_layers.append(nn.Linear(prev_channels, num_out_classify))
                self.classifier_layers.append(nn.ReLU())
            elif self.cls_mode == 'hnps':
                assert len(classifier_network_spec) > 1
                prev_channels = in_channels
                for num_channels in classifier_nfn_channels:
                    self.classifier_layers.append(MODE2LAYER[classifier_model_mode](classifier_network_spec, in_channels=prev_channels, out_channels=num_channels,init_type=hnps_init))
                    prev_channels = num_channels
                    self.classifier_layers.append(TupleOp(nn.ReLU()))

                self.classifier_layers.append(MlpHead(network_spec=classifier_network_spec, in_channels=prev_channels, pool_mode=classifier_pool_mode, num_out=num_out_classify, sigmoid=False, lnorm=True))
            else:
                assert False

            self.classifier_layers = nn.Sequential(*self.classifier_layers)
            final_mlp_inp += num_out_classify

        #MLP for embedding
        if self.emb_mode != 'no':
            assert self.emb_mode == 'mlp'

            embedding_weight_shape = embedding_network_spec.get_matrices_shape()[0]
            self.embedding_layers.append(nn.Linear(embedding_network_spec.get_num_params(), num_out_embedding))
            #self.embedding_layers.append(nn.LayerNorm(num_out_embedding))

            self.embedding_layers = nn.Sequential(*self.embedding_layers)
            final_mlp_inp += num_out_embedding

        #Final head
        self.final_classifier_head.append(nn.Linear(final_mlp_inp, 1))
        self.final_classifier_head.append(nn.Sigmoid())
        self.final_classifier_head=nn.Sequential(*self.final_classifier_head)
    def forward(self, embedding, classifier, encoder):
        features = []
        if self.enc_mode != 'no':
            for (i,layer) in enumerate(self.encoder_layers):
                encoder = layer(encoder)
            features.append(encoder)

        if self.cls_mode != 'no':
            for layer in self.classifier_layers:
                classifier = layer(classifier)
            features.append(classifier)

        if self.emb_mode != 'no':
            embedding = torch.cat([torch.cat([torch.flatten(embedding['weight'][i],start_dim=1) for i in range(len(embedding['weight']))], dim=1),\
                                  torch.cat([torch.flatten(embedding['bias'][i],start_dim=1) for i in range(len(embedding['bias']))],dim =1)], dim=1)
            embedding = embedding.cuda()
            for layer in self.embedding_layers:
                embedding = layer(embedding)
            features.append(embedding)
        features = torch.cat(features, dim=-1)

        for layer in self.final_classifier_head:
            features = layer(features)

        return (features)
