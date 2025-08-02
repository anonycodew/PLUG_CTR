import torch
import torch.nn as nn
import torch.nn.functional as F 
from model.BasicLayer import FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear, BasicCTRLLM, MLP_Block, FeaturesLinear
from itertools import combinations


class HFM(BasicCTRLLM):
    def __init__(self, 
                 field_dims, embed_dim=16, 
                 interaction_type="circular_convolution",
                 use_dnn=True,
                 hidden_units=[64, 64],
                 hidden_activations=["relu", "relu"],
                 batch_norm=False,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 *args, **kwargs):
        super(HFM, self).__init__(field_dims, embed_dim,*args, **kwargs)
        self.lr_layer = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.hfm_layer = HolographicInteraction(len(field_dims), interaction_type=interaction_type)
        self.use_dnn = use_dnn
        if self.use_dnn:
            input_dim = int(len(field_dims)* (len(field_dims) - 1) / 2) * embed_dim
            self.dnn = MLP_Block(input_dim=input_dim,
                                 output_dim=1, 
                                 hidden_units=hidden_units,
                                 hidden_activations=hidden_activations,
                                 output_activation=None,
                                 dropout_rates=net_dropout, 
                                 batch_norm=batch_norm)
        else:
            self.proj_h = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, input_ids):
        feature_emb = self.embedding(input_ids)
        interact_out = self.hfm_layer(feature_emb)
        if self.use_dnn:
            hfm_out = self.dnn(torch.flatten(interact_out, start_dim=1))
        else:
            hfm_out = self.proj_h(interact_out.sum(dim=1))
        y_pred = hfm_out + self.lr_layer(input_ids)

        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict



class HolographicInteraction(nn.Module):
    def __init__(self, num_fields, interaction_type="circular_convolution"):
        super(HolographicInteraction, self).__init__()
        self.interaction_type = interaction_type
        if self.interaction_type == "circular_correlation":
            self.conj_sign =  nn.Parameter(torch.tensor([1., -1.]), requires_grad=False)
        self.triu_index = nn.Parameter(torch.triu_indices(num_fields, num_fields, offset=1), requires_grad=False)

    def forward(self, feature_emb):
        emb1 =  torch.index_select(feature_emb, 1, self.triu_index[0])
        emb2 = torch.index_select(feature_emb, 1, self.triu_index[1])
        if self.interaction_type == "hadamard_product":
            interact_tensor = emb1 * emb2
        elif self.interaction_type == "circular_convolution":
            fft1 = torch.view_as_real(torch.fft.fft(emb1))
            fft2 = torch.view_as_real(torch.fft.fft(emb2))
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(fft_product)))[..., 0]
        elif self.interaction_type == "circular_correlation":
            fft1_emb = torch.view_as_real(torch.fft.fft(emb1))
            fft1 = fft1_emb * self.conj_sign.expand_as(fft1_emb)
            fft2 = torch.view_as_real(torch.fft.fft(emb2))
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(fft_product)))[..., 0]
        else:
            raise ValueError("interaction_type={} not supported.".format(self.interaction_type))
        return interact_tensor
