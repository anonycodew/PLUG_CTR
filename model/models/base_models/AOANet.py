import torch
from torch import nn
import torch.nn.functional as F
from model.BasicLayer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicCTRLLM, MLP_Block

class AOANet_sp(BasicCTRLLM):
    def __init__(self, 
                 field_dims, embed_dim=16, 
                 dnn_hidden_units=[256,256,256],
                 dnn_hidden_activations="ReLU",
                 num_interaction_layers=2,
                 num_subspaces=6,
                 net_dropout=0.2,
                 batch_norm=True,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        embed_dim_ = int(32 / 2)
        self.embedding_1 = FeaturesEmbedding(field_dims, embed_dim_)
        self.embedding_2 = FeaturesEmbedding(field_dims, embed_dim_)
        num_fields=len(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim_
        self.dnn = MLP_Block(input_dim=num_fields*embed_dim_,
                             output_dim=None, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.gin = GeneralizedInteractionNet(num_interaction_layers, 
                                             num_subspaces, 
                                             num_fields, 
                                             embed_dim_)
        self.fc = nn.Linear(dnn_hidden_units[-1] + num_subspaces * embed_dim_, 1)

            
    def forward(self, input_ids):
        x_embed1 = self.embedding_1(input_ids) # B,F*E/2
        x_embed2 = self.embedding_2(input_ids) # B,F*E/2
        dnn_out = self.dnn(x_embed1.flatten(start_dim=1))
        interact_out = self.gin(x_embed2).flatten(start_dim=1)
        y_pred = self.fc(torch.cat([dnn_out, interact_out], dim=-1))

        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict


class AOANet(BasicCTRLLM):
    """ The AOANet model
        References:
          - Lang Lang, Zhenlong Zhu, Xuanye Liu, Jianxin Zhao, Jixing Xu, Minghui Shan: 
            Architecture and Operation Adaptive Network for Online Recommendations, KDD 2021.
          - [PDF] https://dl.acm.org/doi/pdf/10.1145/3447548.3467133
    """
    def __init__(self, 
                 field_dims, embed_dim=16, 
                 dnn_hidden_units=[400, 400, 400],
                 dnn_hidden_activations="ReLU",
                 num_interaction_layers=3,
                 num_subspaces=4,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields=len(field_dims)
        self.dnn = MLP_Block(input_dim=num_fields*embed_dim,
                             output_dim=None, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.gin = GeneralizedInteractionNet(num_interaction_layers, 
                                             num_subspaces, 
                                             num_fields, 
                                             embed_dim)
        self.fc = nn.Linear(dnn_hidden_units[-1] + num_subspaces * embed_dim, 1)

            
    def forward(self, input_ids):

        feat_emb = self.embedding(input_ids)
        dnn_out = self.dnn(feat_emb.flatten(start_dim=1))
        interact_out = self.gin(feat_emb).flatten(start_dim=1)
        y_pred = self.fc(torch.cat([dnn_out, interact_out], dim=-1))
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict


class GeneralizedInteractionNet(nn.Module):
    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = nn.ModuleList([GeneralizedInteraction(num_fields if i == 0 else num_subspaces, 
                                                           num_subspaces, 
                                                           num_fields, 
                                                           embedding_dim) \
                                     for i in range(num_layers)])
    
    def forward(self, B_0):
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i
            

class GeneralizedInteraction(nn.Module):
    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.eye(embedding_dim, embedding_dim).unsqueeze(0).repeat(output_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(input_subspaces * num_fields, output_subspaces))
        self.h = nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        outer_product = torch.einsum("bnh,bnd->bnhd",
                                     B_0.repeat(1, self.input_subspaces, 1), 
                                     B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1, self.embedding_dim)) # b x (field*in) x d x d 
        fusion = torch.matmul(outer_product.permute(0, 2, 3, 1), self.alpha) # b x d x d x out
        fusion = self.W * fusion.permute(0, 3, 1, 2) # b x out x d x d
        B_i = torch.matmul(fusion, self.h).squeeze(-1) # b x out x d
        return B_i
