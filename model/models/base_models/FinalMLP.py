
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicLayer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicCTRLLM, MLP_Block

class FinalMLP(BasicCTRLLM):
    def __init__(self, 
                 field_dims,
                 embed_dim,
                 mlp1_hidden_units=[256, 256],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[256, 256],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0,
                 mlp2_batch_norm=False,
                 use_fs=False,
                 fs_hidden_units=[256],
                 fs1_context=[],
                 fs2_context=[],
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim,*args, **kwargs)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        feature_dim = embed_dim * len(field_dims)
        self.mlp1 = MLP_Block(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp1_dropout,
                              batch_norm=mlp1_batch_norm)
        self.mlp2 = MLP_Block(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp2_dropout, 
                              batch_norm=mlp2_batch_norm)
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(field_dims, 
                                              embed_dim, 
                                              feature_dim, 
                                              embed_dim, 
                                              fs_hidden_units, 
                                              len(field_dims))
        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
            
    def forward(self, input_ids):
        """
        Inputs: [X,y]
        """
        flat_emb = self.embedding(input_ids).flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(input_ids, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))

        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict


class FeatureSelection(nn.Module):
    def __init__(self, field_dims, embed_dim, feature_dim, embedding_dim, fs_hidden_units=[], 
                 num_fields=0):
        super(FeatureSelection, self).__init__()
        self.fs1_ctx_emb = FeaturesEmbedding(field_dims, embed_dim)

        self.fs2_ctx_emb = FeaturesEmbedding(field_dims, embed_dim)
        self.fs1_gate = MLP_Block(input_dim=embedding_dim * num_fields,
                                    output_dim=feature_dim,
                                    hidden_units=fs_hidden_units,
                                    hidden_activations="ReLU",
                                    output_activation="Sigmoid",
                                    batch_norm=False)
                                    
        self.fs2_gate = MLP_Block(input_dim=embedding_dim * num_fields,
                                    output_dim=feature_dim,
                                    hidden_units=fs_hidden_units,
                                    hidden_activations="ReLU",
                                    output_activation="Sigmoid",
                                    batch_norm=False)

    def forward(self, input_ids, flat_emb):
        fs1_input = self.fs1_ctx_emb(input_ids).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        
        fs2_input = self.fs2_ctx_emb(input_ids).flatten(start_dim=1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output
