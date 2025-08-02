import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.BasicLayer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicCTRLLM, FactorizationMachine


class DeepFM(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim,  mlp_dims=(400, 400, 400), 
                 dropout=0.5,*args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        
        self.embed_output_dim = len(field_dims) * embed_dim
        
        self.mlp = MultiLayerPerceptron(input_dim=self.embed_output_dim,
                                        mlp_dims=mlp_dims,
                                        dropout=dropout,
                                        output_layer=False)

        self.output_layer = nn.Linear(mlp_dims[-1], 1)
    
    def feature_interaction(self, x_emb):
        x_fi = x_emb.flatten(start_dim=1)
        x_fi = self.mlp(x_fi)
        return x_fi
    
    def forward(self, user_item_ids):
        x_emb = self.embedding(user_item_ids) # B, F, E 
        x_fi = self.feature_interaction(x_emb) # B, mlps[-1]
        y_pred = self.output_layer(x_fi) + self.fm(x_emb) # B, 1
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict

class DeepFM_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim,  mlp_dims=(400, 400, 400), 
                 dropout=0.5,*args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        
        self.tabular_feature_dimension = mlp_dims[-1]
        
        self.mlp = MultiLayerPerceptron(input_dim=self.embed_output_dim,
                                        mlp_dims=mlp_dims,
                                        dropout=dropout,
                                        output_layer=False)

        self.prediction_layer = nn.Linear(mlp_dims[-1], 1)
    
    
    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
        
    def feature_interaction(self, x_emb: list[Tensor]):
        x_flatten = x_emb.reshape(-1, self.embed_output_dim) 
        cross_cn = self.mlp(x_flatten)
        return cross_cn
    
    def compute_logits(self, x_fi):
        y_pred = self.prediction_layer(x_fi)
        return y_pred
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B, F, E 
        x_fi = self.feature_interaction(x_emb) # B, F*E
        y_pred = self.compute_logits(x_fi) + self.fm(x_emb)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred, 
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict