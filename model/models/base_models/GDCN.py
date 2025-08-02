
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicLayer import BasicCTRLLM, FeaturesLinear, FeaturesEmbedding, GateCrossNetwork, MultiLayerPerceptron
from torch import Tensor

class GCN(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 cross_layers=2, mlp_layers=(400,400,400), 
                 dropout=0.5, *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.embed_dim = embed_dim
        
        self.feature_interaction_encoder = GateCrossNetwork(self.embed_output_dim, cross_layers)
        
        self.pred_layer = torch.nn.Linear(self.embed_output_dim, 1)
    
    def forward(self, x):
        x_embed = self.embedding(x).view(-1, self.embed_output_dim)
        cross_cn = self.feature_interaction_encoder(x_embed)
        y_pred = self.pred_layer(cross_cn)
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict

class GDCN(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 cross_layers=3, mlp_layers=(400,400,400), 
                 dropout=0.5, type="parallel", *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.embed_dim = embed_dim
        
        self.feature_interaction_encoder = GateCrossNetwork(self.embed_output_dim, 
                                                        cross_layers)
        
        self.prediction_layer = MultiLayerPerceptron(self.embed_output_dim, 
                                                     mlp_dims = mlp_layers,
                                                     output_layer=False,
                                                     dropout=dropout)
        if type == "stack":
            self.output_layer = nn.Linear(mlp_layers[-1], 1)
        elif type == "parallel":
            self.output_layer = nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)
        else:
            raise ValueError("Type are not in Stack or Parallel.")
        
    
    def feature_interaction(self, x_emb: list[Tensor]):
        x_flatten = x_emb.reshape(-1, self.embed_output_dim) 
        cross_cn = self.feature_interaction_encoder(x_flatten)
        mlp_cn = self.prediction_layer(x_flatten) # B, M
        cross_cat = torch.cat([cross_cn, mlp_cn], dim=1)
        return cross_cat
    
    
    def forward(self, user_item_ids):
        x_emb = self.embedding(user_item_ids) # B, F, E 
        x_fi = self.feature_interaction(x_emb) # B, F*E
        y_pred = self.output_layer(x_fi) # B, 1
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict

class GCN_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 cross_layers=3, mlp_layers=(256, 256, 256), 
                 dropout=0.5, type="parallel"):
        super().__init__(field_dims, embed_dim)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.feature_interaction_layers = cross_layers
        
        self.feature_interaction_encoder = GateCrossNetwork(self.embed_output_dim, 
                                                        cross_layers)
    
        self.output_layer = nn.Linear(self.embed_output_dim, 1)     
           
        self.tabular_feature_dimension = self.embed_output_dim

    
    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
        
    def feature_interaction(self, x_emb: list[Tensor]):
        x_flatten = x_emb.reshape(-1, self.embed_output_dim) 
        x_fi = self.feature_interaction_encoder(x_flatten)
        return x_fi, x_fi
    
    def compute_logits(self, x_fi):
        y_pred = self.output_layer(x_fi)
        return y_pred
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B, F, E 
        x_fi,_ = self.feature_interaction(x_emb) # B, F*E
        y_pred = self.compute_logits(x_fi)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict

class GDCN_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, cross_layers=3, 
                 mlp_layers=(256, 256, 256), 
                 dropout=0.5, type="parallel"):
        super().__init__(field_dims, embed_dim)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.feature_interaction_layers = cross_layers
        
        self.feature_interaction_encoder = GateCrossNetwork(self.embed_output_dim, 
                                                        cross_layers)
        
        self.mlp_encoder = MultiLayerPerceptron(self.embed_output_dim, 
                                                     mlp_dims = mlp_layers,
                                                     output_layer=False,
                                                     dropout=dropout)

        
        self.output_layer = nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)        
        self.tabular_feature_dimension = self.embed_output_dim

    
    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
        
    def feature_interaction(self, x_emb: list[Tensor]):
        x_flatten = x_emb.reshape(-1, self.embed_output_dim) 
        x_fi = self.feature_interaction_encoder(x_flatten)
        cross_mlp = self.mlp_encoder(x_flatten)
        x_fi_cat = torch.cat([x_fi, cross_mlp], dim=1) 
        return x_fi, x_fi_cat
    
    def compute_logits(self, x_fi):
        y_pred = self.output_layer(x_fi)
        return y_pred
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B, F, E 
        x_fi, x_fi_cat = self.feature_interaction(x_emb) # B, F*E
        y_pred = self.compute_logits(x_fi_cat)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict