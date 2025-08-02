import torch
import torch.nn as nn
import torch.nn.functional as F 

import math 

from model.BasicLayer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicCTRLLM

class AFN(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 LNN_dim=20, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400),
                 dropouts=(0.4, 0.4), *args, **kwargs):
        super(AFN, self).__init__(field_dims, embed_dim, *args, **kwargs)
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.LNN = LNN(self.num_fields, embed_dim,  LNN_dim)
        
        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0],output_layer=True)


    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
            
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B,F,E 
        
        lnn_out = self.LNN(x_emb)
        y_pred = self.mlp(lnn_out)
    
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict

class AFNP(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, LNN_dim=20, mlp_dims=(400, 400, 400),
                 mlp_dims2=(400, 400, 400),
                 dropouts=(0.4, 0.4), *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.LNN = LNN(self.num_fields, embed_dim,  LNN_dim)
        
        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0],output_layer=True)
        
        self.tabular_feature_dimension = self.LNN_output_dim
        
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_dims2, dropouts[1], output_layer=True)

    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
            
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B,F,E 
        
        lnn_out = self.LNN(x_emb)
        x_lnn = self.mlp(lnn_out)
        x_dnn = self.mlp2(x_emb.view(-1, self.embed_output_dim))
        y_pred = x_dnn + x_lnn
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict

class AFNP_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, LNN_dim=20, mlp_layers=(400, 400, 400),
                 dropouts=(0.5, 0.5), *args, **kwargs):
        super(AFNP, self).__init__(field_dims, embed_dim, *args, **kwargs)
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim

        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.LNN = LNN(self.num_fields, embed_dim,  LNN_dim)
        
        self.mlp_encoder = MultiLayerPerceptron(self.LNN_output_dim, mlp_layers, 
                                        dropouts[0], output_layer=True)
        
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp2 = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, dropouts[1], 
                                         output_layer=True)
        
        self.tabular_feature_dimension = self.LNN_output_dim



    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
    
    def feature_interaction(self, x_emb:torch.Tensor):
        x_fi = self.LNN(x_emb)
        return x_fi, x_fi

    def compute_logits(self, x_fi, user_item_ids=None):
        y_pred = self.mlp_encoder(x_fi)
        return y_pred
            
    
    def forward(self, user_item_ids, return_features=True):
        x_emb = self.feature_embedding(user_item_ids)  
        x_fi, _  = self.feature_interaction(x_emb)
        y_pred = self.compute_logits(x_fi) + self.mlp2(x_emb.view(-1, self.embed_output_dim))
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict

class AFN_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, LNN_dim=20, mlp_layers=(400, 400, 400),
                 dropouts=(0.5, 0.5), *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim

        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.LNN = LNN(self.num_fields, embed_dim,  LNN_dim)
        
        self.mlp_encoder = MultiLayerPerceptron(self.LNN_output_dim, mlp_layers, 
                                        dropouts[0], output_layer=True)
        
        self.tabular_feature_dimension = self.LNN_output_dim


    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
    
    def feature_interaction(self, x_emb:torch.Tensor):
        x_fi = self.LNN(x_emb)
        return x_fi, x_fi

    def compute_logits(self, x_fi, user_item_ids=None):
        y_pred = self.mlp_encoder(x_fi)
        return y_pred
            
    
    def forward(self, user_item_ids, return_features=True):
        x_emb = self.feature_embedding(user_item_ids) # B,F,E 
        x_fi, _  = self.feature_interaction(x_emb)
        y_pred = self.compute_logits(x_fi)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict

class LNN(torch.nn.Module):
    def __init__(self, num_fields, embed_dim, LNN_dim, bias=False):
        super(LNN, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.lnn_output_dim = LNN_dim * embed_dim

        self.weight = torch.nn.Parameter(torch.Tensor(LNN_dim, num_fields))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(LNN_dim, embed_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields, embedding_size)``
        """
        embed_x_abs = torch.abs(x)  # Computes the element-wise absolute value of the given input tensor.
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        # Logarithmic Transformation
        embed_x_log = torch.log1p(embed_x_afn)  # torch.log1p
        lnn_out = torch.matmul(self.weight, embed_x_log)
        if self.bias is not None:
            lnn_out += self.bias

        # and torch.expm1
        lnn_exp = torch.expm1(lnn_out)
        output = F.relu(lnn_exp).contiguous().view(-1, self.lnn_output_dim)
        return output