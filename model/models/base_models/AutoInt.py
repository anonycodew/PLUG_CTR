import torch
import torch.nn.functional as F
import torch.nn as nn

from model.BasicLayer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicCTRLLM


class AutoIntP(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 atten_embed_dim=16, num_heads=1,
                 attention_layers=3, mlp_dims=(400, 400, 400), 
                 dropouts=(0.3,0.3), has_residual=True, prediction_type = "linear",
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_fields = len(field_dims)
        self.lr_layer = FeaturesLinear(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, 
                                        mlp_dims, dropouts[0], 
                                        output_layer=True)

        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embed_dim if i == 0 else atten_embed_dim,
                                     attention_dim=atten_embed_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=dropouts[1], 
                                     use_residual=False, 
                                     use_scale=False,
                                     layer_norm=False) \
             for i in range(attention_layers)])
        
        self.fc = nn.Linear(len(field_dims) * atten_embed_dim, 1)
        

        
    def forward(self, input_ids):
        x_emb = self.embedding(input_ids)
        attention_out = self.self_attention(x_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        
        y_att = self.fc(attention_out)
        y_dnn = self.mlp(x_emb.flatten(start_dim=1))
        y_pred = self.lr_layer(input_ids) +  y_dnn + y_att
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict


class AutoInt(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 atten_embed_dim=16, num_heads=2,
                 attention_layers=3, mlp_dims=(400, 400, 400), 
                 dropouts=(0.5,0.5), has_residual=True, prediction_type = "linear",
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_fields = len(field_dims)
        self.lr_layer = FeaturesLinear(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim

        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embed_dim if i == 0 else atten_embed_dim,
                                     attention_dim=atten_embed_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=dropouts[1], 
                                     use_residual=True, 
                                     use_scale=False,
                                     layer_norm=False) \
             for i in range(attention_layers)])
        
        self.fc = nn.Linear(len(field_dims) * atten_embed_dim, 1)
        

        
    def forward(self, input_ids):
        x_emb = self.embedding(input_ids)
        attention_out = self.self_attention(x_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        
        y_att = self.fc(attention_out)
        y_pred = self.lr_layer(input_ids) + y_att
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict


class AutoIntP_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 atten_embed_dim=16, num_heads=2,
                 attention_layers=3, mlp_dims=(400, 400, 400), 
                 dropouts=(0.5,0.5), has_residual=True, prediction_type = "linear",
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_fields = len(field_dims)
        self.lr_layer = FeaturesLinear(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, 
                                        mlp_dims, dropouts[0], 
                                        output_layer=True)
        
        self.tabular_feature_dimension = self.embed_output_dim
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embed_dim if i == 0 else atten_embed_dim,
                                     attention_dim=atten_embed_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=dropouts[1], 
                                     use_residual=True, 
                                     use_scale=False,
                                     layer_norm=False) \
             for i in range(attention_layers)])
        
        if prediction_type == "linear":
            self.output_layer = nn.Linear(len(field_dims) * atten_embed_dim, 1)
        elif prediction_type == "mlp":
            self.output_layer =  MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[0], output_layer=True)
   
    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
        
    def feature_interaction(self, x_emb: list[torch.Tensor]):
        x_fi = self.self_attention(x_emb)
        x_fi = torch.flatten(x_fi, start_dim=1) 
        return x_fi, x_fi
    
    def compute_logits(self, x_fi, user_item_ids=None):
        y_pred = self.output_layer(x_fi) # B, M
        return y_pred
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B, F, E 
        x_fi, _ = self.feature_interaction(x_emb) # B, F*E
        y_pred = self.compute_logits(x_fi) + self.mlp(x_emb.flatten(start_dim=1)) + self.lr_layer(user_item_ids)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict


class AutoInt_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 atten_embed_dim=16, num_heads=2,
                 attention_layers=3, mlp_dims=(400, 400, 400), 
                 dropouts=(0.5,0.5), has_residual=True, prediction_type = "linear",
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_fields = len(field_dims)
        self.lr_layer = FeaturesLinear(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, 
                                        mlp_dims, dropouts[0], 
                                        output_layer=True)
        
        self.tabular_feature_dimension = self.embed_output_dim
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embed_dim if i == 0 else atten_embed_dim,
                                     attention_dim=atten_embed_dim, 
                                     num_heads=num_heads, 
                                     dropout_rate=dropouts[1], 
                                     use_residual=True, 
                                     use_scale=False,
                                     layer_norm=False) \
             for i in range(attention_layers)])
        
        if prediction_type == "linear":
            self.output_layer = nn.Linear(len(field_dims) * atten_embed_dim, 1)
        elif prediction_type == "mlp":
            self.output_layer =  MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[0], output_layer=True)
   
    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
        
    def feature_interaction(self, x_emb: list[torch.Tensor]):
        x_fi = self.self_attention(x_emb)
        x_fi = torch.flatten(x_fi, start_dim=1) 
        return x_fi, x_fi
    
    def compute_logits(self, x_fi):
        y_pred = self.output_layer(x_fi) # B, M
        return y_pred
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B, F, E 
        x_fi, _ = self.feature_interaction(x_emb) # B, F*E
        y_pred = self.compute_logits(x_fi)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict
    

class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        residual = X
        
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention
    
        