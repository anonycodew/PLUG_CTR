
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicLayer import BasicCTRLLM, FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicCTRLLM
from torch import Tensor

class FCN_PLUG(BasicCTRLLM):
    def __init__(self,
                 field_dims, embed_dim=16,
                 num_deep_cross_layers=3,
                 num_shallow_cross_layers=3,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim)
        
        self.embedding_layer = MultiHeadFeatureEmbedding(field_dims, embed_dim * num_heads, num_heads)
        
        input_dim = embed_dim*len(field_dims)
        self.embed_output_dim = input_dim
        self.ECN = ExponentialCrossNetwork(input_dim=input_dim,
                                           num_cross_layers=num_deep_cross_layers,
                                           net_dropout=deep_net_dropout,
                                           layer_norm=layer_norm,
                                           batch_norm=batch_norm,
                                           num_heads=num_heads)
        self.LCN = LinearCrossNetwork(input_dim=input_dim,
                                      num_cross_layers=num_shallow_cross_layers,
                                      net_dropout=shallow_net_dropout,
                                      layer_norm=layer_norm,
                                      batch_norm=batch_norm,
                                      num_heads=num_heads)
        self.loss_fn = torch.nn.BCELoss()    

        self.tabular_feature_dimension = input_dim * 2
            

    
    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding_layer(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
        
    def feature_interaction(self, x_emb: list[Tensor]):
        # 这里就是交互的特征
        x_flatten = x_emb.reshape(-1, self.embed_output_dim) 
        x_ecn = self.ECN(x_flatten, return_fi=True)
        x_lcn = self.LCN(x_flatten, return_fi=True)
        x_fi_cat = torch.cat([x_ecn, x_lcn], dim=1) 
        return x_fi_cat, x_fi_cat
    
    def compute_logits(self, x_fi):
        dlogit = self.ECN(x_fi).mean(dim=1)
        slogit = self.LCN(x_fi).mean(dim=1)
        y_pred = (dlogit + slogit) * 0.5
        return y_pred
    
    def forward(self, user_item_ids, return_features=False):
        x_emb = self.feature_embedding(user_item_ids) # B, F, E 
        x_fi, x_fi_cat = self.feature_interaction(x_emb) # B, F*E
        y_pred = self.compute_logits(x_emb)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi_cat}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict


class FCN(BasicCTRLLM):
    def __init__(self,
                 field_dims, 
                 embed_dim=16,
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.embedding_layer = MultiHeadFeatureEmbedding(field_dims, embed_dim * num_heads, num_heads)
        
        input_dim = embed_dim*len(field_dims)
        self.ECN = ExponentialCrossNetwork(input_dim=input_dim,
                                           num_cross_layers=num_deep_cross_layers,
                                           net_dropout=deep_net_dropout,
                                           layer_norm=layer_norm,
                                           batch_norm=batch_norm,
                                           num_heads=num_heads)
        self.LCN = LinearCrossNetwork(input_dim=input_dim,
                                      num_cross_layers=num_shallow_cross_layers,
                                      net_dropout=shallow_net_dropout,
                                      layer_norm=layer_norm,
                                      batch_norm=batch_norm,
                                      num_heads=num_heads)
        self.loss_fn = torch.nn.BCELoss()

        
    def forward(self, user_item_ids):
        feature_emb = self.embedding_layer(user_item_ids)
        dlogit = self.ECN(feature_emb).mean(dim=1)
        slogit = self.LCN(feature_emb).mean(dim=1)
        y_pred = (dlogit + slogit) * 0.5
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict


    def forward2(self, user_item_ids):

        feature_emb = self.embedding_layer(user_item_ids)
        dlogit = self.ECN(feature_emb).mean(dim=1)
        slogit = self.LCN(feature_emb).mean(dim=1)
        logit = (dlogit + slogit) * 0.5
        y_pred = torch.sigmoid(logit).squeeze(1)
        return_dict = {"y_pred": y_pred,
                       "y_d": torch.sigmoid(dlogit).squeeze(1),
                       "y_s": torch.sigmoid(slogit).squeeze(1)}
        return return_dict

    def add_loss(self, user_item_ids, y_true):
        return_dict = self.forward2(user_item_ids)
        y_pred = return_dict["y_pred"]
        y_d = return_dict["y_d"]
        y_s = return_dict["y_s"]
        loss = self.loss_fn(y_pred, y_true)
        loss_d = self.loss_fn(y_d, y_true)
        loss_s = self.loss_fn(y_s, y_true)
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros(1).to(weight_d.device))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros(1).to(weight_s.device))
        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss


class MultiHeadFeatureEmbedding(nn.Module):
    def __init__(self, field_nums, embedding_dim, num_heads=1):
        super(MultiHeadFeatureEmbedding, self).__init__()
        self.num_heads = num_heads
        self.embedding = FeaturesEmbedding(field_nums, embedding_dim)

    def forward(self, X):  # H = num_heads
        feature_emb = self.embedding(X)  # B × F × D
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(multihead_feature_emb, 2,
                                                                            dim=-1)  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = multihead_feature_emb1.flatten(start_dim=2), \
                                                         multihead_feature_emb2.flatten(
                                                             start_dim=2)  # B × H × FD/2H; B × H × FD/2H
        multihead_feature_emb = torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)
        return multihead_feature_emb  # B × H × FD/H


class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=False,
                 net_dropout=0.1,
                 num_heads=1):
        super(ExponentialCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.dfc = nn.Linear(input_dim, 1)

    def forward(self, x, return_fi = False):
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.dfc(x)
        if return_fi:
            return x 
        return logit


class LinearCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1,
                 num_heads=1):
        super(LinearCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.sfc = nn.Linear(input_dim, 1)

    def forward(self, x, return_fi = False):
        x0 = x
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x0 * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.sfc(x)
        if return_fi:
            return x 
        return logit