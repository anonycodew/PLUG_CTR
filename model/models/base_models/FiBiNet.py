import torch
import torch.nn as nn
import torch.nn.functional as F 

from model.BasicLayer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron, BasicCTRLLM
import itertools

class FiBiNet(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 mlp_layers=(400, 400, 400), dropout=0.5, 
                 bilinear_type="interaction", *args, ** kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        num_fields = len(field_dims)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        
        self.senet = SenetLayer(num_fields)

        self.bilinear = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)
        self.bilinear2 = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)


        num_inter = num_fields * (num_fields - 1) // 2
        self.embed_output_size = num_inter * embed_dim
        self.mlp = MultiLayerPerceptron(2 * self.embed_output_size, mlp_layers, 
                                        dropout=dropout, output_layer=True)
        
    def feature_interaction(self, x_emb):

        x_senet, x_weight = self.senet(x_emb)
        x_bi1 = self.bilinear(x_emb)
        x_bi2 = self.bilinear2(x_senet)
        x_fi = torch.cat([x_bi1.view(x_emb.size(0), -1),
                           x_bi2.view(x_emb.size(0), -1)], dim=1)
        return  x_fi
        
    def forward(self, user_item_ids):
        x_emb = self.embedding(user_item_ids)
        x_fi = self.feature_interaction(x_emb)
        y_pred = self.mlp(x_fi)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict

class FiBiNet_PLUG(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, 
                 mlp_layers=(400, 400, 400), dropout=0.5, 
                 bilinear_type="each", *args, ** kwargs):
        super(FiBiNet, self).__init__(field_dims, embed_dim, *args, **kwargs)
        num_fields = len(field_dims)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        
        self.senet = SenetLayer(num_fields)

        self.bilinear = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)
        self.bilinear2 = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)


        num_inter = num_fields * (num_fields - 1) // 2
        self.embed_output_size = num_inter * embed_dim
        self.mlp = MultiLayerPerceptron(2 * self.embed_output_size, mlp_layers, 
                                        dropout=dropout,output_layer=True)
        
        self.tabular_feature_dimension = 2 * self.embed_output_size


    def feature_embedding(self, user_item_ids, if_flatten=False):
        return self.embedding(user_item_ids) if if_flatten==False else self.embedding(user_item_ids).flatten(start_dim=1)
    
        
    def feature_interaction(self, x_emb):
        x_senet, x_weight = self.senet(x_emb)
        x_bi1 = self.bilinear(x_emb)
        x_bi2 = self.bilinear2(x_senet)

        x_fi = torch.cat([x_bi1.view(x_emb.size(0), -1),
                           x_bi2.view(x_emb.size(0), -1)], dim=1)
        return  x_fi, x_fi

    def compute_logits(self, x_fi):
        return self.mlp(x_fi)
        
    def forward(self, user_item_ids, return_features=True):
        x_emb = self.feature_embedding(user_item_ids)
        x_fi = self.feature_interaction(x_emb)
        y_pred = self.compute_logits(x_fi)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        if return_features:
            return_dict = {"y_pred": y_pred,
                           "x_fi": x_fi}
        else:
            return_dict = {"y_pred": y_pred}
        return return_dict
    

class SenetLayer(nn.Module):
    def __init__(self, field_length, ratio=1):
        super(SenetLayer, self).__init__()
        self.temp_dim = max(1, field_length // ratio)
        self.excitation = nn.Sequential(
            nn.Linear(field_length, self.temp_dim),
            nn.ReLU(),
            nn.Linear(self.temp_dim, field_length),
            nn.ReLU()
        )

    def forward(self, x_emb):
        Z_mean = torch.max(x_emb, dim=2, keepdim=True)[0].transpose(1, 2)
        # Z_mean = torch.mean(x_emb, dim=2, keepdim=True).transpose(1, 2)
        A_weight = self.excitation(Z_mean).transpose(1, 2)
        V_embed = torch.mul(A_weight, x_emb)
        return V_embed, A_weight
    
class BilinearInteractionLayer(nn.Module):
    def __init__(self, filed_size, embedding_size, bilinear_type="each"):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        self.bilinear = nn.ModuleList()

        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(
                embedding_size, embedding_size, bias=False)

        elif self.bilinear_type == "each":
            for i in range(filed_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))

        elif self.bilinear_type == "interaction":
            for i, j in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError
        # self.to(device)


    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]

        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]

        elif self.bilinear_type == "interaction":
            # hardmard
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)
        