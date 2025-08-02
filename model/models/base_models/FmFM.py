import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicLayer import BasicCTRLLM, FeaturesLinear, FeaturesEmbedding, FactorizationMachine
from torch import Tensor


class FmFM(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, interaction_type="matrix", *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.lr = FeaturesLinear(field_dims)
        self.num_field = len(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.inter_num = self.num_field * (self.num_field - 1) // 2
        self.field_interaction_type = interaction_type
        if self.field_interaction_type == "vector":  
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim))
        elif self.field_interaction_type == "matrix": 
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim, embed_dim))
            
        nn.init.xavier_uniform_(self.interaction_weight.data)
        self.row, self.col = list(), list()
        for i in range(self.num_field - 1):
            for j in range(i + 1, self.num_field):
                self.row.append(i), self.col.append(j)

    def forward(self, x):

        x_emb = self.embedding(x)
        left_emb = x_emb[:, self.row]
        right_emb = x_emb[:, self.col]
        if self.field_interaction_type == "vector":
            left_emb = left_emb * self.interaction_weight 
        elif self.field_interaction_type == "matrix":
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        y_pred = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict