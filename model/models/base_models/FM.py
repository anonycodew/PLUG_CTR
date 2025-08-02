import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicLayer import BasicCTRLLM, FeaturesLinear, FeaturesEmbedding, FactorizationMachine
from torch import Tensor

class FM(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim,
                 *args, **kwargs): 
        super(FM, self).__init__(field_dims, embed_dim, *args, **kwargs)
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        y_pred = self.lr(x) + self.fm(x_emb)

        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict

class LR(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=None,*args, **kwargs):
        super(LR, self).__init__(field_dims, embed_dim, *args, **kwargs)
        self.lr = FeaturesLinear(field_dims)

    def forward(self, x):
        y_pred = self.lr(x)

        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict