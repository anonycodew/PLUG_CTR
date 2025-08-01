import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicLayer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, BasicCTRLLM, MLP_Block

class CIN(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, mlp_dims=(400,400,400), dropout=0.5,
                 cross_layer_sizes=(100, 100), split_half=False, *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)
        y_pred = self.cin(x_emb)
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict
    

class CompressedInteractionNetwork(nn.Module):
    """
    xDeepFM - CIN
    """

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))

class xDeepFM(BasicCTRLLM):
    def __init__(self, field_dims, embed_dim=16, mlp_dims=(400, 400, 400), dropout=0.5,
                 cross_layer_sizes=(50, 50), split_half=True, *args, **kwargs):
        super().__init__(field_dims, embed_dim, *args, **kwargs)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=True)
        # self.linear = FeaturesLinear(field_dims)

    def forward(self, input_ids):
        x_emb = self.embedding(input_ids)

        cin_term = self.cin(x_emb)
        
        mlp_term = self.mlp(x_emb.view(-1, self.embed_output_dim))

        y_pred =  cin_term + mlp_term
        
        y_pred = self.output_activation(y_pred).squeeze(1)
        return_dict = {"y_pred": y_pred}
        return return_dict