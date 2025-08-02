import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from .utils import get_activation, get_loss


class BasicCTRLLM(nn.Module):
    def __init__(self, 
                 field_dims=None, 
                 embed_dim=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_activation = self.get_output_activation(task="binary_classification")
        self.loss_fn = get_loss(loss="binary_cross_entropy")    
    
    def feature_interaction(self, x_emb):
        raise NotImplementedError

    def forward(self, user_item_ids, token_ids, mask):
        raise NotImplementedError 

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean') 
        return loss

    def count_parameters(self, count_embedding=True):

        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and 'embedding' in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        print(f'total number of parameters: {total_params}')
        return total_params

    def reset_parameters(self):
        def reset_default_params(m):
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(reset_default_params)

    def model_to_device(self):
        self.to(device=self.device)

    def get_output_activation(self, task="binary_classification"):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))
        

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
        
    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)


class FeaturesLinear(torch.nn.Module):
    """
    Linear regression layer for CTR prediction.
    """

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.float64)

    def forward(self, x):
        """
        :param x: B,F
        :return: B,1
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
    

class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        # x: B,F,E
        square_of_sum = torch.sum(x, dim=1) ** 2 # sum of square
        sum_of_square = torch.sum(x ** 2, dim=1) # square of sum 
        ix = square_of_sum - sum_of_square # 
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)  # B,1
        return 0.5 * ix  # B,E


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        """
        :param field_dims: list
        :param embed_dim
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.float64)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        """
        :param x: B,F
        :return: B,F,E
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, cn_layers):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(cn_layers)
        ])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])
    
    def forward(self, x, return_fi=False):
        x0 = x
        fis = []
        for i in range(self.cn_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
            if return_fi:
                fis.append(x)
                
        if return_fi: 
            return x, fis
        else: 
            return x

class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim, cn_layers=3):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])
        

    def forward(self, x, return_fi=False):
        x0 = x
        fis = []
        for i in range(self.cn_layers):
            xw = self.w[i](x)  # B,F*E => B,F*E
            x = x0 * (xw + self.b[i]) + x  
            if return_fi:
                fis.append(x)
                
        if return_fi: 
            return x, fis
        else: 
            return x
    
    
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout=0.2, output_layer=False):
        super().__init__()
        layers = list()
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)

        

class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: B,F,E
        """
        square_of_sum = torch.sum(x, dim=1) ** 2  # B，embed_dim
        sum_of_square = torch.sum(x ** 2, dim=1)  # B，embed_dim
        ix = square_of_sum - sum_of_square  # B,embed_dim
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True) # B，1
        return 0.5 * ix


class GateCrossNetwork(nn.Module):
    def __init__(self, input_dim, cn_layers=3, acti="sig"):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.wg = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

        if acti == "identity":
            self.activation = nn.Identity()
        elif acti == "relu":
            self.activation = nn.ReLU()
        elif acti == "tanh":
            self.activation = nn.Tanh()
        elif acti == "sig":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("You shoule choose the right activation function among (identity, relu, tanh and sig)")

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x)
            xg = self.activation(self.wg[i](x)) 
            x = x0 * (xw  + self.b[i]) * xg  + x 
        return x
    

class MLP_Block(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None, 
                 dropout_rates=0.0,
                 batch_norm=False, 
                 bn_only_once=False, # Set True for inference speed up
                 use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        if batch_norm and bn_only_once:
            dense_layers.append(nn.BatchNorm1d(input_dim))
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm and not bn_only_once:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.mlp(inputs)


class DualFENLayer(nn.Module):
    def __init__(self, field_length, embed_dim, embed_dims=(256, 256, 256), att_size=64, num_heads=8):
        super(DualFENLayer, self).__init__()
        input_dim = field_length * embed_dim  
        self.mlp = MultiLayerPerceptron(input_dim, embed_dims, dropout=0.5, output_layer=False)
        self.multihead = MultiHeadAttentionL(model_dim=embed_dim, dk=att_size, num_heads=num_heads)
        self.trans_vec_size = att_size * num_heads * field_length
        self.trans_vec = nn.Linear(self.trans_vec_size, field_length, bias=False)
        self.trans_bit = nn.Linear(embed_dims[-1], field_length, bias=False)

    def forward(self, x_emb):
        x_con = x_emb.view(x_emb.size(0), -1)  # [B, ?]

        m_bit = self.mlp(x_con)

        x_att2 = self.multihead(x_emb, x_emb, x_emb) 
        m_vec = self.trans_vec(x_att2.view(-1, self.trans_vec_size)) 
        m_bit = self.trans_bit(m_bit)

        x_att = m_bit + m_vec 
        x_emb = x_emb * x_att.unsqueeze(2)
        return x_emb, x_att

class MultiHeadAttentionL(nn.Module):
    def __init__(self, model_dim=256, dk=32, num_heads=16):
        super(MultiHeadAttentionL, self).__init__()

        self.dim_per_head = dk  
        self.num_heads = num_heads

        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.linear_residual = nn.Linear(model_dim, self.dim_per_head * num_heads)


    def _dot_product_attention(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(1, 2)) * scale
        attention = torch.softmax(attention, dim=2)
        attention = torch.dropout(attention, p=0.0, train=self.training)
        context = torch.bmm(attention, v)
        return context, attention

    def forward(self, key0, value0, query0, attn_mask=None):
        batch_size = key0.size(0)

        key = self.linear_k(key0)  # K = UWk [B, 10, 256*16]
        value = self.linear_v(value0)  # Q = UWv [B, 10, 256*16]
        query = self.linear_q(query0)  # V = UWq [B, 10, 256*16]

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attention = self._dot_product_attention(query, key, value, scale)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)  # [B, 10, 256*h]

        residual = self.linear_residual(query0)
        residual = residual.view(batch_size, -1, self.dim_per_head * self.num_heads)  # [B, 10, 256*h]

        output = torch.relu(residual + context)  
        return output