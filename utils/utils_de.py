import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os  
import numpy as np

def load_trained_embedding(from_model,to_model):
    model_dict = to_model.state_dict()
    state_dict_trained = {name: param for name, param in from_model.named_parameters() if name in model_dict.keys()}
    model_dict.update(state_dict_trained)
    to_model.load_state_dict(model_dict)
    return to_model


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


def seed_everything(seed=2025):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True