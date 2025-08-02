from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import math 


from model import models
from model.models import * 
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os 
from alldatasets import *


llm_root = "./llm_download/modelscope/"
llm_model_path =  {
    "qwen_3B":          llm_root + "Qwen/Qwen2.5-3B-Instruct",
    "ds_qwen_1_5B":     llm_root + "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen_1_5B":        llm_root + "Qwen/Qwen2.5-1.5B-Instruct",
    "llama_1B":         llm_root + "Llama/Llama-3.2-1B-Instruct",
    "llama_3B":         llm_root + "Llama/Llama-3.2-3B-Instruct",
    "rebotera_base":    llm_root + "bert/rebotera-base",
    "bert_large":       llm_root + "bert/bert-large",
    "qwen_emb":         llm_root + "embedding/Qwen/Qwen3-Embedding-0.6B",
    "m3e":              llm_root + "embedding/M3E/m3e_large",
    "bge":              llm_root + "embedding/BGE/bge_m3",
    "tiny_bert":        llm_root + "embedding/TinyBert",
}


def get_ctr_for_llm(model_name, field_dims, config):
    if model_name == "DCNv2_PLUG":
        return DCNv2_PLUG(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)
        
    elif model_name == "DCNv1_PLUG":
        return DCNv1_PLUG(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)

    elif model_name == "DeepFM_PLUG":
        return DeepFM_PLUG(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)

    elif model_name == "WDL_PLUG":
        return WDL_PLUG(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)
        
    elif model_name == "AFN_PLUG":
        return AFN_PLUG(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)
        
    elif model_name == "FNN_PLUG":
        return FNN_PLUG(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)
    
    elif model_name == "DCNv2":
        return DCNv2(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)

    elif model_name == "DCNv1":
        return DCNv1(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)
    
    elif model_name == "xDeepFM":
        return xDeepFM(field_dims=field_dims, embed_dim=config.embed_dim, 
                       cross_layers=config.cross_layers)
    else:
        try:
            model_ctr = getattr(models, "FINAL")
            return model_ctr(field_dims=field_dims, embed_dim=config.embed_dim)  
        except ValueError:
            raise ValueError('unknown model name: ' + model_name)


def get_plug_model(llm_ctr_name, ctr_model_name, base_llm_name, field_dims, config):
    
    llm_path = llm_model_path[base_llm_name]
    assert os.path.exists(llm_path), f"model not exists in: {llm_path}"
    llm_config = AutoConfig.from_pretrained(llm_path)
    if llm_ctr_name == "PLUG":     
        ctr_model = get_ctr_for_llm(ctr_model_name, field_dims, config)
        return PLUG_Framework(ctr_model= ctr_model, llm_config = llm_config,if_norm=config.norm, 
                          tao=config.tao, projection_dim=config.reduce_dimension)  


def get_file_path(dataset_name, llm_name, layer_num="last"):
    path_root = f"./alldatasets/semantic_hdf5"
    path_dataset = f"/{dataset_name}_hdf5_embeddings/"
    file_name = f"semantic_representations_{llm_name}_{layer_num}_mean_pool.npy"
    final_path = path_root + path_dataset + file_name
    assert os.path.exists(final_path), f"File not exists: {final_path}"
    return final_path

def get_dataset(dataset_name, batch_size=4096, num_workers=4, tokenizer=None, type="hdf5", 
                llm_model=None, npy_layer="last" ):
    if dataset_name == "ml1m":
        npy_file_path =  get_file_path(dataset_name, llm_model, str(npy_layer))
        field_dims, trainLoader, validLoader, testLoader = ml1m_dataloader_train(batch_size=batch_size, 
                                                                                     type=type, file_path=npy_file_path)
        return field_dims, trainLoader, validLoader, testLoader
    elif dataset_name == "frappe":
        npy_file_path =  get_file_path(dataset_name, llm_model, str(npy_layer))
        field_dims, trainLoader, validLoader, testLoader = frappe_dataloader_train(batch_size=batch_size, 
                                                                                     type=type, file_path=npy_file_path)
        return field_dims, trainLoader, validLoader, testLoader
    else:
        raise ValueError('unknown dataset name: ' + dataset_name)



@dataclass
class CTRModelArguments:
    bit_layers: int = field(default=1)
    mlp_layer: int=field(default=256)
    att_size: int = field(default=32)
    fusion: str= field(default="add")
    if_ln: int = field(default=0)
    cn_layers :int =field(default=3)
    ctr_model_name: str = field(default="fm")
    embed_size: int = field(default=32)
    type_name: str = field(default="deemb")
    norm: str = field(default="norm")
    embed_dropout_rate: float = field(default=0.0)
    gcn_layers :int = field(default=4)
    cn_layers :int =field(default=3)
    hidden_size: int = field(default=128)
    num_hidden_layers: int = field(default=1)
    bridge_type: str = field(default="gate")
    hidden_act: str = field(default='relu')
    hidden_dropout_rate: float = field(default=0.0)
    num_attn_heads: int = field(default=1)
    attn_probs_dropout_rate: float = field(default=0.1)
    intermediate_size: int = field(default=128)
    norm_first: bool = field(default=False)
    layer_norm_eps: float = field(default=1e-12)
    res_conn: bool = field(default=False)
    output_dim: int = field(default=1)
    num_cross_layers: int = field(default=1)
    share_embedding: bool = field(default=False)
    channels: str = field(default='14,16,18,20')
    kernel_heights: str = field(default='7,7,7,7')
    pooling_sizes: str = field(default='2,2,2,2')
    recombined_channels: str = field(default='3,3,3,3')
    conv_act: str = field(default='tanh')
    reduction_ratio: int = field(default=3)
    bilinear_type: str = field(default='field_interaction')
    reuse_graph_layer: bool = field(default=False)
    attn_scale: bool = field(default=False)
    use_lr: bool = field(default=False)
    attn_size: int = field(default=40)
    num_attn_layers: int = field(default=2)
    cin_layer_units: str = field(default='50,50')
    product_type: str = field(default='inner')
    outer_product_kernel_type: str = field(default='mat')
    dnn_size: int = field(default=1000, metadata={'help': "The size of each dnn layer"})
    num_dnn_layers: int = field(default=0, metadata={"help": "The number of dnn layers"})
    dnn_act: str = field(default='relu', metadata={'help': "The activation function for dnn layers"})
    dnn_drop: float = field(default=0.0, metadata={'help': "The dropout for dnn layers"})

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output