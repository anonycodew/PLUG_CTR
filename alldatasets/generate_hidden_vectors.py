import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os 
import tqdm  
import numpy as np 
from torch.utils.data import DataLoader
import pickle 
print(os.getcwd())

os.environ["TORCH_USE_CUDA_DSA"] = "TRUE"

from transformers import AutoTokenizer, AutoModel, AutoConfig, RobertaModel, RobertaForSequenceClassification


llm_root = "./llm_download/"
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


def get_llm_model(llm_path):
    
    tokenizer =AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True) # left padding 
    llm_model = AutoModel.from_pretrained(llm_path, trust_remote_code=True, device_map="auto")
    llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]" 
    tokenizer.padding_side = "left"
    
    count = 0
    for name, param in llm_model.named_parameters():
        count += param.numel()
    print(f"model_name:{llm_path},model_parameters:{count}")
    
    return tokenizer, llm_model, llm_config

def load_dataset(batch_size=256, dataset_name="beauty"):
    path_root = "."
    path_dict = {
        "frappe":   path_root + "/frappe/LLM_CTR_frappe_data.p",
        "book":     path_root + "/book/LLM_CTR_Book_data.p",
        "ml1m":     path_root + "/ml1m/LLM_CTR_ML1M_data.p",
        "beauty":   path_root + "/amazon_beauty/LLM_CTR_AZ_beauty_data.p",
    } 
    data_path = path_dict[dataset_name]
    print(data_path)
    items, targets, sentences, field_dims = pickle.load(open(data_path, 'rb')) 
    sentences  = list(sentences.values)
    dataloader = DataLoader(sentences, batch_size) 
    return dataloader

import torch

def cal_valid_token_average_with_mask(
    token_embeddings: torch.Tensor, 
    attention_mask: torch.Tensor,
    eps: float = 1e-8  ) -> torch.Tensor:
    """    
    Parameters :
        token_embeddings: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        eps: 1e-8
    
    returns:
        Theaverage hidden representation of the valid tokens, [batch_size, hidden_size]
    """
    if not isinstance(token_embeddings, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TypeError("Inputs must be PyTorch tensors.")

    mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
    
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)  # [batch_size, hidden_size]
    valid_tokens = torch.sum(mask, dim=1) + eps                
    
    mean_embeddings = sum_embeddings / valid_tokens
    
    return mean_embeddings

def generate_and_save_hidden_vectors(tokenizer, llm_config, llm_model, dataloader, llm_name, data_set_name, save_root="./semantic_hdf5"):
    torch.cuda.empty_cache()


    all_pooled_representations_last = []


    llm_model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            sample_outs = tokenizer(batch, add_special_tokens=True, 
                                max_length=128, truncation=True, padding=True, 
                                return_tensors="pt",  padding_side='left').to(llm_model.device)
            outputs  = llm_model(**sample_outs, output_hidden_states=True) 
            
            pooled_representation_last = cal_valid_token_average_with_mask(outputs["hidden_states"][-1], sample_outs["attention_mask"])
            pooled_representation_last = pooled_representation_last.detach().cpu().numpy()
            all_pooled_representations_last.append(pooled_representation_last)

    save_root = save_root + f"/{data_set_name}_hdf5_embeddings/"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    hidden_layer_indexs = ["last"]
    
    for index, all_pooled in enumerate([all_pooled_representations_last]):
        if all_pooled == []:
            continue
        hidden_layer_index = hidden_layer_indexs[index]
        merged = np.concatenate(all_pooled, axis=0) 
        print(merged.shape)
        file_name = f"semantic_representations_{llm_name}_{hidden_layer_index}_mean_pool.npy"
        np.save(save_root + file_name, merged)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    dataset = ["beauty"] # , "ml1m","fashion","beauty","book"
    model_select_names  = ["tiny_bert"] # qwen_emb llama_1B QWen_1_5B ,"QWen_1_5B"  m3e 
    for dataset_name in dataset:
        dataloader = load_dataset(batch_size=128, dataset_name=dataset_name)
        for llm_name in model_select_names:
            print(llm_name)
            print(dataset_name)
            llm_path = llm_model_path[llm_name]
            tokenizer, llm_model, llm_config = get_llm_model(llm_path)
            generate_and_save_hidden_vectors(tokenizer, llm_config, llm_model, dataloader, llm_name, data_set_name=dataset_name)
            del llm_model
            del tokenizer
            del llm_config
            print(f"{llm_name}_{dataset_name} generate hidden vectors done!")