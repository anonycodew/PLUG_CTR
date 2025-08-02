import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm 
import pickle 

import pickle
import os 
import h5py 


class LoadData():
    def __init__(self, data_path=None, dataset_name="Frappe"):
        # the csv_file data_path
        self.dataset_name = dataset_name
        self.df_data = pd.read_csv(data_path, header = 0)

        self.items = self.df_data[self.df_data.columns[:-2]]
        
        self.sentences = self.df_data[self.df_data.columns[-1]]
        
        self.targets = self.df_data[self.df_data.columns[-2]]
                
        self.features_M = {}
        self.save_path = f"./LLM_CTR_{self.dataset_name}_data.p"
        print(self.items.columns)
        self.construct_df()


    def construct_df(self):
        self.field_dims = []
        for name in self.items.columns:
            print(name)
            field_set = set(self.items[name])
            self.field_dims.append(len(field_set))
            
            maps = {val: k for k, val in enumerate(field_set)}
            self.items[name] = self.items[name].map(maps)
            self.features_M[name] = maps
            
        
        print(self.field_dims)
        if not os.path.exists(self.save_path):
            pickle.dump((self.items, self.targets, self.sentences, self.field_dims), open("LLM_CTR_ML1M_data.p", "wb"))
            print("Save the data to {self.save_path} successfully!")
         
      
class collactor():
    def __init__(self, tokenizer):
        self.token = tokenizer
    
    def __call__(self, batch_data):
        # batch_data = [(user_item, sentence, label), (user_item, sentence, label), (user_item, sentence, label)]
        user_item, sentences, labels = zip(*batch_data)
        user_item, sentences, labels = torch.tensor(np.array(list(user_item))), list(sentences), torch.tensor(np.array(list(labels)))
        if self.token is None:
            return user_item, labels, None, None
        else:
            token_sentences = self.token(sentences, add_special_tokens=True, max_length=128, 
                                         truncation=True, padding=True, return_tensors="pt")
            return user_item, labels, token_sentences["input_ids"], token_sentences["attention_mask"]


class Base_Dataset(Dataset):
    def __init__(self, items: pd.DataFrame, 
                 targets: pd.DataFrame):
        self.items = items
        self.targets = targets
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        item_ = self.items.iloc[index].values
        target_ = self.targets.iloc[index]
        return item_, target_

class HDF5_Dataset(Dataset):
    def __init__(self, 
                 items: pd.DataFrame, 
                 targets: pd.DataFrame, 
                 file_path: str = "../semantic_hdf5/hdf5_embeddings/semantic_representations_llama_1B_last_mean_pool.npy"):
        self.items = items 
        self.targets = targets
        # The hdf5 file which saves the semantic representation representation, 
        # before your use if you should generate the corresponding hdf5 file by yourself.
        self.data = np.load(file_path) 

        
    def __len__(self):
        return len(self.items) 
    
    def __getitem__(self, index):
        item_ = self.items.iloc[index].values
        target_ = self.targets.iloc[index]
        single_representation_ = torch.from_numpy(self.data[index])
        return item_, single_representation_, target_

def frappe_dataloader_train(batch_size=128, file_path=None, num_workers=4, tokenzier=None, type = "hdf5"):
    # In this file, we have saved the items, targets, sentences, and field_dims. Sentences is the textual modality.
    path = "./LLM_CTR_frappe_data.p"  
    if not os.path.exists(path):
        dl = LoadData()
        
    items, targets, sentences, field_dims = pickle.load(open(path, 'rb'))
  
    all_length = len(items)
    
    valid_size = int(0.1 * all_length)
    train_size = all_length - valid_size
    print("all_length", train_size + valid_size)
    if type == "hdf5": # For PLUG
        print(file_path)
        train_dataset = HDF5_Dataset(items, targets, file_path=file_path)
        
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size],
                                                                    generator=torch.Generator().manual_seed(2025))
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size - valid_size, valid_size],
                                                                     generator=torch.Generator().manual_seed(2025))
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,
                                                   shuffle=True, drop_last=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=2)
        
    elif type == "base": # base 
        train_dataset = Base_Dataset(items, targets)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size - valid_size, valid_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    return field_dims, train_loader, valid_loader, test_loader

if __name__ == '__main__':

    fields, train_loader, valid_loader, test_loader = frappe_dataloader_train(batch_size=128)
    print(fields)
    print(len(train_loader))
    one_iter = iter(train_loader)
    item, sentence, target = next(one_iter)
    print(item.shape)
    print(target)
    print(sentence[0])