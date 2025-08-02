# %% [markdown]
# ## 一、download the amazon beauty dataset
# https://nijianmo.github.io/amazon/index.html

# %%
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
#	reviews (371,345 reviews)	metadata (32,992 products)
beauty_full_df = pd.read_json("All_Beauty.json", lines=True)
beauty_meta_df = pd.read_json("meta_All_Beauty.json", lines=True)
beauty_meta_df.fillna("unknown", inplace=True)
beauty_full_df.fillna("unknown", inplace=True)

# %%
item_selected = beauty_meta_df[["title","brand","asin","price"]] # repeated  asin
item_selected = item_selected.drop_duplicates(subset="asin", keep="first") 

# ["overall","reviewTime","reviewerID","asin","reviewerName","reviewText","summary"]
user_selected = beauty_full_df[["overall","asin","reviewerID", "reviewerName"]]
filtered_item = item_selected[item_selected["asin"].isin(user_selected["asin"])]
# Merge the two dataframes
merge_full = user_selected.merge(filtered_item, on="asin", how="left")
merge_full = merge_full.fillna("unknown")

# %%
for name in merge_full.columns:
    print(name,":", len(merge_full[name].unique()))

# %%
merge_beauty_full = merge_full

map_lable = lambda x: 1 if x >= 4 else 0
merge_beauty_full["label"] = merge_beauty_full["overall"].apply(map_lable)
merge_beauty_full.drop(columns=["overall"], inplace=True)

def generate_sentence(row):
    sentence = (
        f"The ID of the product is {row['asin']} and name of the product is {row['title']}. "
        f"And the brand name of the product is the {row['brand']}. It's price is {row['price'].strip()}. " 
        f"The ID of the reviewer is {row['reviewerID']}, whose name is {row["reviewerName"]}. "
    )
    return sentence
merge_beauty_full["sentence"] = merge_beauty_full.apply(generate_sentence, axis=1)


# %%
merge_beauty_full.to_csv('beauty_full_processed.csv', index=False, header=True)
merge_beauty_full = pd.read_csv('beauty_full_processed.csv', header=0)

# %%
for name in merge_beauty_full.columns:
    print(name,":", len(merge_beauty_full[name].unique()))

# %%
import pickle
import os 

class LoadData():
    def __init__(self, data_path=None, save_path="LLM_CTR_AZ_Beauty_data.p", drop_columns=None):
        
        self.df_data = pd.read_csv(data_path, header = 0)
        if "summary" in self.df_data.columns:
            self.df_data.drop(columns=["summary"], inplace=True)
        
        self.items = self.df_data[self.df_data.columns[:-2]]
        self.sentences = self.df_data[self.df_data.columns[-1]]
        self.targets = self.df_data[self.df_data.columns[-2]]
        self.features_M = {}
        self.save_path = save_path
        self.construct_df()


    def construct_df(self):
        self.field_dims = []
        for name in self.items.columns:
            field_set = self.items[name].unique()
            self.field_dims.append(len(self.items[name].unique()))
            maps = {val: k for k, val in enumerate(field_set)}
            self.items[name] = self.items[name].map(maps)
            self.features_M[name] = maps
        
        print(self.field_dims)
        print(sum(self.field_dims))
        if not os.path.exists(self.save_path):
            pickle.dump((self.items, self.targets, self.sentences, self.field_dims), open(self.save_path, "wb"))
            print(f"Save the data to {self.save_path} successfully!")
data_path = "beauty_full_processed.csv"
fashion_LoadData = LoadData(data_path=data_path, save_path="LLM_CTR_AZ_Beauty_data.p")
