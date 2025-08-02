# The sourcecode of the PLUG framework

## Requirements 
- python==3.12.7
- torch==2.6.0+cu124
- transformers==4.51.0
- numpy=1.26.4
- nni=3.0
- pandas=2.2.2
- tqdm=4.67.1

## Download four datasets
- [Frappe](https://www.baltrunas.info/context-aware/frappe)
- [ML1M](https://grouplens.org/datasets/movielens/)
- [Book](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
- [Beauty](https://cseweb.ucsd.edu/~jmcauley/datasets.html)

According the link, the datasets should be downloaded and placed in the corresponding folder. Due to the capacity of the attachments and anonymity requirements, we will give the processed data for direct use after receipt. Take the Beauty for example, the processed file is: 

```./alldatasets/amazon_beauty/process_beauty.py```

After you download the datasets, you can run the following code to preprocess the data, and you will get "LLM_CTR_AZ_beauty_data.p", you can load it with pickle package. 

```
items, targets, sentences, field_dims = pickle.load(open(path, 'rb')) 
```
And then you can generate the semantic representations with the pre-trained models, following the next steps.

## Download the pre-trained models from Huggingface or Modelscope platform.

- TinyBERT(14.4M)
- ReBERTa_base(110M)
- BERT_large(336M)
- BGE-m3(567M)
- QWen-emb (596M)
- LLaMa3.2(1B)
- Qwen2.5(1.5B)
- DS_R1(1.5B)
- QWen2.5(3B)
- LlaMa3.2(3B)

After downloading the pre-trained models, place them in the pre-trained_models folder. And then generating the semantic representations using the file: 

```./alldatasets/generate_hidden_vectors.py ```


## Run the code with python 
```CUDA_VISIBLE_DEVICES=0 python run_plug.py```

Additionally, you can run the code with nni for parameter tuning
``` nnictl create --config main_plug_nni.yaml  --port 6006```