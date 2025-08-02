
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time     
import sys
import tqdm 
import os 
from sklearn.metrics import log_loss, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import nni 
import random
import numpy as np 

import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["NUMEXPR_MAX_THREADS"] = r"64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from utils.earlystoping import EarlyStopping, EarlyStoppingLoss
from utils_get_data_model import get_dataset, get_plug_model, CTRModelArguments

config = CTRModelArguments
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def main(dataset_name, llm_ctr_model, ctr_model_name, weight_decay, epoch, cn_layers, llm_model, npy_layer,
         batch_size, embed_dim, learning_rate,path,save_dir, hint, device, choice, alpha, tao, norm, beta,reduce_dimension):
    field_dims, trainLoader, validLoader, testLoader = get_dataset(dataset_name, llm_model=llm_model, npy_layer=npy_layer,
                                                                   batch_size=batch_size, type="hdf5")
    print(field_dims)
    time_fix_y_m = time.strftime("%y%m%d", time.localtime())


    for K in [embed_dim]:
        paths = os.path.join(save_dir, dataset_name, llm_model, llm_ctr_model, ctr_model_name, str(K))
        if not os.path.exists(paths):
            os.makedirs(paths) 

        with open(paths + f"/{npy_layer}_{norm}_{alpha}_{beta}_{K}_{tao}_{reduce_dimension}_{batch_size}_{learning_rate}_{weight_decay}_{time_fix_y_m}.p",
                  "a+") as fout:
            fout.write("dataset_name:{}\tllm_model:{}\tllm_ctr_model:{}\tmodel_name:{}\tBatch_size:{}\tembed_dim:{}\talpha:{}\tbeta:{}\ttao:{}\treduce_dimension:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\n"
                       .format(dataset_name, llm_model, llm_ctr_model, ctr_model_name, batch_size,  K, alpha, beta,tao, reduce_dimension,learning_rate, time.strftime("%m%d%H%M%S", time.localtime()), weight_decay))
            print("Start train -- K : {}".format(K))

            criterion = torch.nn.BCELoss()
            config.cross_layers = cn_layers
            config.embed_dim = embed_dim
            config.batch_size = batch_size
            config.field_dims = field_dims
            config.tao = tao
            config.norm = norm
            config.reduce_dimension = reduce_dimension
            
            model = get_plug_model(llm_ctr_model, ctr_model_name, llm_model, field_dims, config).cuda()

            optimizer = torch.optim.Adam(
                params=model.parameters(), lr=learning_rate, 
                weight_decay=weight_decay)

            # Initial EarlyStopping
            early_stopping = EarlyStopping(patience=6, verbose=True, prefix=path)
            scheduler_min = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)
            val_auc_best = 0
            test_auc_best = 0
            auc_index_record = ""

            val_loss_best = 1000
            test_loss_best = 1000
            loss_index_record = ""

            for epoch_i in range(epoch):
                print(__file__, ctr_model_name, K, epoch_i, "/", epoch)
                print("Batch_size:{}\tembed_dim:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\t"
                      .format(batch_size, K, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay))
                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion, alpha=alpha, beta=beta, model_name=ctr_model_name)
                val_auc, val_loss = test_roc_loss(model, validLoader, llm_name=llm_ctr_model)
                test_auc, test_loss = test_roc_loss(model, testLoader, llm_name=llm_ctr_model)
                
                nni.report_intermediate_result(test_loss)

                scheduler_min.step(test_loss)
                end = time.time()
                
                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    test_auc_best = test_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    # test_loss_best = test_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                early_stopping(val_auc)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))
            print("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))
            fout.write("auc_best:\t{}\nloss_best:\t{}\n\n".format(auc_index_record, loss_index_record))
            
            fout.write("###########"*2 + "\n\n\n")
            
def train(model, optimizer, data_loader, criterion, 
          alpha: float = 1.0, beta=0.1, model_name :str = "fm", print_num = 500 ):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item_ids, representations, labels) in enumerate(tqdm.tqdm(data_loader)):

        user_item_ids, labels = user_item_ids.long().cuda(), labels.float().cuda()
        representations = representations.float().cuda()
        
        model.zero_grad()
        outputs  = model(user_item_ids=user_item_ids, semantic_features=representations)  
        y_pred = outputs["y_pred"]
        loss_y = criterion(y_pred, labels)
        cl_loss = outputs["cl_loss"] 
        loss = loss_y + alpha*cl_loss
        
        if "y_pred_llm" in outputs.keys():
            loss_llm = criterion(outputs["y_pred_llm"], labels)
            loss += beta * loss_llm
        
        loss.backward()
        optimizer.step()

        pred.extend(y_pred.tolist())
        target.extend(labels.tolist())
        total_loss += loss.item()

    ave_loss = total_loss / (i + 1)
    return ave_loss


def test_roc_loss(model, data_loader, llm_name=None):
    model.eval()
    targets, predicts = list(), list()
    total_loss = 0
    with torch.no_grad():
        for i, (user_item_ids, representations, labels) in enumerate(tqdm.tqdm(data_loader)):
            user_item_ids, labels = user_item_ids.long().cuda(), labels.float().cuda()
            outputs = model.ctr_model(user_item_ids, return_features=False)
            y_pred = outputs["y_pred"]
            targets.extend(labels.tolist())
            predicts.extend(y_pred.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


if __name__ == '__main__':
    
    RECEIVED_PARAMS = {
        "dataset_name": "frappe",
        "ctr_model_name":"DCNv2_PLUG",
        "learning_rate":1e-2,
        "weight_decay":1e-4,
        "batch_size":4096,
        "embed_dim":16,
        "repeats":10,
        "interaction_layers":3,
        "reduce_dimension":64,
        "llm_ctr_model":"PLUG",
        "llm_model":"tiny_bert",
        "decoder_layer":"last",
        "alpha":"1.0",
        "beta":"1.0",
        "tao":"0.75",
        "norm":"norm",
        "epoch":50,
        "save_path":"plug_chkpt",
    }
    
    # RECEIVED_PARAMS = nni.get_next_parameter() 
    print(RECEIVED_PARAMS)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default=RECEIVED_PARAMS["dataset_name"], help="dataset_name")
    parser.add_argument('--ctr_model_name', default=RECEIVED_PARAMS["ctr_model_name"], help="model_name")

    parser.add_argument('--learning_rate', default=RECEIVED_PARAMS["learning_rate"], type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=RECEIVED_PARAMS["weight_decay"], type=float, help="weight_decay")
    parser.add_argument('--batch_size', default=RECEIVED_PARAMS["batch_size"], type=int)
    parser.add_argument('--embed_dim', default=RECEIVED_PARAMS["embed_dim"], type=int, help="the size of feature dimension")
    parser.add_argument('--repeats', type=int, default=RECEIVED_PARAMS["repeats"], help="")
    parser.add_argument('--cn_layers', type=int, default=RECEIVED_PARAMS["interaction_layers"], help="")
    parser.add_argument('--reduce_dimension', type=int, default=RECEIVED_PARAMS["reduce_dimension"], help="")


    parser.add_argument('--llm_ctr_model', type=str, default=RECEIVED_PARAMS["llm_ctr_model"], help="")
    parser.add_argument('--llm_model', type=str, default=RECEIVED_PARAMS["llm_model"], help="")
    parser.add_argument('--npy_layer', type=str, default=RECEIVED_PARAMS["decoder_layer"], help="")
    parser.add_argument('--alpha', type=float, default=RECEIVED_PARAMS["alpha"], help="")
    parser.add_argument('--beta', type=float, default=RECEIVED_PARAMS["beta"], help="")
    parser.add_argument('--tao', type=float, default=RECEIVED_PARAMS["tao"], help="") 
    parser.add_argument('--norm', type=str, default=RECEIVED_PARAMS["norm"], help="")

    parser.add_argument('--epoch', default=RECEIVED_PARAMS["epoch"], type=int, help="choice")
    parser.add_argument('--path', default="../data/", type=str, help="")
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--choice', default=0, type=int, help="choice")
    parser.add_argument('--save_dir', default=f'../chkpt/plug/{RECEIVED_PARAMS["save_path"]}/', help="") 
    parser.add_argument('--hint', default="dcn", help="")
    
    args = parser.parse_args()
    for i in range(args.repeats):
        main(
            dataset_name=args.dataset_name,
            llm_ctr_model = args.llm_ctr_model,
            ctr_model_name=args.ctr_model_name,
            weight_decay = args.weight_decay,
            batch_size = args.batch_size,
            embed_dim =args.embed_dim, 
            learning_rate = args.learning_rate, 
            npy_layer = args.npy_layer,
            llm_model=args.llm_model,
            epoch=args.epoch,
            path = args.path,
            device = args.device, 
            choice= args.choice,
            save_dir=args.save_dir,
            cn_layers = args.cn_layers,
            hint=args.hint,
            alpha = args.alpha,
            tao = args.tao,
            norm = args.norm,
            beta = args.beta,
            reduce_dimension = args.reduce_dimension
        )