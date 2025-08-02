import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.000001, prefix = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.prefix_path = prefix

    def __call__(self, val_auc):

        score = val_auc

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print("Now auc:{}\tBest_auc:{}".format(val_auc, self.best_score))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.prefix_path+'/es_checkpoint.pt')	
        self.val_loss_min = val_loss

class EarlyStoppingLoss:
    def __init__(self, patience=7, verbose=False, delta=0.000001, prefix = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.prefix_path = prefix

    def __call__(self, val_loss):
        score = val_loss

        if self.best_score is None:
            self.best_score = score

        elif score > self.best_score - self.delta or score==self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print("Now loss:{}\tBest_loss:{}".format(val_loss,self.best_score))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.prefix_path+'/es_checkpoint.pt')
        self.val_loss_min = val_loss