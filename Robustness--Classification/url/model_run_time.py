#-*- coding:utf-8 -*-

import os
import sys
import copy
import pickle
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import *
import time


from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import  DataLoader

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 0
seed_everything(SEED)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


from utils import get_data, get_coeffs
from model import ists_dataset, ists_classifier, train, evaluate 


model_name_list = [
    'lstm',
]


if not os.path.exists('../time_results'):
    os.mkdir('../time_results')
    
    
def run_model(data_name, missing_rate, model_name, model_config, EPOCHS=100, SEED=SEED, CHECK=False):
    print(data_name, missing_rate, model_name)

    # check exist
    out_name = '_'.join([data_name, str(missing_rate), model_name, str(SEED)])
    out_path = '../time_results/{}/{}/{}'.format(data_name, str(missing_rate), out_name)

    if CHECK & os.path.exists(out_path):
        return None
    
    # setup out path
    if not os.path.exists('../time_results/{}'.format(data_name)):
        os.mkdir('../time_results/{}'.format(data_name))
    
    # load data
    X_missing, X_mask, X_delta,Y,train_idx,valid_idx,test_idx = get_data(data_name,missing_rate=missing_rate)
    X_missing,X_mask,X_delta = torch.tensor(X_missing),torch.tensor(X_mask),torch.tensor(X_delta)
    num_data = X_missing.shape[0]
    seq_len = X_missing.shape[1]
    num_dim = X_missing.shape[2]
    num_class = len(np.unique(Y))
    
    # set batch_size by the number of data
    batch_size = 2**4
    for i in range(4,5): #8
        if 2**i > num_data/2:
            break
        batch_size = 2**i

    # set learning params
    if model_config['lr'] is None:
        lr = 1e-3 * (batch_size / 2**4)
    else:
        lr = model_config['lr']
    
    # check model_name and settings
    if model_name in ['gru-dt', 'gru-d', 'gru-ode', 'ode-rnn', 'ncde', 'ancde', 'exit']:
        interpolate = 'natural'
    else:
        interpolate = 'hermite'

    if model_name in ['gru-dt', 'gru-d', 'ode-rnn']:
        use_intensity = True
    else:
        use_intensity = False

    if not os.path.exists('../time_results/{}/{}'.format(data_name, str(missing_rate))):
        os.mkdir('../time_results/{}/{}'.format(data_name, str(missing_rate)))
       
    ## data split    
    seed_everything(SEED) 
    X_train = X_missing[train_idx]
    coeffs = get_coeffs(X_missing,X_mask,X_delta,interpolate,use_intensity)

    out = []
    for Xi, train_Xi in zip(X_missing.unbind(dim=-1), X_train.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    X_missing_norm = torch.stack(out, dim=-1)

    train_dataset = ists_dataset(Y, X_missing_norm, X_mask, X_delta, coeffs, train_idx)
    valid_dataset = ists_dataset(Y, X_missing_norm, X_mask, X_delta, coeffs, valid_idx)
    test_dataset = ists_dataset(Y, X_missing_norm, X_mask, X_delta, coeffs, test_idx)

    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_batch = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # set params
    if model_name in ['cnn', 'cnn-3', 'cnn-5', 'cnn-7', 'rnn', 'lstm', 'gru', 'gru-simple', 'grud', 'bilstm', 'tlstm', 'plstm', 'tglstm', 'transformer',]:
        num_layers = model_config['num_layers']
        num_hidden_layers = None
    elif model_name in ['sand', 'mtan', 'miam']:
        num_layers = 1 
        num_hidden_layers = None
    else:
        num_layers = 1
        num_hidden_layers = model_config['num_layers']
    
    # get model_kwargs
    model_kwargs = {
        'hidden_dim': model_config['hidden_dim'], 
        'hidden_hidden_dim': model_config['hidden_dim'], 
        'num_layers': num_layers, 
        'num_hidden_layers': num_hidden_layers,
    }
    
    start_time = time.time()
    # set tmp_path
    if not os.path.exists(os.path.join(os.path.join(os.getcwd(),'tmp'))):
        os.mkdir(os.path.join(os.path.join(os.getcwd(),'tmp')))
    tmp_path = os.path.join(os.getcwd(), 'tmp/{}_{}_{}.npy'.format(data_name, missing_rate, str(SEED))) 

    # set model
    model = ists_classifier(model_name=model_name, input_dim=num_dim, seq_len=seq_len, num_class=num_class, dropout=0.1, use_intensity=use_intensity, 
                            method='euler', file=tmp_path, device='cuda', **model_kwargs)
    model = model.to(device)

    # set loss & optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr*0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = np.infty
    best_model_wts = copy.deepcopy(model.state_dict())
    patient = 0

    for e in tqdm(range(EPOCHS)):
        train_loss = train(model, optimizer, criterion, train_batch, device)
        valid_loss = evaluate(model, criterion, valid_batch, device)
        test_loss = evaluate(model, criterion, test_batch, device)

        if e % 10 == 0:
            print(e, train_loss, valid_loss, test_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patient = 0
        else:
            patient += 1

        #if (e > 20) & (patient > 10):
        #    break

        scheduler.step()

    duration = time.time()-start_time
    # using trained model
    try:
        model.load_state_dict(best_model_wts)
    except:
        pass

    # predict
    model.eval()

    y_true, y_pred, logit_list = [], [], []
    with torch.no_grad():
        for batch in test_batch:
            y = batch['label'].long().to(device)
            seq = torch.stack([
                torch.nan_to_num(batch['x_missing'], 0),
                batch['x_mask'].unsqueeze(-1).repeat((1,1,batch['x_missing'].shape[-1])),
                batch['x_delta'].unsqueeze(-1).repeat((1,1,batch['x_missing'].shape[-1])),
            ], dim=1).to(device)

            if model_name in ['latentsde', 'leap']:  
                logit, loss = model(seq, batch['coeffs'].to(device))
                # logit = torch.nan_to_num(logit) # replace nan
                ce_loss = criterion(logit, y) 
                loss = ce_loss + loss
            else:
                logit = model(seq, batch['coeffs'].to(device))
                # logit = torch.nan_to_num(logit) # replace nan
                ce_loss = criterion(logit, y)
                loss = ce_loss

            y_true.append(y.cpu().numpy())
            y_pred.append(logit.max(1)[1].cpu().numpy())
            logit_list.append(logit.cpu().numpy())

    y_true = np.array([x for y in y_true for x in y]).flatten()
    y_pred = np.array([x for y in y_pred for x in y]).flatten()
    logit_list = np.array([x for y in logit_list for x in y])

    acc_score = accuracy_score(y_true,y_pred)
    f_score = f1_score(y_true,y_pred, average='macro')

    print(data_name, missing_rate, model_name, acc_score, f_score)

    out_name = '_'.join([data_name, str(missing_rate), model_name, str(SEED)])
    with open('../time_results/{}/{}/{}'.format(data_name, str(missing_rate), out_name), 'wb') as f:
        pickle.dump([y_true, y_pred, logit_list,dict(duration = duration,accuracy = acc_score,f1score = f_score)], f)
    
    
##### run all
valid_datasets = [
    "ArrowHead",
    "BME",
    "ERing",
    "UWaveGestureLibrary",   
]


# run experiments
for missing_rate in [0.0]:
    for data_name in valid_datasets:
        for model_name in model_name_list:
            #try:
            param_name = '_'.join([data_name, model_name])
            with open((os.path.join('params', data_name, param_name)), 'rb') as f:
                model_config = pickle.load(f)
                
            run_model(data_name, missing_rate, model_name, model_config, EPOCHS=100, SEED=SEED, CHECK=True)
    
            #except:
            #    continue

