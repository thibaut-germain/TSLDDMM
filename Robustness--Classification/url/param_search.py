#-*- coding:utf-8 -*-

import os
import copy
import pickle
import random

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import *

from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
SEED = 0
seed_everything(SEED)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


from utils import get_data, get_coeffs
from model import ists_dataset, ists_classifier, train, evaluate 

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


model_name_list = [
    'rnn', 'lstm', 'gru',
    'mtan', 'miam',
    'ode-lstm',
    'ancde', 'exit',
    "neuralsde_1_18", "neuralsde_4_17"
]


if not os.path.exists('params'):
    os.mkdir('params')

# set model
def tune_model(data_name, missing_rate, model_name, model_config, EPOCHS=100, SEED=SEED):
    print(data_name)
    
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
    if model_name in ['lstm-mean', 'gru-mean', 'gru-simple', 'grud', 'tlstm', 'plstm', 'tglstm']:
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
    
    # set ancde
    if not os.path.exists(os.path.join(os.path.join(os.getcwd(),'ancde'))):
        os.mkdir(os.path.join(os.path.join(os.getcwd(),'ancde')))
    ancde_path = os.path.join(os.getcwd(), 'ancde/{}_{}.npy'.format(data_name, str(SEED))) # for ancde model

    # set model
    model = ists_classifier(model_name=model_name, input_dim=num_dim, seq_len=seq_len, num_class=num_class, dropout=0.1, use_intensity=use_intensity, 
                            method='euler', file=ancde_path, device='cuda', **model_kwargs)
    model = model.to(device)

    # set loss & optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr*0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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

        if (e > 10) & (patient > 5):
            break

        scheduler.step()
        valid_loss = np.nan_to_num(valid_loss,np.infty) # ignore
        ray.train.report(dict(loss=valid_loss))
    print("Finished Training")


def optimize(data_name, missing_rate, model_name, num_samples=20, max_num_epochs=10):
    model_config = {
        "data_name": data_name, 
        "missing_rate": missing_rate, 
        "model_name": model_name, 
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_dim": tune.choice([16, 32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3, 4]),
    }
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    def optimize_model(model_config):
        tune_model(data_name=data_name, missing_rate=missing_rate, model_name=model_name, 
                   model_config=model_config, EPOCHS=max_num_epochs, SEED=SEED)
    
    result = tune.run(
        tune.with_parameters(optimize_model),
        resources_per_trial={"cpu": 16, "gpu": 1.0},
        config=model_config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        stop={"training_iteration": 10}
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    
    # save params
    if not os.path.exists(os.path.join('params', data_name)):
        os.mkdir(os.path.join('params', data_name))
    
    out_name = '_'.join([data_name, model_name])
    with open(os.path.join('params', data_name, out_name), 'wb') as f:
        pickle.dump(best_trial.config, f)
    
    return best_trial


##### run all
valid_datasets = [
    "ArrowHead",
    "BME",
    "ECG200",
    "FacesUCR",
    "GunPoint",
    "PhalangesOutlinesCorrect",
    "Trace",
    "Cricket",
    "ERing",
    "Handwriting",
    "Libras",
    "NATOPS",
    "RacketSports",
    "UWaveGestureLibrary",
    "ArticularyWordRecognition"
]



# optimize parameters
for data_name in valid_datasets[:1]:
    for model_name in model_name_list[:1]:
        
        out_name = '_'.join([data_name, model_name])
        if os.path.exists(os.path.join('params', data_name, out_name)):
            continue

        best_trial = optimize(data_name=data_name, missing_rate=0.0, model_name=model_name, num_samples=20, max_num_epochs=10)


