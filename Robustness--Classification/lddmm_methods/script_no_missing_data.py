import pandas as pd
import numpy as np 
import time 
import os
import random
import sys
sys.path.insert(0,"../../")

from sklearn.metrics import f1_score,accuracy_score
import optax
from optax.schedules import warmup_cosine_decay_schedule

from src.kernel import VFTSGaussKernel,TSGaussGaussKernel,TSGaussKernel
from src.loss import VarifoldLoss
from src.classifier import MomentaSVC
from utils import get_data, from_mask_timeseries_to_dataset


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

SEED = 0
seed_everything(SEED)


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

estimator_parameters = dict(
    C=1,
    max_per_batch = 1,
    gamma_loss = 0.0,
    niter = 400,
    optimizer = optax.adabelief(warmup_cosine_decay_schedule(0,0.1,40,400,0)),
    nt = 10,
    deltat = 1.,
    verbose = True
)


def set_VFTSGausGaussKernel(X,X_mask,prop=0.33,sampfreq =1.): 
    nd = X.shape[2]-1
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,2*sampfreq)
    return VFTSGaussKernel(1,0.1,t_sig,1,nd)

def set_TSGaussKernel(X,X_mask,prop=0.33,sampfreq =1.): 
    nd = X.shape[2]-1
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,2*sampfreq)
    return TSGaussKernel(t_sig,nd)

def get_settings(X,X_mask,prop=0.33,sampfreq =1.):
    nd = X.shape[2]-1
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,2*sampfreq)
    return t_sig,nd

def set_TSGausGausKernel(X,X_mask,prop = 0.02,sampfreq =1.): 
    nd = X.shape[2]-1
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,1*sampfreq)
    return TSGaussGaussKernel(2,nd,2,nd)

lst =[]
setting_lst = []
for model in ["ts-lddmm","lddmm"]:
    for dataset in valid_datasets:

        print(f"Strarting: {model} {dataset}")
    
        #load data
        X_missing, X_raw_mask, X_delta,Y,train_idx,valid_idx,test_idx = get_data(dataset,missing_rate=0.0)
        X,X_mask = from_mask_timeseries_to_dataset(X_missing,X_raw_mask)
        X_train,X_train_mask,y_train = X[train_idx],X_mask[train_idx],Y[train_idx]
        X_val,X_val_mask,y_val = X[valid_idx],X_mask[valid_idx],Y[valid_idx]
        X_test,X_test_mask,y_test = X[test_idx],X_mask[test_idx],Y[test_idx]
    

        try:
            # Set estimators
            if model == "ts-lddmm":
                Kv = set_VFTSGausGaussKernel(X_train,X_train_mask)
            if model == "lddmm":
                Kv = set_TSGaussKernel(X_train,X_train_mask)
            tsig,nd = get_settings(X_train,X_test_mask)
            Kl = set_TSGausGausKernel(X_train,X_train_mask)
            dataloss = VarifoldLoss(Kl)
            estimator = MomentaSVC(Kv,dataloss,**estimator_parameters)

            #fit estimator
            Cs = [100.,10.,1.,0.1]
            start_time = time.time()
            estimator.fit(X_train,X_train_mask,y_train)
            estimator.set_C(X_val,X_val_mask,y_val,Cs)
            ex_time = time.time()-start_time

            #predict
            predictions = estimator.predict(X_test,X_test_mask)
            accuracy = accuracy_score(y_test,predictions)
            f1score = f1_score(y_test,predictions,average="macro")
            print(f"model: {model} -- dataset: {dataset} -- accuracy: {np.around(accuracy,2)} -- f1score: {np.around(f1score,2)}-- ex_time {ex_time}")

            #save score 
            lst.append([dataset,model,accuracy,f1score,ex_time])
            df = pd.DataFrame(lst, columns=["dataset","model","accuracy","fscore","time"])
            df.to_csv(f"../results/lddmm_methods_0.0_{SEED}.csv")

            #save settings
            setting_lst.append([dataset,model,estimator.C,tsig,nd])
            df = pd.DataFrame(setting_lst, columns=["dataset","model","C","tsig","nd"])
            df.to_csv(f"settings.csv")
        except: 
            continue
