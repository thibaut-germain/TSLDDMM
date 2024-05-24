import pandas as pd
import numpy as np 
import jax.numpy as jnp
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

settings_df = pd.read_csv("settings.csv",index_col=0)


for missing_rate in [0.3,0.5,0.7]: 
    lst = []
    for model in ['ts-lddmm','lddmm']:
        for dataset in valid_datasets: 

            #load data
            X_missing, X_raw_mask, X_delta,Y,train_idx,valid_idx,test_idx = get_data(dataset,missing_rate=missing_rate)
            X,X_mask = from_mask_timeseries_to_dataset(X_missing,X_raw_mask)
            X_train,X_train_mask,y_train = X[train_idx],X_mask[train_idx],Y[train_idx]
            X_val,X_val_mask,y_val = X[valid_idx],X_mask[valid_idx],Y[valid_idx]
            X_test,X_test_mask,y_test = X[test_idx],X_mask[test_idx],Y[test_idx]

            #set estimator
            C = settings_df[(settings_df.dataset == dataset)*(settings_df.model == model)]["C"].values[0]
            tsig = settings_df[(settings_df.dataset == dataset)*(settings_df.model == model)]["tsig"].values[0]
            nd = settings_df[(settings_df.dataset == dataset)*(settings_df.model == model)]["nd"].values[0]
            if model == "ts-lddmm":
                Kv = VFTSGaussKernel(1,0.1,tsig,1,nd)
            if model == "lddmm": 
                Kv = TSGaussKernel(tsig,nd)
            dataloss = VarifoldLoss(TSGaussGaussKernel(2,nd,2,nd))
            other_params = estimator_parameters.copy()
            other_params["C"] = C
            estimator = MomentaSVC(Kv,dataloss,**other_params)

            #fit estimator
            start_time = time.time()
            estimator.fit(X_train,X_train_mask,y_train)
            ex_time = time.time()-start_time

            #predict
            predictions = estimator.predict(X_test,X_test_mask)
            accuracy = accuracy_score(y_test,predictions)
            f1score = f1_score(y_test,predictions,average="macro")
            print(f"rate: {missing_rate}, model: {model} -- dataset: {dataset} -- accuracy: {np.around(accuracy,2)} -- f1score: {np.around(f1score,2)}-- ex_time {ex_time}")
            lst.append([dataset,accuracy,f1score,ex_time])
            df = pd.DataFrame(lst, columns=["dataset","accuracy","fscore","time"])
            df.to_csv(f"../results/lddmm_methods_0.{int(missing_rate*10)}_{SEED}.csv")


