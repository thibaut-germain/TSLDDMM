import pandas as pd
import numpy as np 
import time 
import os
import random
import copy


from utils import get_data
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.base import clone
from sklearn.metrics import f1_score,accuracy_score

from fda import TangentCurveLogisticRegression, FpcaSVM

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

univariate_datasets = [
    "ArrowHead", 
    "BME", 
    "ECG200",
    "FacesUCR",
    "GunPoint", 
    "PhalangesOutlinesCorrect",
    "Trace",
]

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

SEED = 0
seed_everything(SEED)
Cs = [100.,10.,1.,0.1]


lst = []
for dataset in univariate_datasets[:1]:

    #load data 
    X, X_raw_mask, X_delta,Y,train_idx,valid_idx,test_idx = get_data(dataset,missing_rate=0.0)
    X_train, y_train = X[train_idx].squeeze(),Y[train_idx]
    X_val, y_val = X[valid_idx].squeeze(), Y[valid_idx]
    X_test, y_test = X[test_idx].squeeze(), Y[test_idx]

    best_performer = None
    best_score = -np.inf

    for C in Cs:
            tclr = TangentCurveLogisticRegression("l2",C=C,karcher_niter=50,n_jobs=30)
            start_time = time.time()
            tclr.fit(X_train.astype(float),y_train)
            ex_time = time.time()-start_time
            y_pred = tclr.predict(X_val)
            score = f1_score(y_val,y_pred,average="macro")
            print(f"TCLR -- Valdidation: dataset: {dataset} -- C: {C} -- score: {np.around(score,2)}")
            if score > best_score:
                    best_performer = copy.deepcopy(tclr)
                    best_score = score.copy()

    y_pred = best_performer.predict(X_test)
    fscore = f1_score(y_test,y_pred,average="macro")
    ascore = accuracy_score(y_test,y_pred)

    print(f"TCLR -- Test: dataset: {dataset} -- score: {np.around(fscore,2)}")
            
    lst.append([dataset,ascore,fscore,ex_time])
df = pd.DataFrame(lst, columns=["dataset","accuracy","fscore","time"])
df.to_csv(f"../results/tclr_0.0_{SEED}.csv")


lst = []
for dataset in univariate_datasets[:1]:
    
    #load data 
    X, X_raw_mask, X_delta,Y,train_idx,valid_idx,test_idx = get_data(dataset,missing_rate=0.0)
    X_train, y_train = X[train_idx].squeeze(),Y[train_idx]
    X_val, y_val = X[valid_idx].squeeze(), Y[valid_idx]
    X_test, y_test = X[test_idx].squeeze(), Y[test_idx]

    

    best_performer = None
    best_score = -np.inf

    for C in Cs:
            tclr = FpcaSVM(C=C,parallel=True,njobs=30)
            start_time = time.time()
            tclr.fit(X_train,y_train)
            ex_time = time.time()-start_time
            y_pred = tclr.predict(X_val)
            score = f1_score(y_val,y_pred,average="macro")
            print(f"FPCA-SVM -- Valdidation: dataset: {dataset} -- C: {C} -- score: {np.around(score,2)}")
            if score > best_score:
                    best_performer = copy.deepcopy(tclr)
                    best_score = score.copy()

    y_pred = best_performer.predict(X_test)
    fscore = f1_score(y_test,y_pred,average="macro")
    ascore = accuracy_score(y_test,y_pred)

    print(f"SHAPE-FPCA-SVM -- Test: dataset: {dataset} -- score: {np.around(fscore,2)}")
            
    lst.append([dataset,ascore,fscore,ex_time])
df = pd.DataFrame(lst, columns=["dataset","accuracy","fscore","time"])
df.to_csv(f"../results/shape-fpca_0.0_{SEED}.csv")
