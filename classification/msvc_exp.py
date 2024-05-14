import sys
sys.path.insert(0,"../")

import pandas as pd
import numpy as np 
import time
import optax
from optax.schedules import warmup_cosine_decay_schedule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,f1_score

from aeon.datasets import load_classification

from src.utils import from_timeseries_to_dataset
from src.kernel import VFTSGaussKernel,TSGaussGaussKernel,GaussKernel,MyTSGaussKernel
from src.loss import VarifoldLoss,MyLoss,SumVarifoldLoss
from src.classifier import MomentaSVC
from src.lddmm import batch_time_initializer
from src.barycenter import batch_barycenter_initializer
from src.utils import batch_dataset,from_timeseries_to_dataset


def mouse_loader(seed=0):
    y =pd.read_csv("../mouse/dataset/y.csv",index_col=0)
    X = np.load("../mouse/dataset/X.npy")
    X_mask = np.load("../mouse/dataset/X_mask.npy")
    X = X[:,::2,:]
    X_mask = X_mask[:,::2,:]

    lst = []
    for filename in y[y.genotype.isin(["colq","wt"])].filename.unique(): 
        idx = y[(y.filename == filename)*(y.before == "Y")].sample(20,random_state=seed).index
        lst.append(idx)
    idxs = np.concatenate(lst)

    X,X_mask = X[idxs],X_mask[idxs]
    y = y.iloc[idxs]
    y = y.genotype.values.astype(str)
    train_idx,test_idx = train_test_split(np.arange(X.shape[0]),test_size=0.33,shuffle=True,random_state=seed)
    return X[train_idx],X_mask[train_idx],X[test_idx],X_mask[test_idx],y[train_idx],y[test_idx]

def gait_loader(seed=0): 
    y = np.load("../gait_exp/dataset/y.npy")
    X = np.load("../gait_exp/dataset/X.npy")
    X_mask = np.load("../gait_exp/dataset/X_mask.npy")
    train_idx,test_idx = train_test_split(np.arange(X.shape[0]),test_size=0.33,shuffle=True,random_state=seed)
    return X[train_idx],X_mask[train_idx],X[test_idx],X_mask[test_idx],y[train_idx],y[test_idx]


def aeon_loader(dataset_name): 
    X_train,y_train = load_classification(dataset_name,split="train")
    X_train = np.moveaxis(X_train,1,2)
    X_train, X_train_mask= from_timeseries_to_dataset(X_train)
    X_test,y_test = load_classification(dataset_name,split="test")
    X_test = np.moveaxis(X_test,1,2)
    X_test,X_test_mask  = from_timeseries_to_dataset(X_test)
    return X_train,X_train_mask,X_test,X_test_mask,y_train,y_test 

    
def ECG200_loader(seed=0): 
    X_train,y_train = load_classification("ECG200",split="train")
    X_train = np.moveaxis(X_train,1,2)
    X_train, X_train_mask= from_timeseries_to_dataset(X_train)
    X_test,y_test = load_classification("ECG200",split="test")
    X_test = np.moveaxis(X_test,1,2)
    X_test,X_test_mask  = from_timeseries_to_dataset(X_test)
    return X_train,X_train_mask,X_test,X_test_mask,y_train,y_test 

def ArrowHead_loader(seed=0): 
    X_train,y_train = load_classification("ArrowHead",split="train")
    X_train = np.moveaxis(X_train,1,2)
    X_train, X_train_mask = from_timeseries_to_dataset(X_train)
    X_test,y_test = load_classification("ArrowHead",split="test")
    X_test = np.moveaxis(X_test,1,2)
    X_test, X_test_mask = from_timeseries_to_dataset(X_test)
    return X_train,X_train_mask,X_test,X_test_mask,y_train,y_test 


def GunPoint_loader(seed=0): 
    X_train,y_train = load_classification("GunPoint",split="train")
    X_train = np.moveaxis(X_train,1,2)
    X_train,X_train_mask = from_timeseries_to_dataset(X_train)
    X_test,y_test = load_classification("GunPoint",split="test")
    X_test = np.moveaxis(X_test,1,2)
    X_test, X_test_mask = from_timeseries_to_dataset(X_test)
    return X_train,X_train_mask,X_test,X_test_mask,y_train,y_test 


def NATOPS_loader(seed=0): 
    X_train,y_train = load_classification("NATOPS",split="train")
    X_train = np.moveaxis(X_train,1,2)
    X_train, X_train_mask = from_timeseries_to_dataset(X_train)
    X_test,y_test = load_classification("NATOPS",split="test")
    X_test = np.moveaxis(X_test,1,2)
    X_test, X_test_mask = from_timeseries_to_dataset(X_test)
    return X_train,X_train_mask,X_test,X_test_mask,y_train,y_test 


dataloaders = dict(
    #mouse = mouse_loader,
    #gait = gait_loader,
    ECG200 = ECG200_loader,
    ArrowHead = ArrowHead_loader,
    GunPoint = GunPoint_loader,
    NATOPS_loader = NATOPS_loader,
)


multivariate_dataset_lst = [
    "Cricket", # Maybe keep few labels
    "ERing", #idem
    "Handwriting", #idem
    "Libras", #idem
    "NATOPS",
    "RacketSports",
    "UWaveGestureLibrary",
    "ArticularyWordRecognition", 
]

univariate_dataset_lst = [
    "ArrowHead", 
    "BME", 
    "ECG200",
    "FacesUCR",
    "GunPoint", 
    "PhalangesOutlinesCorrect",
    "Trace",
]

dataset_names = univariate_dataset_lst + multivariate_dataset_lst

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

initializer_parameters = dict( 
    gamma_loss = 0.0,
    niter = 200,
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


def set_GaussKernel(X,X_mask,prop=0.33,sampfreq =1.): 
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,sampfreq)
    return GaussKernel(t_sig,1)

def set_TSGausGausKernel(X,X_mask,prop = 0.02,sampfreq =1.): 
    nd = X.shape[2]-1
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,1*sampfreq)
    return TSGaussGaussKernel(t_sig,nd,t_sig,nd)

def set_lst_TSGausGausKernel(X,X_mask,prop = 0.02,sampfreq =1.): 
    nd = X.shape[2]-1
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,1*sampfreq)
    Kl1 = TSGaussGaussKernel(2,nd,2,nd)
    Kl2 = TSGaussGaussKernel(1,nd/2.0,1,nd/2.0)
    Kl3 = TSGaussGaussKernel(1,nd/4.0,1,nd/4.0)
    return [Kl1,Kl2,Kl3]

def set_MyTSGausKernel(X,X_mask,prop = 0.02,sampfreq =1.): 
    nd = X.shape[2]-1
    ns = np.mean(np.sum(X_mask.squeeze(),axis=-1))
    t_sig = max(np.around(np.mean(ns)*prop,0)*sampfreq,1*sampfreq)
    return MyTSGaussKernel(1,nd,nd)

class ClassificationExperiment: 

    def __init__(self,estimator_class,estimator_parameters,initializer_parameters,result_save_path="./results/msvc_result_var_sum.csv",verbose=True) -> None:
        self.estimator_class = estimator_class
        self.estimator_parameters = estimator_parameters
        self.initializer_parameters = initializer_parameters
        self.result_save_path = result_save_path
        self.verbose = verbose

    def start_experiment(self,dataloaders,seed=0): 
        estimator_name = self.estimator_class.__name__
        lst = []
        for dataset_name in dataset_names: 
            print(f"Starting: {dataset_name}")
            X_train,X_train_mask,X_test,X_test_mask,y_train,y_test = aeon_loader(dataset_name)
            lbe = LabelEncoder()
            y_train = lbe.fit_transform(y_train)
            y_test = lbe.transform(y_test)
            #X_train,X_train_mask,X_test,X_test_mask,y_train,y_test = X_train[:2],X_train_mask[:2],X_test[:2],X_test_mask[:2],y_train[:2],y_test[:2]
            #iKv = set_GaussKernel(X_train,X_train_mask)
            Kv = set_VFTSGausGaussKernel(X_train,X_train_mask)
            Kl = set_lst_TSGausGausKernel(X_train,X_train_mask)
            dataloss = SumVarifoldLoss(Kl)
            #barycenter_initalizer = batch_barycenter_initializer(iKv,dataloss,**self.initializer_parameters) 
            #registration_initializer = batch_time_initializer(iKv,dataloss,**self.initializer_parameters)
            estimator = self.estimator_class(Kv,dataloss,**self.estimator_parameters)
            
            start_time = time.time()
            estimator.fit(X_train,X_train_mask,y_train)
            ex_time = time.time()-start_time
            predictions = estimator.predict(X_test,X_test_mask)
            accuracy = accuracy_score(y_test,predictions)
            f1score = f1_score(y_test,predictions,average="macro")
            lst.append([estimator_name,dataset_name,accuracy,f1score,ex_time])
            
            if not self.result_save_path is None:
                self.df_ = pd.DataFrame(lst,columns=["estimator","dataset","accuracy","f1score","execution_time"])
                self.df_.to_csv(self.result_save_path)
            if self.verbose: 
                print(f"estimator: {estimator_name} -- dataset: {dataset_name} -- accuracy: {np.around(accuracy,2)} -- f1score: {np.around(f1score,2)}-- ex_time {ex_time}")
        return self 
    
if __name__ == "__main__": 
    ce = ClassificationExperiment(MomentaSVC,estimator_parameters,initializer_parameters)
    ce.start_experiment(dataloaders)
