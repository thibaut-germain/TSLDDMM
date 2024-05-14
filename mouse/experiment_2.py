import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sys
import optax
from optax.schedules import warmup_cosine_decay_schedule
sys.path.insert(0,"../")

from src.masked.lddmm import registration,Shooting,Flowing
from src.masked.kernel import VFTSGaussKernel,TSGaussGaussKernel
from src.masked.loss import VarifoldLoss
from src.plotting import plot2Dfigure
from src.masked.barycenter import batch_barycenter_registration
from src.utils import batch_dataset

np.random.seed(0)
y =pd.read_csv("./dataset/inv_y.csv",index_col=0)
X = np.load("./dataset/inv_X.npy")
X_mask = np.load("./dataset/inv_X_mask.npy")
X = X[:,::2,:]
X_mask = X_mask[:,::2,:]
Kv = VFTSGaussKernel(1,0.1,150,1,1)
Kl = TSGaussGaussKernel(5,1,5,0.6)
dataloss = VarifoldLoss(Kl)
schedule = warmup_cosine_decay_schedule(0,0.3,80,800,0)
optimizer = optax.adabelief(schedule)

lst = []
for filename in y[y.genotype.isin(["colq","wt"])].filename.unique(): 
    idx = y[(y.before == "Y")*(y.filename == filename)].sample(25).index
    lst.append(idx)
    idx = y[(y.before == "N")*(y.filename == filename)].sample(50).index
    lst.append(idx)
idxs = np.concatenate(lst)

bX,bX_mask = batch_dataset(X[idxs],1,X_mask[idxs])
y = y.iloc[idxs]

if __name__ == "__main__": 

    p0s,q0,q0_mask = batch_barycenter_registration(bX,bX_mask,Kv,dataloss,niter=800,optimizer=optimizer,gamma_loss=1e-4)

    np.save("./results/exp_2_3/idxs.npy",idxs)
    np.save("./results/exp_2_3/p0s.npy",p0s)
    np.save("./results/exp_2_3/q0.npy",q0)
    np.save("./results/exp_2_3/q0_mask.npy",q0_mask)