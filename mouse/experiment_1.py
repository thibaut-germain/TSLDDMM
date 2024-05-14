import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sys
import optax
from optax.schedules import warmup_cosine_decay_schedule
sys.path.insert(0,"../")


from src.kernel import VFTSGaussKernel,TSGaussGaussKernel
from src.loss import VarifoldLoss,SumVarifoldLoss
from src.barycenter import batch_barycenter_registration
from src.utils import batch_dataset

np.random.seed(0)

y =pd.read_csv("./dataset/y.csv",index_col=0)
X = np.load("./dataset/X.npy")
X_mask = np.load("./dataset/X_mask.npy")
X = X[:,::2,:]
X_mask = X_mask[:,::2,:]

lst = []
for filename in y[y.genotype.isin(["colq","wt"])].filename.unique(): 
    idx = y[(y.filename == filename)*(y.before == "Y")].sample(50).index
    lst.append(idx)
idxs = np.concatenate(lst)

bX,bX_mask = batch_dataset(X[idxs],1,X_mask[idxs])
y = y.iloc[idxs]

Kv = VFTSGaussKernel(1,0.1,100,1,1)
Kl1 = TSGaussGaussKernel(2,2,2,2)
Kl2 = TSGaussGaussKernel(1,1,1,1)
Kl3 = TSGaussGaussKernel(1,0.1,1,0.1)
Kls=[Kl1,Kl2,Kl3]
dataloss = SumVarifoldLoss(Kls)
schedule = warmup_cosine_decay_schedule(0,0.3,40,400,0)
optimizer = optax.adabelief(schedule)


if __name__ == "__main__": 

  
    p0s,q0,q0_mask = batch_barycenter_registration(bX,bX_mask,Kv,dataloss,niter=400,optimizer=optimizer,gamma_loss=1e-3)

    np.save("./results/exp_1_0/idxs.npy",idxs)
    np.save("./results/exp_1_0/p0s.npy",p0s)
    np.save("./results/exp_1_0/q0.npy",q0)
    np.save("./results/exp_1_0/q0_mask.npy",q0_mask)