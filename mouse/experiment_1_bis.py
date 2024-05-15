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
idxs = np.load("./results/exp_1_0/idxs.npy")

X = X[:,::2,:]
X_mask = X_mask[:,::2,:]
bX,bX_mask = batch_dataset(X[idxs],1,X_mask[idxs])
y = y.iloc[idxs]

print(y.before.unique())
print(y.genotype.unique())


if __name__ == "__main__": 

    Kv = VFTSGaussKernel(1,0.1,100,1,1)
    Kv = VFTSGaussKernel(1,0.1,100,1,1)
    Kl1 = TSGaussGaussKernel(2,2,2,2)
    Kl2 = TSGaussGaussKernel(1,1,1,1)
    Kl3 = TSGaussGaussKernel(1,0.1,1,0.1)
    Kls=[Kl1,Kl2,Kl3]
    dataloss = SumVarifoldLoss(Kls)
    schedule = warmup_cosine_decay_schedule(0,0.3,40,400,0)
    optimizer = optax.adabelief(schedule)



    for i,filename in enumerate(y.filename.unique()): 
       
        print(f"Mouse: {i+1}/{y.filename.unique().shape[0]} -- {filename}")
        tidxs = y[y.filename == filename].index
        bX,bX_mask = batch_dataset(X[tidxs],1,X_mask[tidxs])
        p0s,q0,q0_mask = batch_barycenter_registration(bX,bX_mask,Kv,dataloss,niter=400,optimizer=optimizer,gamma_loss=1e-3)

        np.save("./results/exp_1_0/"+f"{filename[:-4]}_p0s.npy",p0s)
        np.save("./results/exp_1_0/"+f"{filename[:-4]}_q0.npy",q0)
        np.save("./results/exp_1_0/"+f"{filename[:-4]}_q0_mask.npy",q0_mask)

    print("Done")