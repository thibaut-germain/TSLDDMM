import pandas as pd
import numpy as np 
import sys
import optax
from optax.schedules import warmup_cosine_decay_schedule
sys.path.insert(0,"../")

from src.kernel import TSGaussGaussKernel, TSGaussKernel
from src.loss import SumVarifoldLoss
from src.barycenter import batch_barycenter_registration
from src.utils import batch_dataset

np.random.seed(0)

y =pd.read_csv("./dataset/y_600.csv",index_col=0)
X = np.load("./dataset/X_600.npy")
X_mask = np.load("./dataset/X_600_mask.npy")
X = X[:,::2,:]
X_mask = X_mask[:,::2,:]

lst = []
for filename in y[y.genotype.isin(["colq","wt"])].filename.unique(): 
    idx = y[(y.filename == filename)*(y.before == "Y")].sample(50).index
    lst.append(idx)
idxs = np.concatenate(lst)

bX,bX_mask = batch_dataset(X[idxs],1,X_mask[idxs])
y = y.iloc[idxs]

Kv = TSGaussKernel(150,2)
Kl1 = TSGaussGaussKernel(5,2,5,1)
Kl2 = TSGaussGaussKernel(2,1,2,0.6)
Kl3 = TSGaussGaussKernel(1,0.6,1,0.6)
Kls=[Kl1,Kl2,Kl3]
dataloss = SumVarifoldLoss(Kls)
schedule = warmup_cosine_decay_schedule(0,0.01,80,800,0)
optimizer = optax.adabelief(schedule)
  
p0s,q0,q0_mask = batch_barycenter_registration(bX,bX_mask,Kv,dataloss,niter=600,optimizer=optimizer,gamma_loss=1e-3)
np.save("./results/exp_1_lddmm_0/idxs.npy",idxs)
np.save("./results/exp_1_lddmm_0/p0s.npy",p0s)
np.save("./results/exp_1_lddmm_0/q0.npy",q0)
np.save("./results/exp_1_lddmm_0/q0_mask.npy",q0_mask)



################################################################################################################
################################################################################################################

import pandas as pd
import numpy as np 
import jax.numpy as jnp
import sys
import optax
from optax.schedules import warmup_cosine_decay_schedule
sys.path.insert(0,"../")


from src.kernel import TSGaussGaussKernel, TSGaussKernel
from src.loss import SumVarifoldLoss
from src.barycenter import batch_barycenter_registration
from src.utils import batch_dataset

np.random.seed(0)
y =pd.read_csv("./dataset/y_600.csv",index_col=0)
X = np.load("./dataset/X_600.npy")
X_mask = np.load("./dataset/X_600_mask.npy")
X = X[:,::2,:]
X_mask = X_mask[:,::2,:]

lst = []
for filename in y[y.genotype.isin(["colq","wt"])].filename.unique(): 
    idx = y[(y.before == "Y")*(y.filename == filename)].sample(25).index
    lst.append(idx)
    idx = y[(y.before == "N")*(y.filename == filename)].sample(50).index
    lst.append(idx)
idxs = np.concatenate(lst)

bX,bX_mask = batch_dataset(X[idxs],1,X_mask[idxs])
y = y.iloc[idxs]


Kv = TSGaussKernel(150,2)
Kl0 = TSGaussGaussKernel(30,2,30,1)
Kl1 = TSGaussGaussKernel(5,2,5,1)
Kl2 = TSGaussGaussKernel(2,1,2,0.6)
Kl3 = TSGaussGaussKernel(1,0.1,1,0.1)
Kls=[Kl0,Kl1,Kl2,Kl3]
dataloss = SumVarifoldLoss(Kls)
schedule = warmup_cosine_decay_schedule(0,0.01,80,800,0)
optimizer = optax.adabelief(schedule)

qi = jnp.load("./results/exp_1_1/q0.npy")
qi_mask = jnp.load("./results/exp_1_1/q0_mask.npy")


p0s,q0,q0_mask = batch_barycenter_registration(bX,bX_mask,Kv,dataloss,qi,qi_mask,niter=600,optimizer=optimizer,gamma_loss=1e-3)
np.save("./results/exp_2_lddmm_0/idxs.npy",idxs)
np.save("./results/exp_2_lddmm_0/p0s.npy",p0s)
np.save("./results/exp_2_lddmm_0/q0.npy",q0)
np.save("./results/exp_2_lddmm_0/q0_mask.npy",q0_mask)