import sys
sys.path.insert(0,"../")

import numpy as np
import pandas as pd
import optax
from optax.schedules import warmup_cosine_decay_schedule

from utils import Sampler,TASExperiment

from src.kernel import TSGaussGaussKernel,VFTSGaussKernel
from src.lddmm import Shooting
from src.utils import time_shape_embedding
from src.loss import SumVarifoldLoss

functional_signal = lambda x:  np.sin(2*np.pi*x)
q0 = time_shape_embedding(functional_signal(np.linspace(0,1,300)).reshape(-1,1),)
q0_mask = np.full_like(q0[:,:1],True).astype(bool)
Kv = VFTSGaussKernel(1,0.1,100,1,1)
shoot = Shooting(Kv)

spl = Sampler(q0,q0_mask,shoot,[5,10,15,20],[5,10,15,20],[10,50,100],0)
ps,qs,df= spl.rvs(50)

qs_mask = np.full_like(qs[:,:,:1],True).astype(bool)
Kls = [TSGaussGaussKernel(2,1,2,0.6)]
dataloss = SumVarifoldLoss(Kls)
tasexp = TASExperiment(Kv,q0,q0_mask,dataloss,20,gamma_loss=0,niter=400,optimizer=optax.adabelief(warmup_cosine_decay_schedule(0,0.05,40,400,0)))
tasexp.fit(ps,qs,qs_mask)

edf = tasexp.error_df_
fdf = pd.concat((df,edf),axis=1)
fdf.to_csv("./results/wellspecified.csv")