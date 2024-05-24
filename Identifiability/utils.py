import sys
sys.path.insert(0,"../")

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax.lax import map
from jax import vmap
from sklearn.utils import check_random_state
from functools import partial
from scipy.ndimage import convolve1d
import optax



from src.lddmm import Shooting,batch_one_to_many_registration
from src.utils import batch_dataset
from src.loss import MomentaLoss



class Sampler: 

    def __init__(self,q0,q0_mask,shoot,t_amp,s_amp,smoothness,random_state=None) -> None:
        self.q0 = q0 
        self.q0_mask = q0_mask
        self.shoot = shoot
        self.t_amp =t_amp
        self.s_amp = s_amp
        self.smoothness = smoothness
        self.random_state = check_random_state(random_state)

    def rvs(self,n_samples): 
        p_lst = []
        i_lst = []
        for smoothness in self.smoothness: 
            for t_amp in self.t_amp: 
                for s_amp in self.s_amp: 
                    p = self.random_state.randn(n_samples,*self.q0.shape)
                    p = convolve1d(p,np.ones(smoothness)/smoothness,axis=1,mode="constant")
                    t_norms = np.linalg.norm(p[:,:,0],axis=(1))
                    s_norms = np.linalg.norm(p[:,:,1:],axis=(1,2))
                    p[:,:,:1]*=t_amp/t_norms[:,None,None]
                    p[:,:,1:]*=s_amp/s_norms[:,None,None]
                    p_lst.append(p)
                    i_lst += [[smoothness,t_amp,s_amp]]*n_samples

        self.df_ = pd.DataFrame(i_lst,columns=["smothness", "t_amp", "s_amp"])
        self.p_ = np.concatenate(p_lst)
        self.q_ = map(partial(self.shoot,q0=self.q0,q0_mask=self.q0_mask),self.p_)[1]
        return self.p_, self.q_, self.df_

class TASExperiment: 

    def __init__(self,Kv,q0,q0_mask,dataloss,batch_size = 6,gamma_loss=0.001,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True) -> None:
        self.Kv = Kv 
        self.q0 = q0
        self.q0_mask = q0_mask
        self.dataloss = dataloss
        self.batch_size = batch_size
        self.gamma_loss = gamma_loss
        self.niter = niter
        self.optimizer = optimizer
        self.nt = nt
        self.deltat = deltat
        self.verbose = verbose

    def fit(self,ps,qs,qs_mask): 
        self.ps_ = ps
        self.qs_ = qs
        self.qs_mask_ = qs_mask 
        bqs, bqs_mask = batch_dataset(qs,self.batch_size,qs_mask) 
        shoot = lambda p : Shooting(self.Kv)(p,self.q0,self.q0_mask)[1]
        vmap_shoot = vmap(shoot)
        if self.verbose: 
            print("Starting")
        bps,_,_ = batch_one_to_many_registration(self.q0,self.q0_mask,bqs,bqs_mask,self.Kv,self.dataloss,None,self.gamma_loss,self.niter,self.optimizer,self.nt,self.deltat,self.verbose)
        self.p_ps_ = bps.reshape(-1,bps.shape[2],bps.shape[3])
        self.p_qs_ = vmap_shoot(self.p_ps_)
        if self.verbose: 
            print("Done")  
        return self
    
    def _varifold_error(self,p_qs,p_qs_mask,t_qs,t_qs_mask): 
        return vmap(self.dataloss,(0,0,0,0))(p_qs,p_qs_mask,t_qs,t_qs_mask)
    
    def _momenta_error(self,p_ps,t_ps): 
        return vmap(MomentaLoss(self.Kv,self.q0,self.q0_mask),(0,0))(p_ps,t_ps)
    
    @property
    def varifold_error_(self): 
        return self._varifold_error(self.p_qs_,self.qs_mask_,self.qs_,self.qs_mask_)
    
    @property
    def relative_varifold_error_(self): 
        return np.nan_to_num(self._varifold_error(self.p_qs_,self.qs_mask_,self.qs_,self.qs_mask_)/ self._varifold_error(np.zeros_like(self.qs_),self.qs_mask_,self.qs_,self.qs_mask_))
    
    @property
    def momenta_error_(self): 
        return self._momenta_error(self.p_ps_,self.ps_)
    
    @property
    def relative_momenta_error_(self): 
        return np.nan_to_num(self._momenta_error(self.p_ps_,self.ps_)/self._momenta_error(np.zeros_like(self.ps_,dtype=jnp.float32),self.ps_))

    @property
    def error_df_(self): 
        errors = np.vstack([self.varifold_error_,self.relative_varifold_error_,self.momenta_error_,self.relative_momenta_error_]).T
        df = pd.DataFrame(errors, columns = ["v_e","r_v_e", "m_e", "r_m_e"])
        return df