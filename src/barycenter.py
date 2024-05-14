import numpy as np
import jax.numpy as jnp
from jax import vmap,grad,jit
from jax.tree_util import Partial
import optax

from src.lddmm import LDDMMLoss,TimeLDDMMLoss
from src.loss import VarifoldLoss
from src.optimizer import Optimizer, BatchBarycenterOptimizer,BatchBarycenterTimeOptimizer

####################################################################################################################################
####################################################################################################################################
### GENERAL ###
####################################################################################################################################
####################################################################################################################################

def argmedian_length_from_mask(x):
    return np.argpartition(x, np.sum(x) // 2)[len(x) // 2]

def BarycenterLDDMMLoss(Kv,dataloss:callable,gamma_loss =0.,nt=10,deltat=1.0): 
    unit_loss = LDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    f = vmap(unit_loss,(0,None,None,0,0),0)
    def bloss(p0,q0,q0_mask,q1,q1_mask): 
        return jnp.sum(f(p0,q0,q0_mask,q1,q1_mask))
    return bloss

def barycenter_registration(sigs,sigs_masks,Kv,dataloss:callable,init=None,init_mask=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    if init is None: 
        median_length_idx = np.argsort(np.sum(sigs_masks,axis=1).flatten(),kind='stable')[sigs_masks.shape[0]//2]
        q0 = sigs[median_length_idx]
        q0_mask = sigs_masks[median_length_idx]
    else: 
        q0 = init
        q0_mask = init_mask
    p0 = jnp.zeros_like(sigs,dtype = jnp.float32)
    bloss = BarycenterLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = Optimizer(bloss,niter,optimizer,static_p0=False,static_q0=False,verbose=verbose)
    return *opt(p0,q0,q0_mask,sigs,sigs_masks),q0_mask

def varifold_barycenter_registration(sigs,sigs_masks,Kv,Kl,init=None,init_mask=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True): 
    dataloss = VarifoldLoss(Kl)
    return barycenter_registration(sigs,sigs_masks,Kv,dataloss,init,init_mask,niter,optimizer,gamma_loss,nt,deltat,verbose)

def batch_barycenter_registration(batched_sigs,batched_sigs_masks,Kv,dataloss:callable,init=None,init_mask=None,time_initializer=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    if init is None: 
        sigs_size = np.sum(batched_sigs_masks,axis=2)
        median_length_idx = np.unravel_index(np.argsort(sigs_size.flatten(),kind='stable')[sigs_size.shape[0]//2],sigs_size.shape)[:-1]
        q0 = batched_sigs[median_length_idx]
        q0_mask = batched_sigs_masks[median_length_idx]
    else: 
        q0 = init
        q0_mask = init_mask
    if time_initializer is None:
        batched_p0 = jnp.zeros_like(batched_sigs,dtype = jnp.float32)
    if callable(time_initializer): 
        if verbose: 
            print("Time initialization")
        batched_p0,q0,q0_mask = time_initializer(batched_sigs,batched_sigs_masks,q0,q0_mask)
    bloss = BarycenterLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = BatchBarycenterOptimizer(bloss,niter,optimizer,verbose)
    return *opt(batched_p0,q0,q0_mask,batched_sigs,batched_sigs_masks),q0_mask

def batch_varifold_barycenter_registration(batched_sigs,batched_sigs_masks,Kv,Kl,init=None,init_mask=None,time_initializer=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    return batch_barycenter_registration(batched_sigs,batched_sigs_masks,Kv,dataloss,init,init_mask,time_initializer,niter,optimizer,gamma_loss,nt,deltat,verbose)


####################################################################################################################################
####################################################################################################################################
### WITH TIME INITIALIZATION ###
####################################################################################################################################
####################################################################################################################################
    
def BarycenterTimeLDDMMLoss(Kv,dataloss:callable,gamma_loss =0.,nt=10,deltat=1.0): 
    unit_loss = TimeLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    f = vmap(unit_loss,(0,None,None,0,0),0)
    def bloss(p0,t_q0,s_q0,q0_mask,q1,q1_mask): 
        q0 = jnp.concatenate((t_q0,s_q0),axis=-1)
        return jnp.sum(f(p0,q0,q0_mask,q1,q1_mask))
    return bloss


def batch_barycenter_time_registration(batched_sigs,batched_sigs_masks,Kv,dataloss:callable,init=None,init_mask=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    
    if init is None: 
        sigs_size = np.sum(batched_sigs_masks,axis=2)
        median_length_idx = np.unravel_index(np.argsort(sigs_size.flatten(),kind='stable')[sigs_size.shape[0]//2],sigs_size.shape)[:-1]
        q0 = batched_sigs[median_length_idx]
        q0_mask = batched_sigs_masks[median_length_idx]
    else: 
        q0 = init
        q0_mask = init_mask

    batched_p0 = jnp.zeros_like(batched_sigs[:,:,:,:1],dtype = jnp.float32)
    bloss = BarycenterTimeLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = BatchBarycenterTimeOptimizer(bloss,niter,optimizer,verbose)
    return *opt(batched_p0,q0,q0_mask,batched_sigs,batched_sigs_masks),q0_mask

def batch_varifold_barycenter_time_registration(batched_sigs,batched_sigs_masks,Kv,Kl,init=None,init_mask=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    return batch_barycenter_time_registration(batched_sigs,batched_sigs_masks,Kv,dataloss,init,init_mask,niter,optimizer,gamma_loss,nt,deltat,verbose)


def batch_barycenter_initializer(Kv,dataloss:callable,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    def initializer(batched_sigs,batched_sigs_masks,init,init_mask):
        batched_p,q,qm = batch_barycenter_time_registration(batched_sigs,batched_sigs_masks,Kv,dataloss,init,init_mask,niter,optimizer,gamma_loss,nt,deltat,verbose)
        batched_p = jnp.pad(batched_p,((0,0),(0,0),(0,0),(0,q.shape[1]-1)))
        return batched_p,q,qm
    return initializer


def batch_varifold_barycenter_initializer(Kv,Kl,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    def initializer(batched_sigs,batched_sigs_masks,init,init_mask):
        batched_p,q,qm = batch_barycenter_time_registration(batched_sigs,batched_sigs_masks,Kv,dataloss,init,init_mask,niter,optimizer,gamma_loss,nt,deltat,verbose)
        batched_p = jnp.pad(batched_p,((0,0),(0,0),(0,0),(0,q.shape[1]-1)))
        return batched_p,q,qm
    return initializer