import numpy as np
import jax.numpy as jnp
from jax import vmap 

def ComputeKernelGramMatrix(K):
    f = lambda x,y,u,v : jnp.sum(u*K(x,y,v))
    f = vmap(f,(None,None,0,None),0)
    f = vmap(f,(None,None,None,0),1)
    return f

def time_shape_embedding(x:np.ndarray,sampfreq=1,max_length=None,dtype = jnp.float32): 
    n_s = x.shape[0]
    time = np.arange(n_s,dtype=float).reshape(-1,1)/sampfreq
    t_x = np.hstack((time,x),dtype=dtype)
    if max_length is None:
        return t_x
    else:
        t_x = np.pad(t_x,((0,max_length-n_s),(0,0))).astype(dtype)
        mask = np.full_like(t_x[:,:1],True,dtype=bool)
        mask[n_s:,:] = False
        return t_x, mask

def from_timeseries_to_dataset(X:list,sampfreq=1,dtype=jnp.float32): 
    lengths = np.array([x.shape[0] for x in X])
    max_length = np.max(lengths)
    mask_lst = []
    ts_lst = []
    for ts in X: 
        t_ts, t_mask = time_shape_embedding(ts,sampfreq,max_length,dtype=dtype)
        ts_lst.append(t_ts)
        mask_lst.append(t_mask)
    return jnp.array(ts_lst,dtype=dtype), jnp.array(mask_lst,dtype=jnp.bool_)
    
def batch_dataset(dataset,batch_size,masks=None,dtype=jnp.float32): 
    if dataset.shape[0]%batch_size!=0: 
        raise ValueError("dataset size is not a multiple of batch_size")
    n_ts,n_s,n_d = dataset.shape
    n_batches = n_ts//batch_size
    if masks is None: 
        return dataset.reshape(n_batches,batch_size,n_s,n_d).astype(dtype)
    else: 
        return dataset.reshape(n_batches,batch_size,n_s,n_d).astype(dtype), masks.reshape(n_batches,batch_size,n_s,1)
    
def unbatch_dataset(batched_dataset,batched_masks=None,dtype=jnp.float32): 
    _,_,n_s,n_d = batched_dataset.shape
    if batched_masks is None: 
        return batched_dataset.reshape(-1,n_s,n_d).astype(dtype)
    else: 
        return batched_dataset.reshape(-1,n_s,n_d).astype(dtype), batched_masks.reshape(-1,n_s,1)
    
def MaskMomentaGramMatrix(K,qbar,qbar_mask):
    f = lambda u,v : jnp.sum(u*K(qbar,qbar_mask,qbar,qbar_mask,v))
    f = vmap(f,(0,None),0)
    f = vmap(f,(None,0),1)
    return f