import jax.numpy as jnp
import numpy as np
import jax


def compCLN(x:jnp.ndarray): 
    x0, x1 = x[:-1,:], x[1:,:]
    p = .5*(x0+x1)
    v = (x1-x0)
    m = jnp.sum(v**2,axis=1).reshape(-1,1)
    m = jnp.sqrt(jnp.where(m>0,m,0))
    v = v*jnp.where(m==0.,0.,1./m)
    return m,p,v


def compELT(x:jnp.ndarray): 
    t_x0, s_x0, t_x1, s_x1 = x[:-1,:1], x[:-1,1:], x[1:,:1],x[1:,1:]
    t_p, s_p = .5*(t_x0+t_x1), .5*(s_x0+s_x1)
    t_v = t_x1 - t_x0
    s_v = (s_x1-s_x0)*jnp.where(t_v==0.,0.,1/t_v)
    return t_p,s_p,t_v,s_v


def MyLoss(K:callable): 
    def loss(s_x,mask_s_x,t_x,mask_t_x):
        t_t_p,t_s_p,t_t_v,t_s_v = compELT(t_x)
        s_t_p,s_s_p,s_t_v,s_s_v = compELT(s_x)
        c0 = jnp.sum(t_t_v * K(t_t_p,t_s_p,t_s_v,mask_t_x[1:,:],t_t_p,t_s_p,t_s_v,mask_t_x[1:,:],t_t_v))
        c1 = jnp.sum(s_t_v * K(s_t_p,s_s_p,s_s_v,mask_s_x[1:,:],t_t_p,t_s_p,t_s_v,mask_t_x[1:,:],t_t_v))
        c2 = jnp.sum(s_t_v * K(s_t_p,s_s_p,s_s_v,mask_s_x[1:,:],s_t_p,s_s_p,s_s_v,mask_s_x[1:,:],s_t_v))
        return c0 -2*c1 +c2
    return loss



def VarifoldLoss(K:callable): 
    def loss(s_x,mask_s_x,t_x,mask_t_x): 
        t_m,t_p,t_v = compCLN(t_x)
        s_m,s_p,s_v = compCLN(s_x)
        c0 = jnp.sum(t_m * K(t_p,t_p,t_v,t_v,mask_t_x[1:,:],mask_t_x[1:,:],t_m))
        c1 = jnp.sum(s_m * K(s_p,t_p,s_v,t_v,mask_s_x[1:,:],mask_t_x[1:,:],t_m))
        c2 = jnp.sum(s_m * K(s_p,s_p,s_v,s_v,mask_s_x[1:,:],mask_s_x[1:,:],s_m))
        return c0 -2*c1 + c2
    return loss

def SumVarifoldLoss(K_lst:callable): 
    def loss(s_x,mask_s_x,t_x,mask_t_x): 
        t_m,t_p,t_v = compCLN(t_x)
        s_m,s_p,s_v = compCLN(s_x)
        score = 0
        for K in K_lst:
            c0 = jnp.sum(t_m * K(t_p,t_p,t_v,t_v,mask_t_x[1:,:],mask_t_x[1:,:],t_m))
            c1 = jnp.sum(s_m * K(s_p,t_p,s_v,t_v,mask_s_x[1:,:],mask_t_x[1:,:],t_m))
            c2 = jnp.sum(s_m * K(s_p,s_p,s_v,s_v,mask_s_x[1:,:],mask_s_x[1:,:],s_m))
            score +=  c0 -2*c1 + c2
        return score
    return loss

def MMD(K:callable): 
    def loss(s_x,mask_s_x,t_x,mask_t_x): 
        ns = s_x.shape[0]
        s_ones = np.ones((ns,1))/np.sum(mask_s_x).astype(np.float32)
        t_ones = np.ones((ns,1))/np.sum(mask_t_x).astype(np.float32)

        c0 = jnp.sum(t_ones*K(t_x,mask_t_x,t_x,mask_t_x,t_ones))
        c1 = jnp.sum(s_ones*K(s_x,mask_s_x,t_x,mask_t_x,t_ones))
        c2 = jnp.sum(s_ones*K(s_x,mask_s_x,s_x,mask_s_x,s_ones))
        return c0 -2*c1 + c2
    return loss

def WeightedMMD(K:callable): 
    def loss(s_x,mask_s_x,t_x,mask_t_x): 
        t_m,t_p,t_v = compCLN(t_x)
        s_m,s_p,s_v = compCLN(s_x)
        c0 = jnp.sum(t_m * K(t_p,mask_t_x[1:,:],t_p,mask_t_x[1:,:],t_m))
        c1 = jnp.sum(s_m * K(s_p,mask_s_x[1:,:],t_p,mask_t_x[1:,:],t_m))
        c2 = jnp.sum(s_m * K(s_p,mask_s_x[1:,:],s_p,mask_s_x[1:,:],s_m))
        return c0 -2*c1 + c2
    return loss

def MomentaLoss(K:callable,q0:jnp.ndarray,q0_mask): 
    def loss(s_p,t_p): 
        c0 = jnp.sum(s_p * K(q0,q0_mask,q0,q0_mask,s_p))
        c1 = jnp.sum(t_p * K(q0,q0_mask,q0,q0_mask,s_p))
        c2 = jnp.sum(t_p * K(q0,q0_mask,q0,q0_mask,t_p))
        return c0 -2*c1 + c2
    return loss