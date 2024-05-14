import numpy as np
import jax.numpy as jnp


def GaussKernel(sigma,mu=1): 
    oos2 = 1/sigma**2
    def K(x,mask_x,y,mask_y,b):
        res = jnp.exp(-oos2*jnp.sum((x[:,None,:]-y[None,:,:])**2,axis=2))* (mask_x*mask_y.T)
        return mu*res@b
    return K


def ExpKernel(sigma,mu=1): 
    oos2 = 1/sigma
    def K(x,mask_x,y,mask_y,b):
        m = jnp.sum((x[:,None,:]-y[None,:,:])**2,axis=2)
        m = jnp.sqrt(jnp.where(m>0,m,0))
        res = jnp.exp(-oos2*m)* (mask_x*mask_y.T)
        return mu*res@b
    return K

def TSGaussKernel(t_sigma:float,s_sigma:float,np_dtype = np.float32): 
    t_oos, s_oos  = 1/t_sigma, 1/s_sigma
    def K(x,mask_x,y,mask_y,b): 
        n_d = x.shape[1]
        oos = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos[0,0], oos[0,1:] = t_oos, s_oos
        res = jnp.exp(-jnp.sum(((x*oos)[:,None,:]-(y*oos)[None,:,:])**2,axis=2))
        mask = (mask_x*mask_y.T)
        return (res*mask)@b
    return K

def TSGaussGaussKernel(t_sigma_1:float,s_sigma_1:float,t_sigma_2:float,s_sigma_2:float,np_dtype = np.float32): 
    t_oos_1, s_oos_1  = 1/t_sigma_1, 1/s_sigma_1
    t_oos_2, s_oos_2 = 1/t_sigma_2, 1/s_sigma_2
    def K(x,y,u,v,mask_xu,mask_yv,b): 
        n_d = x.shape[1]
        oos_1 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_1[0,0], oos_1[0,1:] = t_oos_1, s_oos_1
        k1 = jnp.exp(-jnp.sum(((x*oos_1)[:,None,:]-(y*oos_1)[None,:,:])**2,axis=2))
        oos_2 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_2[0,0], oos_2[0,1:] = t_oos_2,s_oos_2
        k2 = jnp.exp(-jnp.sum(((u*oos_2)[:,None,:]-(v*oos_2)[None,:,:])**2,axis=2))
        mask = (mask_xu*mask_yv.T)
        return (k1*k2*mask)@b
    return K

def TSExpExpKernel(t_sigma_1:float,s_sigma_1:float,t_sigma_2:float,s_sigma_2:float,np_dtype = np.float32): 
    t_oos_1, s_oos_1  = 1/t_sigma_1, 1/s_sigma_1
    t_oos_2, s_oos_2 = 1/t_sigma_2, 1/s_sigma_2
    def K(x,y,u,v,mask_xu,mask_yv,b): 
        n_d = x.shape[1]
        oos_1 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_1[0,0], oos_1[0,1:] = t_oos_1, s_oos_1
        m1 = jnp.sum(((x*oos_1)[:,None,:]-(y*oos_1)[None,:,:])**2,axis=2)
        m1 = jnp.sqrt(jnp.where(m1>0,m1,0))
        k1 = jnp.exp(-m1)
        oos_2 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_2[0,0], oos_2[0,1:] = t_oos_2,s_oos_2
        m2 = jnp.sum(((u*oos_2)[:,None,:]-(v*oos_2)[None,:,:])**2,axis=2)
        m2 = jnp.sqrt(jnp.where(m2>0,m2,0))
        k2 = jnp.exp(-m2)
        mask = (mask_xu*mask_yv.T)
        return (k1*k2*mask)@b
    return K


def TSModifKernel(t_sigma_1:float,s_sigma_1:float,t_sigma_2:float,s_sigma_2:float,np_dtype = np.float32): 
    t_oos_1, s_oos_1  = 1/t_sigma_1, 1/s_sigma_1
    t_oos_2, s_oos_2 = 1/t_sigma_2, 1/s_sigma_2
    def K(x,y,u,v,mask_xu,mask_yv,b): 
        t_x, s_x = x[:,:1], x[:,1:]
        t_y, s_y = y[:,:1], y[:,1:]
        t_u, s_u = u[:,:1], u[:,1:]
        t_v, s_v = v[:,:1], v[:,1:]
        t_xy_k = jnp.exp(-t_oos_1*jnp.sqrt(jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2)))
        s_xy_k = jnp.exp(-s_oos_1*jnp.sqrt(jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2)))
        k1 = t_xy_k*s_xy_k
        t_uv_k = jnp.exp(-t_oos_2*jnp.sqrt(jnp.sum((t_u[:,None,:]-t_v[None,:,:])**2,axis=2)))
        s_uv_k = jnp.exp(-s_oos_2*jnp.sqrt(jnp.sum((s_u[:,None,:]-s_v[None,:,:])**2,axis=2)))
        k2 = t_uv_k*s_uv_k
        mask = (mask_xu*mask_yv.T)
        return (k1*k2*mask)@b
    return K

def TSGaussDotKernel(t_sigma_1:float,s_sigma_1:float,t_sigma_2:float,s_sigma_2:float,power:int,np_dtype=np.float32):
    t_oos_1, s_oos_1 = 1/t_sigma_1, 1/s_sigma_1
    t_oos_2, s_oos_2 = 1/t_sigma_2, 1/s_sigma_2
    def K(x,y,u,v,mask_xu,mask_yv,b): 
        n_d = x.shape[1]
        oos_1 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_1[0,0], oos_1[0,1:] = t_oos_1, s_oos_1
        k1 = jnp.exp(-jnp.sum(((x*oos_1)[:,None,:]-(y*oos_1)[None,:,:])**2,axis=2))
        oos_2 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_2[0,0], oos_2[0,1:] = t_oos_2,s_oos_2
        k2 = jnp.sum(((u*oos_2)[:,None,:]*(v*oos_2)[None,:,:]),axis=2)**power
        mask = (mask_xu*mask_yv.T)
        return (k1*k2*mask)@b
    return K 


def VFTSGaussKernel(mu:float,lmbda:float,t_sigma_1:float,t_sigma_2:float,s_sigma:float): 
    t_oos_12 = 1/t_sigma_1**2
    t_oos_22 = 1/t_sigma_2**2
    s_oos2 = 1/s_sigma**2
    def K(x,mask_x,y,mask_y,b): 
        t_x, s_x = x[:,:1], x[:,1:]
        t_y, s_y = y[:,:1], y[:,1:]
        t_b, s_b = b[:,:1], b[:,1:]
        time_sum = jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2)
        t_res_1 = jnp.exp(-t_oos_12*time_sum)
        s_res = jnp.exp(-s_oos2*jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2)-t_oos_22*time_sum)
        mask = mask_x*mask_y.T
        return jnp.hstack((mu*(mask*t_res_1)@(t_b*mask_y), lmbda*(mask*s_res)@(s_b*mask_y)))
    return K 

def VFSGaussKernel(t_sigma:float,s_sigma:float): 
    t_oos2 = 1/t_sigma**2
    s_oos2 = 1/s_sigma**2 
    def K(t_x,s_x,mask_x,t_y,s_y,mask_y,s_b): 
        t_res = jnp.exp(-t_oos2*jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2))
        s_res = jnp.exp(-s_oos2*jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2))
        mask = mask_x*mask_y.T
        return (t_res*s_res*mask)@(s_b*mask_y)
    return K 

def VFTSCauchyKernel(mu:float,lmbda:float,t_sigma_1:float,t_sigma_2:float,s_sigma:float): 
    t_oos_12 = 1/t_sigma_1**2
    t_oos_22 = 1/t_sigma_2**2
    s_oos2 = 1/s_sigma**2
    def K(x,mask_x,y,mask_y,b): 
        t_x, s_x = x[:,:1], x[:,1:]
        t_y, s_y = y[:,:1], y[:,1:]
        t_b, s_b = b[:,:1], b[:,1:]
        time_sum = jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2)
        t_res_1 = 1/(1+t_oos_12*time_sum)
        s_res = 1/(1+s_oos2*jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2)+t_oos_22*time_sum)
        mask = mask_x*mask_y.T
        return jnp.hstack((mu*(mask*t_res_1)@(t_b*mask_y), lmbda*(mask*s_res)@(s_b*mask_y)))
    return K 

def VFTSExpKernel(mu:float,lmbda:float,t_sigma_1:float,t_sigma_2:float,s_sigma:float): 
    t_oos_12 = 1/t_sigma_1
    t_oos_22 = 1/t_sigma_2
    s_oos2 = 1/s_sigma
    def K(x,mask_x,y,mask_y,b): 
        t_x, s_x = x[:,:1], x[:,1:]
        t_y, s_y = y[:,:1], y[:,1:]
        t_b, s_b = b[:,:1], b[:,1:]
        time_sum = jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2)
        time_sum = jnp.sqrt(jnp.where(time_sum>0,time_sum,0))
        t_res_1 = jnp.exp(-t_oos_12*time_sum)
        m = jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2)
        m = jnp.sqrt(jnp.where(m>0,m,0))
        s_res = jnp.exp(-s_oos2*m-t_oos_22*time_sum)
        mask = mask_x*mask_y.T
        return jnp.hstack((mu*(mask*t_res_1)@(t_b*mask_y), lmbda*(mask*s_res)@(s_b*mask_y)))
    return K 


def MyTSGaussKernel(t_sigma,s_sigma,v_sigma):
    t_oos = 1/t_sigma**2
    s_oos = 1/s_sigma**2
    v_oos = 1/v_sigma**2
    def K(t_x,s_x,v_x,x_mask,t_y,s_y,v_y,y_mask,b):
        tK = jnp.exp(-t_oos*jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2))
        sK = jnp.exp(-s_oos*jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2))
        vK = jnp.exp(-v_oos*jnp.sum((v_x[:,None,:]-v_y[None,:,:])**2,axis=2))
        mask = x_mask*y_mask.T
        return (tK*sK*vK*mask)@b
    return K

