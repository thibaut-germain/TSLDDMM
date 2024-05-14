import numpy as np 
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, grad, vmap, jacfwd
import optax
from src.loss import VarifoldLoss
from src.optimizer import Optimizer, RegistrationOptimizer, BatchOneToManyRegistrationOptimizer
####################################################################################################################################
####################################################################################################################################
### GENERAL ###
####################################################################################################################################
####################################################################################################################################

def Hamiltonian(K:callable): 
    def H(p,q,q_mask): 
       return 0.5*jnp.sum(p*K(q,q_mask,q,q_mask,p))
    return H 

def HamiltonianSystem(K:callable):
    H = Hamiltonian(K)
    def HS(p,q,q_mask):
        Gp,Gq = grad(H,(0,1))(p,q,q_mask)
        return -Gq,Gp
    return HS

def HamiltonianRalstonIntegrator(K:callable,nt=10,deltat=1.): 
    HSystem = HamiltonianSystem(K)
    dt = deltat/nt
    def body_function(i,arg): 
        p,q,q_mask = arg
        dp,dq = HSystem(p,q,q_mask)
        pi,qi = p+(2*dt/3)*dp,q+(2*dt/3)*dq
        dpi,dqi = HSystem(pi,qi,q_mask)
        p,q = p+0.25*dt*(dp+3*dpi),q+0.25*dt*(dq+3*dqi)
        return p,q,q_mask
    def f(p,q,q_mask):
        p,q,q_mask = lax.fori_loop(0,nt,body_function,(p,q,q_mask))
        return p,q 
    return f

def Shooting(K:callable,nt=10,deltat=1.0): 
    Hri = HamiltonianRalstonIntegrator(K,nt,deltat)
    def shoot(p0,q0,q0_mask): 
        return Hri(p0,q0,q0_mask)
    return shoot

def LDDMMLoss(K:callable,dataloss:callable,gamma =0.001,nt=10,deltat=1.0): 
    Hm = Hamiltonian(K)
    shoot =Shooting(K,nt,deltat)
    def loss(p0,q0,q0_mask,q1,q1_mask): 
        p,q = shoot(p0,q0,q0_mask)
        return gamma * Hm(p0,q0,q0_mask) + dataloss(q,q0_mask,q1,q1_mask)
    return loss

def FlowRalstonIntegrator(K:callable,nt=10,deltat=1.): 
    HSystem = HamiltonianSystem(K)
    def FSystem(x,p,q,q_mask): 
        x_mask = jnp.full_like(x[:,:1],True,dtype=np.bool_)
        return (K(x,x_mask,q,q_mask,p),) + HSystem(p,q,q_mask)
    dt = deltat/nt
    def body_function(i,arg): 
        x,p,q,q_mask = arg
        dx,dp,dq = FSystem(x,p,q,q_mask)
        xi,pi,qi = x+(2*dt/3)*dx,p+(2*dt/3)*dp,q+(2*dt/3)*dq
        dxi,dpi,dqi = FSystem(xi,pi,qi,q_mask)
        x,p,q = x+0.25*dt*(dx+3*dxi),p+0.25*dt*(dp+3*dpi),q+0.25*dt*(dq+3*dqi)
        return x,p,q,q_mask
    def f(x0,p0,q0,q0_mask):
        x,p,q,q_mask = lax.fori_loop(0,nt,body_function,(x0,p0,q0,q0_mask))
        return x,p,q 
    return f

def Flowing(K:callable,nt=10,deltat=1.): 
    Fri = FlowRalstonIntegrator(K,nt,deltat)
    def flow(x0,p0,q0,q0_mask): 
        return Fri(x0,p0,q0,q0_mask)
    return flow

def DeformationGradient(K:callable,nt=10,deltat=1.0): 
    jacHri = jacfwd(HamiltonianRalstonIntegrator(K,nt,deltat),1)
    def gradient(p0,q0,q0_mask): 
        gr = jacHri(p0,q0,q0_mask)[1]
        idx = np.arange(gr.shape[0])
        gr = gr[idx,:,idx,:]-1
        return gr
    return jit(gradient)

def registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,p0 = None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    loss = LDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = RegistrationOptimizer(loss,niter,optimizer,verbose=verbose)
    if p0 is None:
        p0 = jnp.zeros_like(q0)
    if callable(p0):
        if verbose:
            print("Time initialization")
        p0 = p0(q0,q0_mask,q1,q1_mask)
    p = opt(p0,q0,q0_mask,q1,q1_mask)
    return p,q0,q0_mask

def varifold_registration(q0,q0_mask,q1,q1_mask,Kv,Kl,p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True): 
    dataloss = VarifoldLoss(Kl)
    return registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,p0,gamma_loss,niter,optimizer,nt,deltat,verbose=verbose)


def batch_one_to_many_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    loss = LDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = BatchOneToManyRegistrationOptimizer(loss,niter,optimizer,verbose=verbose)
    if batched_p0 is None:
        batched_p0 = jnp.zeros_like(batched_q1)
    if callable(batched_p0):
        batched_p0 = batched_p0(q0,q0_mask,batched_q1,batched_q1_mask)
    batched_p = opt(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask)
    return batched_p,q0,q0_mask


def batch_one_to_many_varifold_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,Kl,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    return batch_one_to_many_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0,gamma_loss,niter,optimizer,nt,deltat,verbose)
    


####################################################################################################################################
####################################################################################################################################
### TIME ###
####################################################################################################################################
####################################################################################################################################

def TimeLDDMMLoss(K:callable,dataloss:callable,gamma =0.001,nt=10,deltat=1.0): 
    Hm = Hamiltonian(K)
    shoot =Shooting(K,nt,deltat)
    def loss(p0,q0,q0_mask,q1,q1_mask): 
        t_q0, s_q0 = q0[:,:1],q0[:,1:]
        p,t_q = shoot(p0,t_q0,q0_mask)
        return gamma * Hm(p0,t_q0,q0_mask) + dataloss(jnp.hstack((t_q,s_q0)),q0_mask,q1,q1_mask)
    return loss


def TimeShooting(K:callable,nt=10,deltat=1.0): 
    Hri = HamiltonianRalstonIntegrator(K,nt,deltat)
    def shoot(p0,q0,q0_mask): 
        t_q0,s_q0 = q0[:,:1],q0[:,1:]
        p,t_q =Hri(p0,t_q0,q0_mask)
        q = jnp.hstack((t_q,s_q0))
        return p,q
    return shoot

def TimeFlowing(K:callable,nt=10,deltat=1.): 
    Fri = FlowRalstonIntegrator(K,nt,deltat)
    def flow(x0,p0,q0,q0_mask): 
        t_x0,s_x0 = x0[:,:1],x0[:,1:]
        t_q0,s_q0 = q0[:,:1],q0[:,1:]
        t_x,p,t_q = Fri(t_x0,p0,t_q0,q0_mask)
        x = jnp.hstack((t_x,s_x0))
        q = jnp.hstack((t_q,s_q0))
        return x,p,q
    return flow

def TimeDeformationGradient(K:callable,nt=10,deltat=1.0): 
    jacHri = jacfwd(HamiltonianRalstonIntegrator(K,nt,deltat),1)
    def gradient(p0,q0,q0_mask): 
        t_q0,s_q0 = q0[:,:1],q0[:,1:]
        gr =jacHri(p0,t_q0,q0_mask)[1]
        return np.diag(gr[:,0,:,0]).reshape(-1,1)-1
    return gradient

def time_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,gamma_loss=0.001,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    loss = TimeLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = RegistrationOptimizer(loss,niter,optimizer,verbose)
    p0 = jnp.zeros_like(q0[:,:1])
    p = opt(p0,q0,q0_mask,q1,q1_mask)
    return p,q0,q0_mask

def varifold_time_registration(q0,q0_mask,q1,q1_mask,Kv,Kl,gamma_loss=0.001,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    return time_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,gamma_loss,niter,optimizer,nt,deltat,verbose)

def batch_one_to_many_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    loss = TimeLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = BatchOneToManyRegistrationOptimizer(loss,niter,optimizer,verbose=verbose)
    if batched_p0 is None:
        batched_p0 = jnp.zeros_like(batched_q1[:,:,:,:1])
    batched_p = opt(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask)
    return batched_p,q0,q0_mask

def batch_one_to_many_varifold_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,Kl,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    return batch_one_to_many_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0,gamma_loss,niter,optimizer,nt,deltat,verbose)

def time_initializer(Kv,dataloss,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
    def initializer(q0,q0_mask,q1,q1_mask):
        p,q,qm = time_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,None,gamma_loss,niter,optimizer,nt,deltat,verbose)
        return jnp.pad(p,((0,0),(0,q.shape[1]-1)))
    return initializer

def varifold_time_initializer(Kv,Kl,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
    def initializer(q0,q0_mask,q1,q1_mask):
        p,q,qm = varifold_time_registration(q0,q0_mask,q1,q1_mask,Kv,Kl,gamma_loss,niter,optimizer,nt,deltat,verbose)
        return jnp.pad(p,((0,0),(0,q.shape[1]-1)))
    return initializer

def batch_time_initializer(Kv,dataloss,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
    def initializer(q0,q0_mask,batched_q1,batched_q1_mask):
        bp,bq,bqm = batch_one_to_many_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,None,gamma_loss,niter,optimizer,nt,deltat,verbose)
        return jnp.pad(bp,((0,0),(0,0),(0,0),(0,bq.shape[1]-1)))
    return initializer

def batch_varifold_time_initializer(Kv,Kl,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
    def initializer(q0,q0_mask,batched_q1,batched_q1_mask):
        bp,bq,bqm = batch_one_to_many_varifold_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,Kl,None,gamma_loss,niter,optimizer,nt,deltat,verbose)
        return jnp.pad(bp,((0,0),(0,0),(0,0),(0,bq.shape[1]-1)))
    return initializer



####################################################################################################################################
####################################################################################################################################
### SHAPE ###
####################################################################################################################################
####################################################################################################################################

def ShapeHamiltonian(K:callable):
    def H(p,t_q,s_q,q_mask):
        return 0.5*jnp.sum((p*K(t_q,s_q,q_mask,t_q,s_q,q_mask,p)))
    return H

def ShapeHamiltonianSystem(K:callable):
    H = ShapeHamiltonian(K)
    def HS(p,t_q,s_q,q_mask):
        Gp,Gq = grad(H,(0,2))(p,t_q,s_q,q_mask)
        return -Gq,Gp
    return HS

def ShapeHamiltonianRalstonIntegrator(K:callable,nt=10,deltat=1.): 
    HSystem = ShapeHamiltonianSystem(K)
    dt = deltat/nt
    def body_function(i,arg): 
        p,t_q,s_q,q_mask = arg
        dp,ds_q = HSystem(p,t_q,s_q,q_mask)
        pi,s_qi = p+(2*dt/3)*dp,s_q+(2*dt/3)*ds_q
        dpi,ds_qi = HSystem(pi,t_q,s_qi,q_mask)
        p,s_q = p+0.25*dt*(dp+3*dpi),s_q+0.25*dt*(ds_q+3*ds_qi)
        return p,t_q,s_q,q_mask 
    def f(p0,t_q0,s_q0,q0_mask):
        p,t_q,s_q,q_mask = lax.fori_loop(0,nt,body_function,(p0,t_q0,s_q0,q0_mask))
        q = jnp.hstack((t_q,s_q))
        return p,q
    return f

def ShapeShooting(K:callable,nt=10,deltat=1.0): 
    Hri = ShapeHamiltonianRalstonIntegrator(K,nt,deltat)
    def shoot(p0,q0,q0_mask): 
        t_q0, s_q0 = q0[:,:1],q0[:,1:]
        return Hri(p0,t_q0,s_q0,q0_mask)
    return shoot

def ShapeLDDMMLoss(K:callable,dataloss:callable,gamma =0.001,nt=10,deltat=1.0): 
    Hm = ShapeHamiltonian(K)
    shoot = ShapeShooting(K,nt,deltat)
    def loss(p0,q0,q0_mask,q1,q1_mask): 
        t_q0, s_q0 = q0[:,:1],q0[:,1:]
        _,q = shoot(p0,q0,q0_mask)
        return gamma * Hm(p0,t_q0,s_q0,q0_mask) + dataloss(q,q0_mask,q1,q1_mask)
    return loss

def ShapeFlowRalstonIntegrator(K:callable,nt=10,deltat=1.): 
    HSystem = ShapeHamiltonianSystem(K)
    def FSystem(t_x,s_x,p,t_q,s_q,q_mask): 
        x_mask = jnp.full_like(t_x,True,dtype=np.bool_)
        return (K(t_x,s_x,x_mask,t_q,s_q,q_mask,p),) + HSystem(p,t_q,s_q,q_mask)
    dt = deltat/nt
    def body_function(i,arg): 
        t_x,s_x,p,t_q,s_q,q_mask = arg
        ds_x,dp,ds_q = FSystem(t_x,s_x,p,t_q,s_q,q_mask)
        s_xi,pi,s_qi = s_x+(2*dt/3)*ds_x,p+(2*dt/3)*dp,s_q+(2*dt/3)*ds_q
        ds_xi,dpi,ds_qi = FSystem(t_x,s_xi,pi,t_q,s_qi,q_mask)
        s_x,p,s_q = s_x+0.25*dt*(ds_x+3*ds_xi),p+0.25*dt*(dp+3*dpi),s_q+0.25*dt*(ds_q+3*ds_qi)
        return t_x,s_x,p,t_q,s_q,q_mask
    def f(x0,p0,q0,q0_mask):
        t_x0,s_x0 = x0[:,:1],x0[:,1:]
        t_q0,s_q0 = q0[:,:1],q0[:,1:]
        t_x,s_x,p,t_q,s_q,q_mask = lax.fori_loop(0,nt,body_function,(t_x0,s_x0,p0,t_q0,s_q0,q0_mask))
        x,q = jnp.hstack((t_x,s_x)), jnp.hstack((t_q,s_q))
        return x,p,q 
    return f   

def ShapeFlowing(K:callable,nt=10,deltat=1.): 
    Fri = ShapeFlowRalstonIntegrator(K,nt,deltat)
    def flow(x0,p0,q0,q0_mask): 
        return Fri(x0,p0,q0,q0_mask)
    return flow

def ShapeDeformationGradient(K:callable,nt=10,deltat=1.0): 
    jacHri = jacfwd(ShapeHamiltonianRalstonIntegrator(K,nt,deltat),2)
    def gradient(p0,q0,q0_mask): 
        t_q0, s_q0 = q0[:,:1],q0[:,1:]
        gr = jacHri(p0,t_q0,s_q0,q0_mask)[1]
        idx = np.arange(gr.shape[0])
        gr = gr[idx,1:,idx,:]-1
        if gr.shape[-1] == 1: 
            gr = gr.reshape(-1,1)
        return gr
    return gradient

def shape_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,gamma_loss=0.001,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    loss = ShapeLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = RegistrationOptimizer(loss,niter,optimizer,verbose)
    p0 = jnp.zeros_like(q0[:,1:])
    p = opt(p0,q0,q0_mask,q1,q1_mask)
    return p,q0,q0_mask

def shape_varifold_registration(q0,q0_mask,q1,q1_mask,Kv,Kl,gamma_loss=0.001,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    return shape_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,gamma_loss,niter,optimizer,nt,deltat,verbose)

####################################################################################################################################
####################################################################################################################################
### TIME SHAPE ###
####################################################################################################################################
####################################################################################################################################

def TimeShapeShooting(t_Kv:callable,s_Kv:callable,t_nt=10,s_nt=10,t_deltat=1.0,s_deltat=1.0): 
    t_shoot = TimeShooting(t_Kv,t_nt,t_deltat)
    s_shoot = ShapeShooting(s_Kv,s_nt,s_deltat)
    def shoot(t_p0,s_p0,q0,q0_mask):
        t_p,q = t_shoot(t_p0,q0,q0_mask)
        s_p,q = s_shoot(s_p0,q,q0_mask)
        return t_p,s_p,q
    return shoot

def ShapeTimeShooting(t_Kv:callable,s_Kv:callable,t_nt=10,s_nt=10,t_deltat=1.0,s_deltat=1.0): 
    t_shoot = TimeShooting(t_Kv,t_nt,t_deltat)
    s_shoot = ShapeShooting(s_Kv,s_nt,s_deltat)
    def shoot(t_p0,s_p0,q0,q0_mask):
        s_p,q = s_shoot(s_p0,q0,q0_mask)
        t_p,q = t_shoot(t_p0,q,q0_mask)
        return t_p,s_p,q
    return shoot

def TimeShapeFlowing(t_Kv,s_Kv,t_nt=10,s_nt=10,t_deltat=1.0,s_deltat=1.0): 
    t_flow = TimeFlowing(t_Kv,t_nt,t_deltat)
    s_flow = ShapeFlowing(s_Kv,s_nt,s_deltat)
    def flow(x0,t_p0,s_p0,q0,q0_mask): 
        x,p,q = t_flow(x0,t_p0,q0,q0_mask)
        return s_flow(x,s_p0,q,q0_mask)
    return flow

def TimeShapeDeformationGradient(t_Kv,s_Kv,t_nt=10,s_nt=10,t_deltat=1.0, s_deltat=1.0): 
    Dt_gr = TimeDeformationGradient(t_Kv,t_nt,t_deltat)
    Ds_gr = ShapeDeformationGradient(s_Kv,s_nt,s_deltat)
    t_shoot = TimeShooting(t_Kv,t_nt,t_deltat)
    def gradient(t_p0,s_p0,q0,q0_mask): 
        tgr = Dt_gr(t_p0,q0,q0_mask)
        _,q = t_shoot(t_p0,q0,q0_mask)
        sgr = Ds_gr(s_p0,q,q0_mask)
        return tgr,sgr
    return gradient

def time_shape_registration(q0,q0_mask,q1,q1_mask,t_Kv,t_dataloss,s_Kv,s_dataloss,t_gamma_loss=0.001,s_gamma_loss=0.001,t_niter=100,s_niter=100,t_optimizer = optax.adam(learning_rate=0.1),s_optimizer = optax.adam(learning_rate=0.1),t_nt=10,s_nt=10,t_deltat=1.0,s_deltat=1.0,verbose=True): 
    t_p,_,_ = time_registration(q0,q0_mask,q1,q1_mask,t_Kv,t_dataloss,t_gamma_loss,t_niter,t_optimizer,t_nt,t_deltat,verbose) 
    shoot = TimeShooting(t_Kv,t_nt,t_deltat)
    _,i_q = shoot(t_p,q0,q0_mask)
    s_p,_,_ = shape_registration(i_q,q0_mask,q1,q1_mask,s_Kv,s_dataloss,s_gamma_loss,s_niter,s_optimizer,s_nt,s_deltat,verbose)
    return t_p,q0,s_p,i_q,q0_mask

def time_shape_varifold_registration(q0,q0_mask,q1,q1_mask,t_Kv,t_Kl,s_Kv,s_Kl,t_gamma_loss=0.001,s_gamma_loss=0.001,t_niter=100,s_niter=100,t_optimizer = optax.adam(learning_rate=0.1),s_optimizer = optax.adam(learning_rate=0.1),t_nt=10,s_nt=10,t_deltat=1.0,s_deltat=1.0,verbose=True):
    t_dataloss = VarifoldLoss(t_Kl)
    s_dataloss = VarifoldLoss(s_Kl)
    return time_shape_registration(q0,q0_mask,q1,q1_mask,t_Kv,t_dataloss,s_Kv,s_dataloss,t_gamma_loss,s_gamma_loss,t_niter,s_niter,t_optimizer,s_optimizer,t_nt,s_nt,t_deltat,s_deltat,verbose)



