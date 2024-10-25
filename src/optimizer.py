import optax
from jax import jit,grad,value_and_grad,vmap
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import partial


####################################################################################################################################
####################################################################################################################################
### GENERAL ###
####################################################################################################################################
####################################################################################################################################

def Optimizer(loss:callable,niter=100,optimizer = optax.adam(learning_rate=0.1),static_p0 =False, static_q0=True,verbose=True):
    def f(p0,q0,q0_mask,q1,q1_mask):
        
        if (static_p0==False)*(static_q0==True):
            arg = p0
            def map_loss(arg):
                return loss(arg,q0,q0_mask,q1,q1_mask)
        elif (static_p0==False)*(static_q0==False): 
            arg = (p0,q0)
            def map_loss(arg):
                return loss(*arg,q0_mask,q1,q1_mask)
        elif (static_p0==True)*(static_q0==False): 
            arg = q0
            def map_loss(arg): 
                return loss(p0,arg,q0_mask,q1,q1_mask)
            
        opt_state = optimizer.init(arg)
        jit_map_loss = jit(value_and_grad(map_loss))

        def step(arg,opt_state):
            loss_value,grads = jit_map_loss(arg) 
            updates, opt_state = optimizer.update(grads, opt_state,arg)
            arg = optax.apply_updates(arg,updates)
            return arg, opt_state, loss_value
        
        for i in range(niter): 
            arg,opt_state,loss_value = step(arg,opt_state)
            if verbose: 
                if (i==0)+((i+1)%10 == 0):
                    print("iteration: ",i+1,"/",niter, " -- loss: ", "{:0.2f}".format(loss_value))
        return arg
    return f 

####################################################################################################################################
####################################################################################################################################
### Registration ###
####################################################################################################################################
####################################################################################################################################


def RegistrationOptimizer(loss:callable,niter=100,optimizer = optax.adam(learning_rate=0.1),verbose=True):
    def f(p0,q0,q0_mask,q1,q1_mask):

        p=p0
        opt_state = optimizer.init(p)
        jit_loss = jit(value_and_grad(loss))

        def step(p,q0,q0_mask,q1,q1_mask,opt_state):
            loss_value,grad = jit_loss(p,q0,q0_mask,q1,q1_mask) 
            updates, opt_state = optimizer.update(grad, opt_state,p)
            p = optax.apply_updates(p,updates)
            return p, opt_state, loss_value
        
        for i in range(niter): 
            p,opt_state,loss_value = step(p,q0,q0_mask,q1,q1_mask,opt_state)
            if verbose: 
                if (i==0)+((i+1)%10 == 0):
                    print("iteration: ",i+1,"/",niter, " -- loss: ", "{:0.2f}".format(loss_value))
        return p
    return f 

def BatchOneToManyRegistrationOptimizer(loss:callable,niter=100,optimizer = optax.adam(learning_rate=0.1),verbose=True):
    def f(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask):

        n_batches,batch_size = batched_p0.shape[0],batched_p0.shape[1]
        batched_p=batched_p0
        opt_state = optimizer.init(batched_p)
        vmap_loss = vmap(loss, (0,None,None,0,0),0)
        def rloss(p0,q0,q0_mask,q1,q1_mask): 
            return jnp.sum(vmap_loss(p0,q0,q0_mask,q1,q1_mask))
        jit_loss = jit(value_and_grad(rloss))

        def step(batched_p,q,q_mask,batched_q1,batched_q1_mask,opt_state):

            p_grad = np.zeros_like(batched_p)
            loss_value = 0

            for i,(p,q1,q1_mask) in enumerate(zip(batched_p,batched_q1,batched_q1_mask)):
                t_loss_value,grads = jit_loss(p,q,q_mask,q1,q1_mask)
                p_grad[i] = grads
                loss_value += t_loss_value

           
            loss_value /= n_batches*batch_size
            updates, opt_state = optimizer.update(p_grad, opt_state,batched_p)
            batched_p = optax.apply_updates(batched_p,updates)

            return batched_p, opt_state, loss_value
        
        for i in range(niter): 
            batched_p,opt_state,loss_value = step(batched_p,q0,q0_mask,batched_q1,batched_q1_mask,opt_state)
            if verbose: 
                if (i==0)+((i+1)%10 == 0):
                    print("iteration: ",i+1,"/",niter, " -- loss: ", "{:0.2f}".format(loss_value))
        return batched_p
    return f 


####################################################################################################################################
####################################################################################################################################
### Barycenter ###
####################################################################################################################################
####################################################################################################################################

def BatchBarycenterTimeOptimizer(loss:callable,niter=100,optimizer = optax.adam(learning_rate=0.1),verbose=True):
    def f(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask):
        
        n_batches,batch_size = batched_p0.shape[0],batched_p0.shape[1]
        t_q0,s_q0 = q0[:,:1],q0[:,1:]
        batched_p,t_q = batched_p0, t_q0
        opt_state = optimizer.init((batched_p,t_q))
        jit_loss = jit(value_and_grad(loss,argnums=(0,1)))

        def step(batched_p,t_q,s_q,q_mask,batched_q1,batched_q1_mask,opt_state):

            p_grad = np.zeros_like(batched_p)
            loss_value = 0
            t_q_grad = np.zeros_like(t_q)

            for i,(p,q1,q1_mask) in enumerate(zip(batched_p,batched_q1,batched_q1_mask)):
                t_loss_value,grads = jit_loss(p,t_q,s_q,q_mask,q1,q1_mask)
                p_grad[i] = grads[0]
                t_q_grad += grads[1]/(n_batches*batch_size)
                loss_value += t_loss_value

            loss_value /= (n_batches*batch_size)
            updates, opt_state = optimizer.update((p_grad,t_q_grad), opt_state,(batched_p,t_q))
            batched_p,t_q = optax.apply_updates((batched_p,t_q),updates)
            return batched_p, t_q, opt_state, loss_value
        

        for i in range(niter): 
            batched_p,t_q,opt_state,loss_value = step(batched_p,t_q,s_q0,q0_mask,batched_q1,batched_q1_mask,opt_state)
            if verbose: 
                if (i==0)+((i+1)%10 == 0):
                    print("iteration: ",i+1,"/",niter, " -- loss: ", "{:0.2f}".format(loss_value))

        q = jnp.concatenate((t_q,s_q0),axis=-1)
        
        return batched_p,q
    return f 


def BatchBarycenterOptimizer(loss:callable,niter=100,optimizer = optax.adam(learning_rate=0.1),verbose=True):
    def f(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask):
        
        n_batches,batch_size = batched_p0.shape[0],batched_p0.shape[1]
        batched_p,q = batched_p0, q0
        opt_state = optimizer.init((batched_p,q))
        jit_loss = jit(value_and_grad(loss,argnums=(0,1)))

        def step(batched_p,q,q_mask,batched_q1,batched_q1_mask,opt_state):

            p_grad = np.zeros_like(batched_p)
            loss_value = 0
            q_grad = np.zeros_like(q)

            for i,(p,q1,q1_mask) in enumerate(zip(batched_p,batched_q1,batched_q1_mask)):
                t_loss_value,grads = jit_loss(p,q,q_mask,q1,q1_mask)
                p_grad[i] = grads[0]
                q_grad += grads[1]/(n_batches*batch_size)
                loss_value += t_loss_value

            loss_value /= (n_batches*batch_size)
            updates, opt_state = optimizer.update((p_grad,q_grad), opt_state,(batched_p,q))
            batched_p,q = optax.apply_updates((batched_p,q),updates)

            return batched_p, q, opt_state, loss_value
        

        for i in range(niter): 
            batched_p,q,opt_state,loss_value = step(batched_p,q,q0_mask,batched_q1,batched_q1_mask,opt_state)
            if verbose: 
                if (i==0)+((i+1)%10 == 0):
                    print("iteration: ",i+1,"/",niter, " -- loss: ", "{:0.2f}".format(loss_value))
        
        return batched_p,q
    return f 


def BatchIteratedBarycenterOptimizer(shoot:callable,loss:callable,niter=200, update_interval =100, optimizer = optax.adam(learning_rate=0.1),verbose=True):
    def f(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask):
        
        n_batches,batch_size = batched_p0.shape[0],batched_p0.shape[1]
        batched_p,q = batched_p0, q0
        opt_state = optimizer.init((batched_p))
        jit_loss = jit(value_and_grad(loss))

        def step(batched_p,q,q_mask,batched_q1,batched_q1_mask,opt_state):

            p_grad = np.zeros_like(batched_p)
            loss_value = 0

            for i,(p,q1,q1_mask) in enumerate(zip(batched_p,batched_q1,batched_q1_mask)):
                t_loss_value,grads = jit_loss(p,q,q_mask,q1,q1_mask)
                p_grad[i] = grads[0]
                loss_value += t_loss_value

            loss_value /= (n_batches*batch_size)
            updates, opt_state = optimizer.update(p_grad, opt_state,batched_p)
            batched_p = optax.apply_updates(batched_p,updates)

            return batched_p, opt_state, loss_value
        

        for i in range(niter): 
            batched_p,opt_state,loss_value = step(batched_p,q,q0_mask,batched_q1,batched_q1_mask,opt_state)
            if verbose: 
                if (i==0)+((i+1)%10 == 0):
                    print("iteration: ",i+1,"/",niter, " -- loss: ", "{:0.2f}".format(loss_value))
                if ((i+1)!=niter)*(i!=0)*((i+1)%update_interval == 0):
                    print("...Updating barycenter...")
                    pbar = np.mean(batched_p,axis=(0,1))
                    _,q = shoot(pbar,q,q0_mask)
                    batched_p = batched_p - pbar[None,:,:]        
        return batched_p,q
    return f 




