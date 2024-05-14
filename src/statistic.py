import numpy as np
import jax.numpy as jnp
from jax import vmap,jit
from jax.lax import map 
from sklearn.base import BaseEstimator
from numpy.linalg import svd
from scipy.linalg import solve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from functools import partial
from scipy.linalg import sqrtm,inv

def MomentaGramMatrix(K,qbar):
    f = lambda u,v : jnp.sum(u*K(qbar,qbar,v))
    f = vmap(f,(0,None),0)
    f = vmap(f,(None,0),1)
    return f

def MaskMomentaGramMatrix(K,qbar,qbar_mask):
    f = lambda u,v : jnp.sum(u*K(qbar,qbar_mask,qbar,qbar_mask,v))
    f = vmap(f,(0,None),0)
    f = vmap(f,(None,0),1)
    return jit(f)

def MapMaskMomentaGramMatrix(K,qbar,qbar_mask):
    f = lambda u,v : jnp.sum(u*K(qbar,qbar_mask,qbar,qbar_mask,v))
    def mapf(u,v): 
        nu = u.shape[0]
        nv = v.shape[0]
        arr = np.zeros((nu,nv))
        for i in range(nu):
            for j in range(i,nv): 
                val = f(u[i],v[j])
                arr[i,j] = val
                arr[j,i] = val
        return arr 
    return mapf




class MomentaPCA(BaseEstimator):

    def __init__(self,n_component,centered=False,type = "cov") -> None:
        super().__init__()
        self.n_component = n_component
        self.centered = centered
        self.type = type    


    def fit(self,Kv,ps,qbar,qbar_mask): 

        self.Kv_ = Kv
        self.qbar_ = qbar
        self.ps_ = ps

    
        self._mgm = MaskMomentaGramMatrix(self.Kv_,qbar,qbar_mask)
       

        if self.centered:
            r_ps = ps
        else:
            self.m_ps_ = np.mean(ps,axis=0)
            r_ps = ps - self.m_ps_[None,:,:]

        X = self._mgm(r_ps,r_ps)

        if self.type == "cor": 
            inv_norm = 1/np.sqrt(np.diag(X))
            X= X*(inv_norm.reshape(-1,1)*inv_norm.reshape(1,-1))
        
        U,S,Vh = svd(X,full_matrices=True,compute_uv=True,hermitian=True)

        eig_vec = U[:,:self.n_component]/np.sqrt(S[:self.n_component]).reshape(1,-1)
        self.p_pc_= np.sum(eig_vec.T[:,:,None,None] * r_ps[None,:,:,:],axis=1)
        self.p_std_ = np.sqrt(S[:self.n_component]/(X.shape[0]-1))
        self.p_score_ = X @ eig_vec
        
    def transform(self,ps): 
        if self.centered:
            r_ps = ps
        else:
            r_ps = ps - self.m_ps_[None,:,:]
        return self._mgm(r_ps,self.p_pc_)
    

class MomentaLDA(LinearDiscriminantAnalysis):

    def __init__(self, solver = "svd", shrinkage = None, priors = None, n_components = None, store_covariance = False, tol = 0.0001, covariance_estimator = None) -> None:
        super().__init__(solver, shrinkage, priors, n_components, store_covariance, tol, covariance_estimator)

    def _fit_initialization(self,Kv,ps,qbar,qbar_mask): 
        self.Kv_ = Kv
        self.qbar_ = qbar
        self.ps_ = ps
        self._kgm = MaskMomentaGramMatrix(self.Kv_,qbar,qbar_mask)
        self.G_ = self._kgm(self.ps_,self.ps_)
        #U,S,Vh = svd(self.G_,full_matrices=True,compute_uv=True,hermitian=True)
        #self.s_G_ = U @ (np.sqrt(S).reshape(-1,1) * Vh)
        #self.inv_s_G_ = U @ (1/np.sqrt(S).reshape(-1,1)* Vh)
        self.s_G_ = sqrtm(self.G_)
        self.inv_s_G_ = sqrtm(inv(self.G_))

    def fit(self,Kv,ps,qbar,qbar_mask,y):
        self._fit_initialization(Kv,ps,qbar,qbar_mask)
        super().fit(self.s_G_,y)
        self.p_coef_ = (self.coef_/np.linalg.norm(self.coef_,axis=1).reshape(-1,1)) @ self.inv_s_G_
        self.p_lda_ = np.sum(self.p_coef_[:,:,None,None] * self.ps_[None,:,:,:],axis=1)
        self.p_score_ = self.p_coef_ @ self.G_

    def _embedding(self, ps): 
        if ps.ndim ==2: 
            ps = ps[None,:,:]
        embeded =self._kgm(ps,self.ps_) @ self.inv_s_G_ 
        return embeded

    def predict(self,ps):
        embeded = self._embedding(ps)
        return super().predict(embeded)
    
    def predict_proba(self,ps): 
        embeded = self._embedding(ps)
        return super().predict_proba(embeded)

    def predict_log_proba(self, ps):
        embeded = self._embedding(ps)
        return super().predict_log_proba(embeded)
    
    def predict_p_score(self,ps): 
        return self._kgm(ps,self.p_lda_)
