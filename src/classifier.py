import numpy as np
import optax

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
from src.utils import MaskMomentaGramMatrix,batch_dataset,unbatch_dataset
from src.lddmm import batch_one_to_many_registration
from src.barycenter import batch_barycenter_registration


class MomentaSVC(SVC):

    def __init__(self,Kv,dataloss,barycenter_initializer = None,registration_initializer=None,C=1,max_per_batch =10,gamma_loss=0.0,niter=200,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0, shrinking = True, probability = False, tol = 0.001, cache_size = 200, class_weight = None, verbose = True, max_iter =-1, decision_function_shape = "ovr", break_ties = False, random_state = None) -> None:
        super().__init__(C=C, kernel="precomputed",shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=False, max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)
        
        self.Kv = Kv
        self.dataloss = dataloss
        self.barycenter_initializer = barycenter_initializer
        self.registration_initializer = registration_initializer
        self.max_per_batch = max_per_batch
        self.gamma_loss = gamma_loss
        self.niter = niter
        self.optimizer = optimizer
        self.nt = nt
        self.deltat = deltat
        self.verbose = verbose

    def _batch(self,X,X_mask): 
        n_samples = X.shape[0]
        for i in np.arange(self.max_per_batch,0,-1): 
            if n_samples % i ==0: 
                self.batch_size_ = i
                batched_X, batched_X_mask = batch_dataset(X,self.batch_size_,X_mask)
                break
        return batched_X, batched_X_mask
    
    def _unbached(self,batched_X, batched_X_mask=None): 
        return unbatch_dataset(batched_X,batched_X_mask)
    
    def fit(self,X,X_mask,y,init=None,init_mask=None): 
        batched_X,batched_X_mask = self._batch(X,X_mask)
        batched_ps,self.qbar_,self.qbar_mask_ = batch_barycenter_registration(batched_X,batched_X_mask,self.Kv,self.dataloss,init,init_mask,self.barycenter_initializer,self.niter,self.optimizer,self.gamma_loss,self.nt,self.deltat,self.verbose)
        self.ps_ = self._unbached(batched_ps)
        self._mgm = MaskMomentaGramMatrix(self.Kv,self.qbar_,self.qbar_mask_)
        self.X_svc_ = self._mgm(self.ps_,self.ps_)
        self.y_ = y
        super().fit(self.X_svc_,self.y_)
        return self
    
    def set_C(self,X,X_mask,y,C_lst):
        batched_X,batched_X_mask = self._batch(X,X_mask)
        batched_ps,_,_ = batch_one_to_many_registration(self.qbar_,self.qbar_mask_,batched_X,batched_X_mask,self.Kv,self.dataloss,self.registration_initializer,self.gamma_loss,self.niter,self.optimizer,self.nt,self.deltat,self.verbose)
        ps = self._unbached(batched_ps)
        X_svc = self._mgm(ps,self.ps_)
        best_C = 0
        best_score = -np.inf
        for C in C_lst: 
            self.C = C
            super().fit(self.X_svc_,self.y_)
            y_pred = super().predict(X_svc)
            score = f1_score(y,y_pred,average="macro")
            print(f"Validation -- C: {C} -- f1score: {score}")
            if score > best_score: 
                best_C = C
                best_score = score.copy()
        self.C = best_C
        print(f"Best C :{self.C}")
        return self
    
    def predict(self,X,X_mask):
        batched_X,batched_X_mask = self._batch(X,X_mask)
        batched_ps,_,_ = batch_one_to_many_registration(self.qbar_,self.qbar_mask_,batched_X,batched_X_mask,self.Kv,self.dataloss,self.registration_initializer,self.gamma_loss,self.niter,self.optimizer,self.nt,self.deltat,self.verbose)
        ps = self._unbached(batched_ps)
        X_svc = self._mgm(ps,self.ps_)
        return super().predict(X_svc)
    
    def score(self,X,X_mask,y):
        y_pred = self.predict(X,X_mask)
        return accuracy_score(y,y_pred)
    



from sklearn.svm import SVC,OneClassSVM
from sklearn.metrics import accuracy_score
from src.utils import MaskMomentaGramMatrix,batch_dataset,unbatch_dataset
from src.lddmm import batch_one_to_many_registration
from src.barycenter import batch_barycenter_registration

class OneMomentaSVC(OneClassSVM):
    
    def __init__(self,Kv,dataloss,max_per_batch =10,gamma_loss=0.0,niter=200,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,degree = 3, gamma = "scale", coef0 = 0, tol = 0.001, nu = 0.5, shrinking =True, cache_size = 200, verbose=False, max_iter=-1) -> None:
        super().__init__(kernel="precomputed", degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu, shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)

        self.Kv = Kv
        self.dataloss = dataloss
        self.max_per_batch = max_per_batch
        self.gamma_loss = gamma_loss
        self.niter = niter
        self.optimizer = optimizer
        self.nt = nt
        self.deltat = deltat
        self.verbose = verbose

    def _batch(self,X,X_mask): 
        n_samples = X.shape[0]
        for i in np.arange(self.max_per_batch,0,-1): 
            if n_samples % i ==0: 
                self.batch_size_ = i
                batched_X, batched_X_mask = batch_dataset(X,self.batch_size_,X_mask)
                break
        return batched_X, batched_X_mask
    
    def _unbached(self,batched_X, batched_X_mask=None): 
        return unbatch_dataset(batched_X,batched_X_mask)
    
    def fit(self,X,X_mask,y,init=None,init_mask=None): 
        idxs = y==1
        X = X[idxs]
        X_mask = X_mask[idxs]
        batched_X,batched_X_mask = self._batch(X,X_mask)
        batched_ps,self.qbar_,self.qbar_mask_ = batch_barycenter_registration(batched_X,batched_X_mask,self.Kv,self.dataloss,init,init_mask,self.niter,self.optimizer,self.gamma_loss,self.nt,self.deltat,self.verbose)
        self.ps_ = self._unbached(batched_ps)
        self._mgm = MaskMomentaGramMatrix(self.Kv,self.qbar_,self.qbar_mask_)
        X_svc = self._mgm(self.ps_,self.ps_)
        super().fit(X_svc)
        return self
    
    def predict(self,X,X_mask):
        batched_X,batched_X_mask = self._batch(X,X_mask)
        batched_ps,_,_ = batch_one_to_many_registration(self.qbar_,self.qbar_mask_,batched_X,batched_X_mask,self.Kv,self.dataloss,self.gamma_loss,self.niter,self.optimizer,self.nt,self.deltat,self.verbose)
        ps = self._unbached(batched_ps)
        X_svc = self._mgm(ps,self.ps_)
        return np.clip(super().predict(X_svc),0,1)
    
    def score(self,X,X_mask,y):
        y_pred = self.predict(X,X_mask)
        return accuracy_score(y,y_pred)