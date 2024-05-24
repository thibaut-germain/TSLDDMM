import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from joblib import Parallel, delayed

def dataset_to_normalized_srvf(dataset,time,smooth=False): 
    # dataset : (n_ts, n_samples)
    lst = []
    for f in dataset: 
        q = uf.f_to_srsf(f,time,smooth)
        nq = q/np.sqrt(uf.innerprod_q(time,q,q))
        lst.append(nq)
    return np.array(lst)

def shoot_exp(time,qs,v): 
    norm = np.sqrt(uf.innerprod_q(time,v,v))
    if norm>0:
        return np.cos(norm)*qs + np.sin(norm)*(v/norm)
    else: 
        return qs

def inv_shoot_exp(time,qs,qt): 
    inn = uf.innerprod_q(time,qs,qt)
    if (inn<1.0)*(inn>-1.0):
        theta = np.arccos(inn)
        return theta/np.sin(theta)*(qt - inn *qs)
    else: 
        return np.zeros_like(qs,dtype=float)
    
def compute_intermediate(qbar,time,qt):
    gam = uf.optimum_reparam(qbar,time,qt)
    qt_aligned = uf.warp_q_gamma(time,qt,gam)
    qt_aligned = qt_aligned/np.sqrt(uf.innerprod_q(time,qt_aligned,qt_aligned))
    vt = inv_shoot_exp(time,qbar,qt_aligned)
    return gam,qt_aligned,vt

def compute_karcher_mean(time,qs,init=None,niter=20,lr=0.1,random_seed=None,verbose = False,n_jobs=1):
    #qs shape (m_ts, n_samples)
    if not random_seed is None: 
        np.random.seed(random_seed)
    if init is None: 
        idx = np.random.choice(qs.shape[0])
        init = qs[idx]
    qbar = init.copy()
    for i in range(niter): 
        results = Parallel(n_jobs=n_jobs)(delayed(compute_intermediate)(qbar,time,qt) for qt in qs)
        gams,qs_aligned,vs_aligned = list(zip(*results))
        vbar = np.mean(vs_aligned,axis=0)
        qbar = shoot_exp(time,qbar,lr*vbar)
        qbar = qbar/np.sqrt(uf.innerprod_q(time,qbar,qbar))
        if verbose: 
            error = np.sqrt(uf.innerprod_q(time,vbar,vbar))
            print(f"Iteration {i+1}/{niter} -- error : {np.around(error,3)}")
    return qbar,np.array(gams),np.array(qs_aligned),np.array(vs_aligned)

def compute_karcher_covariance(vs_alinged): 
    n_ts = vs_alinged.shape[0]
    return (vs_alinged.T@vs_alinged)/(n_ts-1)

def compute_aligned_shoot_vector(qs,time,qt):
    gam = uf.optimum_reparam(qs,time,qt)
    qt_aligned = uf.warp_q_gamma(time,qt,gam)
    qt_aligned = qt_aligned/np.sqrt(uf.innerprod_q(time,qt_aligned,qt_aligned))
    vt = inv_shoot_exp(time,qs,qt_aligned)
    return vt


class TangentCurveLogisticRegression(LogisticRegression): 

    def __init__(self,penalty=None,n_components=None,tol=0.0001,C=1.0,smooth=False,karcher_niter = 20, karcher_lr = 0.1,verbose=0,n_jobs=1): 
        super().__init__(penalty=penalty,tol=tol,C=C,verbose=verbose)
        self.n_components = n_components
        self.smooth = smooth
        self.karcher_niter = karcher_niter
        self.karcher_lr = karcher_lr
        self.n_jobs = n_jobs

    def fit(self,X,y): 
        # X : time series dataset : n_ts, n_samples
        # y : label : n_ts
        self.time_ = np.arange(X.shape[1],dtype=float)
        self.qs_ = dataset_to_normalized_srvf(X,self.time_,self.smooth)
        self.qbar_,self.gams_,self.qs_aligned_,self.vs_aligned_ = compute_karcher_mean(self.time_,self.qs_,niter=self.karcher_niter,lr=self.karcher_lr,n_jobs=self.n_jobs)
        self.pca_ = PCA(self.n_components)
        self.vs_pca_ = self.pca_.fit_transform(self.vs_aligned_)

        super().fit(self.vs_pca_,y)
        return self
    
    def predict(self,X): 
        tqs = dataset_to_normalized_srvf(X,self.time_,self.smooth)
        vs_aligned = Parallel(self.n_jobs)(delayed(compute_aligned_shoot_vector)(self.qbar_,self.time_,qt)for qt in tqs)
        vs_aligned = np.array(vs_aligned)
        vs_pca = self.pca_.transform(vs_aligned)
        return super().predict(vs_pca)


class FpcaSVM(SVC): 

    def __init__(self,C=1.,kernel = 'rbf',n_components = None,parallel=False,njobs=1): 
        super().__init__(C=C,kernel=kernel)
        self.njobs = njobs
        self.parallel = parallel
        self.n_components = n_components


    def fit(self,X,y): 
        self.n_ts_, self.n_samples_ = X.shape
        self.time_ = np.arange(self.n_samples_,dtype=float)
        obj = fs.fdawarp(X.T,self.time_)
        obj.srsf_align(parallel=self.parallel,cores=self.njobs,MaxItr=50,verbose=False)
        self.jpca_ = fs.fdajpca(obj)
        if self.n_components is None: 
            self.n_components = min(self.n_ts_,self.n_samples_)
        else: 
            self.n_components = min(self.n_ts_,self.n_samples_,self.n_components)
        self.jpca_.calc_fpca(no=self.n_components,parallel=self.parallel,cores=self.njobs)
        super().fit(self.jpca_.coef,y)
        return self
    
    def predict(self, X):
        self.jpca_.project(X.T)
        return super().predict(self.jpca_.new_coef)
