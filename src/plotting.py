import numpy as np
import matplotlib.pyplot as plt


def plot2Dfigure(s_sig,t_sig,p0,shoot,flow,mask_s_sig=None,mask_t_sig=None,nv=50,nh=50): 
    if mask_s_sig is None:
        s_sig = (s_sig,)
        plot_s_sig = s_sig 
    else: 
        plot_s_sig = s_sig[np.where(mask_s_sig==True)[0],:]
        s_sig = (s_sig,mask_s_sig)
  
    if mask_t_sig is None: 
        t_sig = (t_sig,)
        plot_t_sig = t_sig
    else: 
        plot_t_sig = t_sig[np.where(mask_t_sig==True)[0],:]
        t_sig = (t_sig,mask_t_sig)
        

    _,p_sig = shoot(p0,*s_sig)
    if mask_s_sig is None: 
        plot_p_sig = p_sig
    else: 
        plot_p_sig = p_sig[np.where(mask_s_sig==True)[0],:]

    fig,axs = plt.subplots(2,1,sharex=True,sharey=True, figsize = (10,5))
    axs[0].plot(*plot_s_sig.T, label = "from")
    axs[0].legend()
    axs[1].plot(*plot_t_sig.T, color = "green", label = "to")
    axs[1].plot(*plot_p_sig.T, color = "r", label = "warping")
    axs[1].legend()
    sz = 0.2 
    a = min(np.min(plot_s_sig[:,0]),np.min(plot_t_sig[:,0]),np.min(plot_p_sig[:,0]))
    b = max(np.max(plot_s_sig[:,0]),np.max(plot_t_sig[:,0]),np.max(plot_p_sig[:,0]))
    c = min(np.min(plot_s_sig[:,1]),np.min(plot_t_sig[:,1]),np.min(plot_p_sig[:,1]))
    d = max(np.max(plot_s_sig[:,1]),np.max(plot_t_sig[:,1]),np.max(plot_p_sig[:,1]))
    lsp1 = np.linspace(a-sz*(b-a),b+sz*(b-a),nv,dtype=np.float32)
    lsp2 = np.linspace(c-sz*(d-c),d+sz*(d-c),nh,dtype=np.float32)
    X1, X2 = np.meshgrid(lsp1,lsp2)
    axs[0].plot(X1,X2,'k',linewidth=.25)
    axs[0].plot(X1.T,X2.T,'k',linewidth=.25)
    x = np.concatenate((X1.reshape(nv*nh,1),X2.reshape(nv*nh,1)),axis=1)
    phix = flow(x,p0,*s_sig)[0]
    X1 = phix[:,0].reshape(nh,nv)
    X2 = phix[:,1].reshape(nh,nv)
    axs[1].plot(X1,X2,'k',linewidth=.25)
    axs[1].plot(X1.T,X2.T,'k',linewidth=.25)
    return fig,axs