import sys 
sys.path.insert(0,"../")

import numpy as np 
import jax.numpy as jnp
import json 
import optax
from optax.schedules import warmup_cosine_decay_schedule



from src.utils import batch_dataset
from src.kernel import VFTSGaussKernel, TSGaussGaussKernel,GaussKernel
from src.barycenter import batch_varifold_barycenter_registration, batch_barycenter_registration
from src.loss import SumVarifoldLoss


BATCH_SIZE = 75
NITER = 400


scheduler = warmup_cosine_decay_schedule(0,0.3,30,400,0.)
optimizer = optax.adam(scheduler)



Kv = VFTSGaussKernel(1,0.1,40,1,1)
Kl1 = TSGaussGaussKernel(2,2,2,2)
Kl2 = TSGaussGaussKernel(1,1,1,1)
Kl3 = TSGaussGaussKernel(1,0.1,1,0.1)
Kls=[Kl1,Kl2,Kl3]
dataloss = SumVarifoldLoss(Kls)





SAVE_PATH = "./results/SS_2.json"


if __name__ == "__main__":

    # Import data 
    X = np.load("./dataset/X.npy")
    X_mask = np.load("./dataset/X_mask.npy")
    batched_dataset, batched_masks = batch_dataset(X,BATCH_SIZE,X_mask)

    print("Running experiment")
    ps ,qb, qb_mask= batch_barycenter_registration(batched_dataset,batched_masks,Kv,dataloss,niter=NITER,optimizer=optimizer,gamma_loss=0.00)
    
    print("Exporting results")
    dct =dict(
        ps = json.dumps(ps.tolist()),
        qb = json.dumps(qb.tolist()),
        qb_mask = json.dumps(qb_mask.tolist())
    )
    
    with open(SAVE_PATH, "w") as f: 
        f.write(json.dumps(dct))

    print("Done")






