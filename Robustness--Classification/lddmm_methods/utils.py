import numpy as np
import pathlib
import jax.numpy as jnp

valid_datasets = [
    "ArrowHead",
    "BME",
    "ECG200",
    "FacesUCR",
    "GunPoint",
    "PhalangesOutlinesCorrect",
    "Trace",
    "Cricket",
    "ERing",
    "Handwriting",
    "Libras",
    "NATOPS",
    "RacketSports",
    "UWaveGestureLibrary",
    "ArticularyWordRecognition"
]

here = pathlib.Path(__file__).resolve().parent.parent.parent

def get_data(dataname,missing_rate=0.0):
    '''
    get data as tensor
    '''
    assert dataname in valid_datasets, "Must specify a valid dataset name."
    base_filename = here /"datasets"/dataname
    train_idx = np.load(base_filename/"train_idx.npy")
    val_idx = np.load(base_filename/"val_idx.npy")
    test_idx = np.load(base_filename/"test_idx.npy")
    y = np.load(base_filename/"y.npy")
    base_filename = here /"datasets"/dataname/f"0.{np.round(missing_rate*10,0).astype(int)}"
    X = np.load(base_filename/"X.npy")
    X_mask = np.load(base_filename/"X_mask.npy")
    X_delta = np.load(base_filename/"X_delta.npy")    
    return X, X_mask,X_delta,y ,train_idx,val_idx,test_idx


def time_shape_mask_embedding(x,x_mask,max_length,sampfreq=1,dtype = jnp.float32):
    n_s = x.shape[0]
    time = np.arange(n_s,dtype=float).reshape(-1,1)/sampfreq
    t_x = np.hstack((time,x),dtype=dtype)
    t_x = t_x[x_mask.astype(bool)]
    n_s = t_x.shape[0]
    t_x = np.pad(t_x,((0,max_length-n_s),(0,0))).astype(dtype)
    mask = np.full_like(t_x[:,:1],True,dtype=bool)
    mask[n_s:,:] = False
    return t_x, mask

def from_mask_timeseries_to_dataset(X:list,X_mask:list,sampfreq=1,dtype=jnp.float32): 
    lengths = np.array([np.sum(x_mask) for x_mask in X_mask],dtype=int)
    max_length = np.max(lengths)
    mask_lst = []
    ts_lst = []
    for ts,ts_mask in zip(X,X_mask): 
        t_ts, t_mask = time_shape_mask_embedding(ts,ts_mask,max_length,sampfreq,dtype=dtype)
        ts_lst.append(t_ts)
        mask_lst.append(t_mask)
    return jnp.array(ts_lst,dtype=dtype), jnp.array(mask_lst,dtype=jnp.bool_) 