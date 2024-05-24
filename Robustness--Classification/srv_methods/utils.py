import numpy as np
import pathlib

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