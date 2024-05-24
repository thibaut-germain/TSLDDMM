import numpy as np
import torch
import torchcde
import pathlib

from diff_module.NCDE.controldiffeq import natural_cubic_spline_coeffs

def _pad(channel, max_len):
    channel = torch.tensor(channel)
    out = torch.full((max_len,), channel[-1])
    out[:channel.size(0)] = channel
    return out

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


def get_coeffs(X_missing,X_mask,X_delta,interpolate='natural', use_intensity=True):
    max_len = torch.tensor([len(x) for x in X_missing]).max()
    times = torch.linspace(0, 1, max_len)
    intensity = ~torch.isnan(X_missing)
    intensity = intensity.to(X_missing.dtype).cumsum(dim=1)

    values_T = torch.cat([times.repeat((X_missing.shape[0],1)).unsqueeze(-1), X_missing], dim=-1)
    values_TI = torch.cat([times.repeat((X_missing.shape[0],1)).unsqueeze(-1), intensity, X_missing], dim=-1) 

    if interpolate == 'natural':
        if use_intensity:
            coeffs = natural_cubic_spline_coeffs(times, values_TI) # uinsg controldiffeq/interpolation
            coeffs = torch.cat(coeffs, dim=-1)
        else:
            coeffs = natural_cubic_spline_coeffs(times, values_T) # uinsg controldiffeq/interpolation
            coeffs = torch.cat(coeffs, dim=-1)
    elif interpolate == 'hermite':
        if use_intensity:
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(values_TI, times) # using torchcde
        else:
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(values_T, times) # using torchcde
    
    return coeffs
