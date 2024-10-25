<h1 align="center">TS-LDDMM: Time series shape analysis</h1>

## Abstract

Analyzing inter-individual variability of physiological functions is particularly appealing in medical and biological contexts to describe or quantify health conditions. Such analysis can be done by comparing individuals to a reference one with time series as biomedical data.
This paper introduces an unsupervised representation learning (URL) algorithm for time series tailored to inter-individual studies. The idea is to represent time series as deformations of a reference time series. The deformations are diffeomorphisms parameterized and learned by our method called TS-LDDMM. Once the deformations and the reference time series are learned, the vector representations of individual time series are given by the parametrization of their corresponding deformation. At the crossroads between URL for time series and shape analysis, the proposed algorithm handles irregularly sampled multivariate time series of variable lengths and provides shape-based representations of temporal data.
In this work, we establish a representation theorem for the graph of a time series and derive its consequences on the LDDMM framework. We showcase the advantages of our representation compared to existing methods using synthetic data and real-world examples motivated by biomedical applications.


## Functionalities
### Mice breathing behaviors
The experiment is located in the mouse folder. Experiments must be ran in the following order: 
1. script_ts-lddmm_exp_1.py
2. script_ts-lddmm_exp_1_barycenter.py
3. script_ts-lddmm_exp_2.py
4. script_lddmm_exp_1.py
5. script_lddmm_exp_2.py

All parameters are set according to results shown in the paper. To run an experiment, use the following command from the mouse folder:
  ```(bash)
  python <exp_name>.py
  ```

Results can be explored from any of the follow jupyter notebooks: 
- result_ts-lddmm_exp_1.ipynb
- result_ts-lddmm_exp_2.ipynb
- result_lddmm_exp_1.ipynb

### Identifiability
The identifiability experiment is located in the identifiability folder. Experiments can be ran from the identifiability folder with the following the script: 
- wellspecified.py
- misspecified.py
Results can be explored from the result.ipynb jupyter notebook.

### Robustness
The robustness experiment is located in the Robustness--Classification folder.
For the lddmm-based methods run in order the following scripts from the lddmm_methods folder: 
1. script_no_missing_data.py 
2. script_missing_data.py

To run Neural ODEs methods run in order the following scripts from the url folder: 
1. param_search.py 
2. model_run.py

### Shape analysis lassification
The shap analysis classification experiment is located in the Robustness--Classification folder.
For the lddmm-based methods run the following script from the lddmm_methods folder: 
1. script_no_missing_data.py (note that you also need to run it for the robustness experiment)

For SRV-based method methods run the following script from the srv_methods folder: 
1. scipt_srv.py

## Prerequisites

1.  To run experiments with shape related methods (ts-lddmm,lddmm,tclr,shape-fpca) use the following environment: 

```(bash) 
conda create --name shape --file shape_requirements.txt
``` 

2.  To run experiments with neural ODEs methods use the following environment: 

```(bash) 
conda create --name n-ode --file n-ode_requirements.txt
``` 

