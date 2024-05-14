import ruptures as rpt
from scipy import signal
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from itertools import groupby 

class PhaseSeg(BaseEstimator,TransformerMixin):

    def __init__(self, sampfreq:int, prominence:float, wlen:float, cycle_minimum_duration:float,cycle_maximum_duration:float,trainig_size_per_interval=None,interval=None) -> None:
        """Initialization 

        Args: 
            sampfreq (int): Sampling frequency 
            prominence (float): Peak prominence for volume, in ml.
            wlen (float): Peak prominence search window, in seconds. 
            cycle_minimum_duration (float): minimum cycle duration, in seconds. 
            cycle_maximum_duration (float): maximum cycle duration, in seconds.  
            trainig_size_per_interval (int): number of cycles slected randomly on each interval. 
            interval (float): interval duration, in seconds. 
        """
        super().__init__()
        self.sampfreq = sampfreq
        self.prominence = prominence
        self.wlen = wlen 
        self.cycle_minimum_duration = cycle_minimum_duration
        self.cycle_maximum_duration = cycle_maximum_duration
        self.trainig_size_per_interval = trainig_size_per_interval
        self.interval = interval

    def fit(self,X): 
        """Compute detrended volume

        Args:
            X (np.ndarray): original sequence, shape (N,).

        """
        self.flow_ = X
        self._event_detection()
        self._filtration()
        if (self.trainig_size_per_interval is not None) & (self.interval is not None): 
            self._training_selection()

    def _volume(self,X:np.ndarray)->np.ndarray:
        """Compute detrended volume

        Args:
            X (np.ndarray): original sequence, shape (N,).

        Returns:
            np.ndarray: Detrended volume, shape(N,)
        """
        arr = signal.detrend(np.cumsum(X*1/self.sampfreq))
        arr = arr.astype(float)
        return arr

    def _event_detection(self)->None:
        """
        Detect inhalation and exhalation start_time
        """
        #removing unwanted time
        self.flow_ = self.flow_.astype(float)

        #getting volume
        self.volume_ = self._volume(self.flow_)

        #Getting start inspiration
        self.insp_start_ = signal.find_peaks(-self.volume_,prominence=self.prominence,wlen=int(self.wlen*self.sampfreq))[0]

        #Getting start expiration
        exp_start = list()
        for (start, end) in rpt.utils.pairwise(self.insp_start_):
            exp_start.append(np.argmax(self.volume_[start:end]) + start)
        self.exp_start_= np.array(exp_start).astype(int)

        return self

    def _filtration(self): 
        #initialization
        duration = (self.insp_start_[1:]-self.insp_start_[:-1])/self.sampfreq
        self.valid_mask_ = np.ones_like(duration).astype(bool)

        #minimum duration
        mask_minimum = duration<self.cycle_minimum_duration
        self.valid_mask_[mask_minimum] = 0

        #maximum duration 
        mask_maximum = duration>self.cycle_maximum_duration
        self.valid_mask_[mask_maximum] = 0

        #too short inhlation sequences
        inhalation_mask = (self.exp_start_-self.insp_start_[:-1])<3
        self.valid_mask_[inhalation_mask] = 0

        #too short exhalation sequences
        exhalation_mask = (self.insp_start_[1:]-self.exp_start_)<3
        self.valid_mask_[exhalation_mask] = 0

        return self

    def _training_selection(self): 
        #partition in intervals
        valid_cycle_start = self.insp_start_[:-1][self.valid_mask_]
        interval_size = int(self.sampfreq*self.interval)
        lsts = []
        idx = np.arange(len(self.valid_mask_))[self.valid_mask_]
        arr = np.c_[idx,valid_cycle_start]
        for i,group in groupby(arr, lambda x : x[1]//interval_size ):
            g_arr = np.array(list(group))[:,0]
            lsts.append(g_arr)

        #randomly select sequences
        training_selection = []
        for lst in lsts: 
            if len(lst)<self.trainig_size_per_interval: 
                training_selection.append(lst)
            else:
                training_selection.append(np.sort(np.random.choice(lst,self.trainig_size_per_interval,replace=False)))
        #create the mask
        training_selection = np.concatenate(training_selection).astype(int)
        self.training_selection_ = np.zeros_like(self.valid_mask_).astype(bool)
        self.training_selection_[training_selection] = True

        return self

    def get_inhalation_index(self):
        arr = np.vstack((self.insp_start_[:-1],self.exp_start_)).T
        return arr.astype(int)

    def get_exhalation_index(self):
        arr = np.vstack((self.exp_start_,self.insp_start_[1:])).T
        return arr.astype(int)

    def get_cycle_index(self): 
        arr = np.vstack((self.insp_start_[:-1],self.insp_start_[1:])).T
        return arr.astype(int)
    
    def get_inhalation(self):
        lst = []
        for start,end in self.get_inhalation_index():
            lst.append(self.flow_[start:end])
        return np.array(lst,dtype=object)

    def get_exhalation(self):
        lst = []
        for start,end in self.get_exhalation_index(): 
            lst.append(self.flow_[start:end])
        return np.array(lst,dtype=object)
    
    def get_cycle(self): 
        lst = []
        for start,end in self.get_cycle_index(): 
            lst.append(self.flow_[start:end])
        return np.array(lst,dtype=object)

    def get_sequences(self,kind,mask=None): 
        if mask is not None: 
            if kind == 'inhalation':
                return self.get_inhalation()[mask]
            elif kind == 'exhalation': 
                return self.get_exhalation()[mask]
            elif kind == "cycle":
                return self.get_cycle()[mask]
            else: 
                raise ValueError('Invalid parameter')
        else: 
            if kind == 'inhalation':
                return self.get_inhalation()
            elif kind == 'exhalation': 
                return self.get_exhalation()
            elif kind == "cycle":
                return self.get_cycle()
            else: 
                raise ValueError('Invalid parameter')
            
    def get_cycle_data(self,data,mask=None): 
        data = data.astype(float)
        lst = []
        for start,end in self.get_cycle_index(): 
            lst.append(data[:,start:end].T)
        return np.array(lst,dtype=object)[mask]
        