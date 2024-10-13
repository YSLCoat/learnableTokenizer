"""_summary_
Written by Marius Aasan
"""

import warnings
import torch.optim.lr_scheduler as lrsched
import numpy as np

from typing import Optional
from torch.optim import Optimizer


class CosineDecay(lrsched._LRScheduler):

    '''Cosine LR decay with sinusoidal warmup.
    '''
    
    def __init__(
        self, optimizer:Optimizer, lr_start:float, lr_stop:float, epochs:int, 
        warmup_ratio:float, batch_size:Optional[int]=None, 
        n_samples:Optional[int]=None, last_epoch:int=-1, verbose:bool=False
    ):
        '''Initializes scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer
            lr_start (float): Start learning rate.
            lr_stop (float): Stop learning rate.
            epochs (int): Length of decay schedule in epochs.
            warmup_ratio (float): Ratio of epochs to be used for warmup.
            batch_size (int): Number of samples per batch/step.
            n_samples (int): Total number of samples per epoch.
            last_epoch (int): Last epoch for continuation, standard from PyTorch. Default: -1
            verbose (bool): If True, prints a message to stdout for each update. Default: False.        
        '''
        assert 0 <= lr_start
        assert 0 <= lr_stop
        self.lr_start = lr_start
        self.lr_stop = lr_stop
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.optimizer = optimizer

        # For batchwise steps
        if n_samples is not None and batch_size is not None:
            self._epochsteps = -(-n_samples//batch_size)

        # For epochwise steps
        else:
            self._epochsteps = 1

        super().__init__(optimizer, last_epoch, verbose=verbose)
        
        
    def get_lr(self):
        if not self._get_lr_called_within_step: # type: ignore
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )
        
        t = self.last_epoch / self._epochsteps
        return [self.F(t, lr_max) for lr_max in self.base_lrs]                
        
            
    def v(self, x):
        return (np.cos(np.pi*np.clip(x,0,1)) + 1) / 2
        
    def u(self, x):
        d = self.warmup_ratio
        return (np.cos(np.pi*(np.clip(x,0,d)/d-1)) + 1) / 2
    
    def W(self, x, lr_max):
        diff = (lr_max - self.lr_start)
        return diff*self.u(x) + self.lr_start
    
    def D(self, x, lr_max):
        d = self.warmup_ratio
        diff = (1 - self.lr_stop / lr_max)
        return diff*self.v(np.maximum(x-d,0)/(1-d)) + self.lr_stop/lr_max
        
    def F(self, x, lr_max):
        T = self.epochs
        x = x/T
        return self.W(x, lr_max)*self.D(x, lr_max)