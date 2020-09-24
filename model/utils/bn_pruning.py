import torch
import numpy as np

class BNOptimizer():
    @staticmethod
    def updataBN(prune_flag,model,decay_factor,prune_idx):
        '''
        @brif:
        prune_flag:  True or False ; weather sparsity the scale factor in BN_layer
        model:       model which in training
        decay_factor:the factor for speeding up sparsiting
        prune_idx:    
        '''
        if prune_flag:
            for idx in prune_idx:
                raise "too lazy" 


