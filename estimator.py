## Causal Effect Estimator
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

def ITE(Y_sim,Y_ctf,T):
    """ Calculate ITE ground truth.
    """
    t0_indices = (T == 0).nonzero().squeeze()
    t1_indices = (T == 1).nonzero().squeeze()

    y_sim_t0 =Y_sim.clone().scatter(0,t1_indices,0)
    y_ctf_t0 =Y_ctf.clone().scatter(0,t0_indices,0) # choose t=1's ctf outcome(mask t0 indices as 0)

    y_sim_t1 =Y_sim.clone().scatter(0,t0_indices,0)
    y_ctf_t1 =Y_ctf.clone().scatter(0,t1_indices,0) # choose t=0's ctf outcome(mask t1 indices as 0)

    y_t0 = y_sim_t0 + y_ctf_t0
    y_t1 = y_sim_t1 + y_ctf_t1

    return y_t1 - y_t0


