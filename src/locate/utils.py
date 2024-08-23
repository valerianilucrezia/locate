""" Utils

A set of utils function to run automatically an enetire inference cycle, plotting and saving results.

"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pyro.optim import ClippedAdam
from pyro.infer import SVI, TraceGraph_ELBO
import torch
from pandas.core.common import flatten
import pyro

def plot_loss(loss, save = False, output = "run1"):
    """Plot loss function.

    Parameters
    ----------
    loss : np.array
        The loss.
    save : bool, optional
        Boolean specifyng if the plot as to be saves.
    output : string, optional
        Where to save the plot.

    Returns
    -------
    matplotpib
        Plot.

    Examples
    --------
    >>> plot_loss(loss)
    """
    
    plt.plot(loss)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    if(save):
        plt.savefig(output + "_ELBO.png")

def collect_params(pars):
    """_summary_

    Parameters
    ----------
    pars : _type_
        _description_
    """
    
    pars = list(flatten([value.detach().tolist() for key, value in pars.items()]))
    return(np.array(pars))

def retrieve_params():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    
    param_names = pyro.get_param_store()
    res = {nms: pyro.param(nms) for nms in param_names}
    return res



from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def filter_tail_vaf(vaf):
    """_summary_

    Parameters
    ----------
    vaf : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    x = (vaf/dp).squeeze(1)
    
    # kernel adjust the data
    kernel_adjust = 1.0  
    bandwidth = 'scott'  
    kde = gaussian_kde(x, bw_method=bandwidth)
    
    peaks, heights = find_peaks(kde, height=0)
    
    return vaf
    