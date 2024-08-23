""" Utils

A set of utils function to run automatically an enetire inference cycle, plotting and saving results.

"""

import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_results_simulations(snp, snv, res, params, purity, coverage):
    """_summary_

    Parameters
    ----------
    snp : _type_
        _description_
    snv : _type_
        _description_
    res : _type_
        _description_
    params : _type_
        _description_
    purity : _type_
        _description_
    coverage : _type_
        _description_
    """
    
    sns.set_theme(style="white", font_scale=1)
    fig, axes = plt.subplots(4, 1, figsize=(8, 7))

    baf = sns.scatterplot(data=snp, x="pos", y="baf", s=2, ax=axes[0], hue="CN_1")
    dr = sns.scatterplot(data=snp, x="pos", y="dr", s=2, ax=axes[1], hue="CN_1", legend=False)
    vaf = sns.scatterplot(data=snv, x="pos", y="vaf", s=2, ax=axes[2], hue="CN_1", legend=False)

    cn = sns.scatterplot(data=res, x="pos", y="CN_Major", s=2, ax=axes[3], legend=False)
    cn = sns.scatterplot(data=res, x="pos", y="CN_minor", s=2, ax=axes[3], legend=False)

    axes[0].set_ylim(0,1) 
    sns.move_legend(
        baf, 
        "lower center",
        bbox_to_anchor=(.5, 1.2), ncol=4, title=None, frameon=True,
    )

    axes[0].set_title(f'True Purity = {purity}, Coverage = {coverage}')
    inf_purity = float(params['purity'])
    axes[3].set_title(f'Inferred Purity = {round(inf_purity, 2)}')
    fig.tight_layout()
    