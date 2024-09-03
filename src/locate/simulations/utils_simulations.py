import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import torch
import os

def create_data_input(snp, snv):
    """_summary_

    Parameters
    ----------
    snp : _type_
        _description_
    snv : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    snv = snv.loc[snv['coverage'] > snv['nv']] 
    take_idx = snv.index
    snp = snp[snp.index.isin(take_idx)]


    data_input = {'baf':torch.tensor(np.array(snp['baf']).reshape(-1, 1)), 
                'dr':torch.tensor(np.array(snp['dr']).reshape(-1, 1)),
                'vaf':torch.tensor(np.array(snv['nv']).reshape(-1, 1)),
                'dp':torch.tensor(np.array(snv['coverage']).reshape(-1, 1)),
                'dp_snp':torch.tensor(np.array(snp['cov']).reshape(-1, 1))}
    return data_input, snp, snv


def plot_simulations(snp, snv, purity, coverage, path='', save = False):
    """_summary_

    Parameters
    ----------
    snp : _type_
        _description_
    snv : _type_
        _description_
    purity : _type_
        _description_
    coverage : _type_
        _description_
    path : str, optional
        _description_, by default ''
    save : bool, optional
        _description_, by default False
    """
    
    sns.set_theme(style="white", font_scale=1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7))

    baf = sns.scatterplot(data=snp, x="pos", y="baf", s=2, ax=axes[0], hue="CN_1")
    dr = sns.scatterplot(data=snp, x="pos", y="dr", s=2, ax=axes[1], hue="CN_1", legend=False)
    vaf = sns.scatterplot(data=snv, x="pos", y="vaf", s=2, ax=axes[2], hue="CN_1", legend=False)

    axes[0].set_ylim(0,1) 
    sns.move_legend(
        baf, 
        "lower center",
        bbox_to_anchor=(.5, 1.2), ncol=4, title=None, frameon=True,
        prop={'size': 12},
        markerscale=5 
    )

    axes[0].set_title(f'Purity = {purity}, Coverage = {coverage}')
    fig.tight_layout()
    
def save_data(SNP, SNV, path, name):
    """_summary_

    Parameters
    ----------
    SNP : _type_
        _description_
    SNV : _type_
        _description_
    path : _type_
        _description_
    name : _type_
        _description_
    """
    
    data_input = {}
    data_input['baf'] = torch.tensor(np.expand_dims(SNP.baf,1)).float()
    data_input['dr'] = torch.tensor(np.expand_dims(SNP.dr,1)).float()
    data_input['vaf'] = torch.tensor(np.expand_dims(SNV.vaf,1)).float()
    
    f = open(os.path.join(path,name+'.pkl'),"wb")
    pickle.dump(data_input, f)
    f.close()
    return
    