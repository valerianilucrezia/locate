#!/usr/bin/env python

import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import TraceMeanField_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, Trace_ELBO

import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import re

import argparse
import os

import locate.locate as l
from locate.models import Clonal
from locate.utils import estimate_ploidy
from locate.evaluation.metric import score_cn_predictions


def read_data(in_file):
    data = pd.read_csv(in_file, sep = ',', on_bad_lines='skip') 
    data['pos'] = range(1, len(data) + 1)
    data['baf'] = data['median_baf'].apply(lambda x: 1 - x if x > 0.5 else x)
    data = data[data['baf']>0]
    data_input = {'baf':torch.tensor(np.array(data.baf).reshape(-1, 1)),
                'dr':torch.tensor(np.array(data.median_dr).reshape(-1, 1)),
                'dp_snp':torch.tensor(np.array(data.mean_dp).reshape(-1, 1)),
                'orginal_baf':torch.tensor(np.array(data.median_baf).reshape(-1, 1)),
                'vaf':torch.tensor(np.array(data.vaf).reshape(-1, 1)),
                'dp_snv':torch.tensor(np.array(data.DP).reshape(-1, 1))}
    
    return data, data_input

def run_hmm(data_input, 
            gpu = False, 
            nsteps = 100, 
            fix_ploidy = True, 
            vaf = False, 
            ploidy = None,
            bps = None):
    
    locate = l.LOCATE(CUDA = gpu)

    locate.set_model(Clonal)
    locate.set_optimizer(ClippedAdam)
    locate.set_loss(TraceEnum_ELBO)
    
    if vaf:
        locate.initialize_model({"baf":data_input["baf"],
                                "dr":data_input["dr"], 
                                "dp_snp":data_input["dp_snp"], 
                                "vaf": data_input["vaf"], 
                                "dp": data_input["dp"]})
    else:
        locate.initialize_model({"baf":data_input["baf"],
                                "dr":data_input["dr"], 
                                "dp_snp":data_input["dp_snp"], 
                                "vaf": None, 
                                "dp": None})

    ploidy = [ploidy if ploidy is not None else 0]
    locate.set_model_params({"jumping_prob" : 1e-6,
                            "fix_purity": False,
                            "fix_ploidy" : fix_ploidy, 
                            "prior_purity": 0,
                            "prior_ploidy": ploidy,
                            "scaling_factors": [1,1,1],
                            'hidden_dim': 4,
                            "prior_bp": bps,
                            "lambda_cn": 1.0,          # ΔCN penalty slope used to shape the prior
                            "alpha_conc": 50.0,        # global Dirichlet concentration (higher = stronger prior)
                            "alpha_self_boost": 3.0,   # extra mass on the diagonal
                            "alpha_dip_boost": 0.5,    # extra mass on diploid column (if present)
                            "bp_strength": 3.0})
    
    ll = locate.run(steps = nsteps, param_optimizer = {"lr" : 0.05}, guide_kind="normal")
    params = locate.learned_parameters_Clonal()
    return params



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, help="base", default = "")
    parser.add_argument("-s", "--sim", type=str, help="sample", default = "sim_21")
    
    parser.add_argument("-p", "--ploidy", type=bool, help="ploidy", default = True)
    parser.add_argument("-v", "--vaf", type=bool, help="vaf", default = False)  
    parser.add_argument("-b", "--bps", type=bool, help="breakpoint", default = False)  

    args = parser.parse_args()
    
    sim_dir = args.base + '/' + args.sim
    combinations = [ i for i in os.listdir(sim_dir) if os.path.isdir(sim_dir + '/' + i)]
    
    for comb in combinations:
        data,data_input = read_data(f'{sim_dir}/{comb}/mirr_smooth_snv.csv')
        ploidy, _ = estimate_ploidy(data, return_details=True)
        
        fix_ploidy = args.ploidy
        vaf = args.vaf
        bps = args.bps
        
        if not bps:
            prior_bps = None
        params = run_hmm(data_input, fix_ploidy=fix_ploidy, vaf=vaf, bps = prior_bps)
        
        name = f'vaf_{vaf}_ploidy_{ploidy}_bps_{bps}'
        out_dir = f'{sim_dir}/{comb}/{name}'
        os.makedirs(out_dir,exist_ok=True)
        
        inf_purity = params['purity']
        if not fix_ploidy:
            inf_ploidy = params['ploidy']
        else:
            inf_ploidy = None
        
        purity = comb.split('_')[-1]
        parameters = pd.DataFrame({'purity':purity, 
                                   'ploidy':ploidy, 
                                   'inf_purity':inf_purity, 
                                   'inf_ploidy':inf_ploidy})
        parameters.to_csv(f'{out_dir}/params.csv', header=True)

        res = pd.DataFrame({'CN_Major':params["CN_Major"],
                    'CN_minor':params["CN_minor"],
                    'pos':[i for i in range(len(params["CN_minor"]))]})
        res.to_csv(f'{out_dir}/cna.csv', header=True)
        
        out_df = score_cn_predictions(data, res, sample_name=f'{args.sim}_{comb}')
        out_df['summary'].to_csv(f'{out_dir}/summary.csv', header=True)
        out_df['per_class_total'].to_csv(f'{out_dir}/summary_per_class_total.csv', header=True)
        out_df['per_class_pair'].to_csv(f'{out_dir}/summary_per_class_pair.csv', header=True)
        
        
    
    
