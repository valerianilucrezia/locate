#!/usr/bin/env python

import pyro
from pyro.optim import Adam, ClippedAdam
from pyro.infer import TraceMeanField_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, Trace_ELBO

import pandas as pd
import torch
import numpy as np

import argparse
import os

import locate.locate as l
from locate.models import Clonal

print(torch.cuda.is_available())

def read_data(in_file, data_type = 'ill'):
    #tmp_csv = f'{in_dir}/tmp_{data_type}_{ch}.csv'    

    data = pd.read_csv(in_file, sep = ',', on_bad_lines='skip') 
    data['pos'] = range(1, len(data) + 1)
    
    if data_type == 'ill':
        data['original_BAF'] = data['BAF_ILL']
        data['BAF_ILL'] = data['BAF_ILL'].apply(lambda x: 1.01 - x if x > 0.5 else x)
        data['BAF_ILL'] = data['BAF_ILL'].apply(lambda x: 0.01 if x == 0 else x)
        
        data = data[data['BAF_ILL']>0]
        data_input = {'baf':torch.tensor(np.array(data.BAF_ILL).reshape(-1, 1)),
                      'dr':torch.tensor(np.array(data.DR_ILL).reshape(-1, 1)),
                      'dp_snp':torch.tensor(np.array(data.DP_ILL).reshape(-1, 1)),
                      'orginal_baf':torch.tensor(np.array(data.original_BAF).reshape(-1, 1))}
    
    elif data_type == 'np':
        data['original_BAF'] = data['BAF_NP']
        data['BAF_NP'] = data['BAF_NP'].apply(lambda x: 1.01 - x if x > 0.5 else x)
        data['BAF_NP'] = data['BAF_NP'].apply(lambda x: 0.01 if x == 0 else x)
        
        data = data[data['BAF_NP']>0]
        data_input = {'baf':torch.tensor(np.array(data.BAF_NP).reshape(-1, 1)),
                      'dr':torch.tensor(np.array(data.DR_NP).reshape(-1, 1)),
                      'dp_snp':torch.tensor(np.array(data.DP_NP).reshape(-1, 1)),
                      'orginal_baf':torch.tensor(np.array(data.original_BAF).reshape(-1, 1))}
    
    return data_input

def run_hmm(data_input, gpu = False, nsteps = 400):
    locate = l.LOCATE(CUDA = gpu)

    locate.set_model(Clonal)
    locate.set_optimizer(ClippedAdam)
    locate.set_loss(TraceEnum_ELBO)
    locate.initialize_model({"baf": data_input["baf"],
                            "dr": data_input["dr"], 
                            "dp_snp": data_input["dp_snp"], 
                            "vaf": None, 
                            "dp": None
                            })

    locate.set_model_params({"jumping_prob" : 1e-6,
                            "fix_purity": False,
                            "prior_purity": 0.99,
                            "prior_ploidy": 2,
                            "scaling_factors": [1,1,1],
                            "prior_bp": False,
                            'init_probs': torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
                            })
    
    ll = locate.run(steps = nsteps, param_optimizer = {"lr" : 0.05})
    params = locate.learned_parameters_Clonal()
    return params



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, help="indir", default = "")
    parser.add_argument("-s", "--sample", type=str, help="sample", default = "PDO61")
    parser.add_argument("-c", "--chr", type=str, help="chromosome", default = "1")
    parser.add_argument("-o", "--outdir", type=str, help="outdir", default = "")
    args = parser.parse_args()

    gpu = torch.cuda.is_available()

    f = f'{args.indir}/{args.sample}_chr{args.chr}.csv'
    data_ill = read_data(f, data_type = 'ill')
    data_np = read_data(f, data_type = 'np')
 
    params_ill = run_hmm(data_ill, gpu = gpu)
    params_np = run_hmm(data_np, gpu = gpu)
    
    res_ill = pd.DataFrame({'CN_Major':params_ill["CN_Major"]+0.05,
                           'CN_minor':params_ill["CN_minor"]-0.05,
                            'baf':data_ill['baf'].view(-1).tolist(),
                            'dr':data_ill['dr'].view(-1).tolist(),
                            'original_baf':data_ill['orginal_baf'].view(-1).tolist(),
                            'pos':[i for i in range(len(params_ill["CN_minor"]))],
                            })
    
    res_np = pd.DataFrame({'CN_Major':params_np["CN_Major"]+0.05,
                           'CN_minor':params_np["CN_minor"]-0.05,
                            'baf':data_np['baf'].view(-1).tolist(),
                            'dr':data_np['dr'].view(-1).tolist(),
                            'original_baf':data_np['orginal_baf'].view(-1).tolist(),
                            'pos':[i for i in range(len(params_np["CN_minor"]))],
                            })
    

    os.makedirs(f'{args.outdir}/{args.sample}/', exist_ok=True)
    res_ill.to_csv(f'{args.outdir}/{args.sample}/{args.sample}_chr{args.chr}_inference_ILL.csv', header=True)
    res_np.to_csv(f'{args.outdir}/{args.sample}//{args.sample}_chr{args.chr}_inference_NP.csv', header=True)
    
    
