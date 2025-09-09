#!/usr/bin/env python

import pandas as pd
import torch
import numpy as np

import argparse
import os

from locate.segmentation.multivariate_clasp import MultivariateClaSP

def read_data(base):
    data = pd.read_csv(base + "/mirr_smooth_snv.csv", sep = ',', on_bad_lines='skip') 

    data['pos'] = range(1, len(data) + 1)
    data['median_baf'] = data['median_baf'].apply(lambda x: 1 - x if x > 0.5 else x)
    data = data[data['median_baf']>0]

    tmp_csv = base + '/tmp.csv'
    test_df = pd.DataFrame({"baf":data.median_baf, "dr":data.median_dr, "pos":data.pos, 'vaf':data.vaf})
    test_df.to_csv(path_or_buf=tmp_csv, sep=",")
    return

def run_segmentation(base, md = 'sum', thr = 1e-10, wsize = 50):
    multiClasp = MultivariateClaSP(base + '/tmp.csv', 
                                   mode=md, 
                                   out_dir=base, 
                                   threshold=thr, 
                                   window_size=wsize,
                                   cna_id = None,
                                   frequencies = ['baf', 'dr'])
    multiClasp.analyze_time_series()
    print('inferred breakpoints =', multiClasp.all_cps)
    np.save(f'{base}/{md}_{wsize}_{thr}.npy', multiClasp.all_cps)
    return 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, help="base", default = "")
    parser.add_argument("-s", "--sim", type=str, help="sample", default = "sim_21")
    
    parser.add_argument("-m", "--mode", type=str, help="mode", default = "max")
    parser.add_argument("-w", "--windsize", type=int, help="window size", default = 5)
    parser.add_argument("-t", "--thr", type=float, help="thr", default = 1e-15)    

    args = parser.parse_args()
    
    sim_dir = args.base + '/' + args.sim
    combinations = [ i for i in os.listdir(sim_dir) if os.path.isdir(sim_dir + '/' + i)]
    
    for comb in combinations:
        print(f'data = {args.sim} {comb}')
        print(f'with params = {args.mode}, {args.windsize}, {args.thr}')

        read_data(sim_dir+'/'+comb)
        run_segmentation(sim_dir+'/'+comb, md = args.mode, thr = args.thr, wsize = args.windsize)
    
