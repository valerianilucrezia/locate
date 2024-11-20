#!/usr/bin/env python

import pandas as pd
import torch
import numpy as np

import argparse
from locate.segmentation.multivariate_clasp import MultivariateClaSP

def read_data(in_file, in_dir, ch, data_type = 'ill'):
    tmp_csv = f'{in_dir}/tmp_{data_type}_{ch}.csv'    

    data = pd.read_csv(in_file, sep = ',', on_bad_lines='skip') 
    data['pos'] = range(1, len(data) + 1)
    
    if data_type == 'ill':
        data['BAF_ILL'] = data['BAF_ILL'].apply(lambda x: 1 - x if x > 0.5 else x)
        data = data[data['BAF_ILL']>0]
        test_df = pd.DataFrame({"baf":data.BAF_ILL, "dr":data.DR_ILL, "pos":data.pos})
        test_df.to_csv(path_or_buf=tmp_csv, sep=",")
    
    elif data_type == 'np':
        data['BAF_NP'] = data['BAF_NP'].apply(lambda x: 1 - x if x > 0.5 else x)
        data = data[data['BAF_NP']>0]
        test_df = pd.DataFrame({"baf":data.BAF_NP, "dr":data.DR_NP, "pos":data.pos})
        test_df.to_csv(path_or_buf=tmp_csv, sep=",")
    
    return tmp_csv

def run_segmentation(tmp_file, md = 'sum', thr = 1e-10, wsize = 50):
    multiClasp = MultivariateClaSP(tmp_file, 
                                   mode=md, 
                                   out_dir='./tmp', 
                                   threshold=thr, 
                                   window_size=wsize,
                                   cna_id = None,
                                   frequencies = ['baf', 'dr'])
    multiClasp.analyze_time_series()
    return [multiClasp.all_cps, f'{md}_{thr}_{wsize}']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, help="indir", default = "")
    parser.add_argument("-s", "--sample", type=str, help="sample", default = "PDO61")
    parser.add_argument("-c", "--chr", type=str, help="chromosome", default = "1")
    parser.add_argument("-o", "--outdir", type=str, help="outdir", default = "")
    args = parser.parse_args()

    f = f'{args.indir}/{args.sample}_chr{args.chr}.csv'
    print('start processing data')
    data_ill = read_data(f, args.indir, args.chr, data_type = 'ill')
    data_np = read_data(f, args.indir, args.chr, data_type = 'np')

    print('start segmentation Illumina')
    bp_ill = run_segmentation(data_ill)

    print('start segmentation Nanopore')
    bp_np = run_segmentation(data_np)
    
    np.save(arr = np.array(bp_ill[0]), file = f'{args.outdir}/{args.sample}_chr{args.chr}_ILL_{bp_ill[1]}')
    np.save(arr = np.array(bp_np[0]), file = f'{args.outdir}/{args.sample}_chr{args.chr}_ILL_{bp_np[1]}')
    
    
    
