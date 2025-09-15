
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

from sklearn.metrics import precision_recall_fscore_support


def read_data(in_file):
    data = pd.read_csv(in_file, sep = ',', on_bad_lines='skip')
    data = data.sort_values(by=['pos']) 
    data['baf'] = data['median_BAF'].apply(lambda x: 1 - x if x > 0.5 else x)
    data = data[data['baf']>0]
    data_input = {'baf':torch.tensor(np.array(data.baf).reshape(-1, 1)),
                'dr':torch.tensor(np.array(data.median_DR).reshape(-1, 1)),
                'dp_snp':torch.tensor(np.array(data.mean_DP).reshape(-1, 1)),
                'orginal_baf':torch.tensor(np.array(data.median_BAF).reshape(-1, 1))}
                #'vaf':torch.tensor(np.array(data.vaf).reshape(-1, 1)),
                #'dp_snv':torch.tensor(np.array(data.DP).reshape(-1, 1))
    data['pos'] = range(1, len(data) + 1)
    return data, data_input

def run_hmm(data_input, 
            gpu = False, 
            nsteps = 100, 
            fix_ploidy = True, 
            fix_purity = True,
            vaf = False, 
            bps = None,
            ploidy = None,
            purity = None,
            sample_type = None):
    
    locate = l.LOCATE(CUDA = gpu)

    locate.set_model(Clonal)
    locate.set_optimizer(ClippedAdam)
    locate.set_loss(TraceEnum_ELBO)
    
    if vaf:
        locate.initialize_model({"baf":data_input["baf"],
                                "dr":data_input["dr"], 
                                "dp_snp":data_input["dp_snp"], 
                                "vaf": data_input["vaf"], 
                                "dp": data_input["dp_snv"]})
    else:
        locate.initialize_model({"baf":data_input["baf"],
                                "dr":data_input["dr"], 
                                "dp_snp":data_input["dp_snp"], 
                                "vaf": None, 
                                "dp": None})

    ploidy = ploidy if ploidy is not None else 0
    locate.set_model_params({"jumping_prob" : 1e-6,
                            "fix_purity": fix_purity,
                            "fix_ploidy" : fix_ploidy, 
                            "prior_purity": purity,
                            "prior_ploidy": ploidy,
                            "scaling_factors": [1,1,1],
                            'hidden_dim': 4,
                            "prior_bp": bps,
                            "lambda_cn": 1.0,          # ΔCN penalty slope used to shape the prior
                            "alpha_conc": 50.0,        # global Dirichlet concentration (higher = stronger prior)
                            "alpha_self_boost": 3.0,   # extra mass on the diagonal
                            "alpha_dip_boost": 0.5,    # extra mass on diploid column (if present)
                            "bp_strength": 3.0,
                            "sample_type":sample_type})
    
    ll = locate.run(steps = nsteps, param_optimizer = {"lr" : 0.05}, guide_kind="normal")
    params = locate.learned_parameters_Clonal()
    return params


def segment_allele_specific_cn(df, pos_col="pos",
                               major_col="CN_Major", minor_col="CN_minor"):
    df = df.sort_values(pos_col).reset_index(drop=True)
    change = ((df[major_col] != df[major_col].shift()) |
              (df[minor_col] != df[minor_col].shift()))
    seg_id = change.cumsum()
    segs = (df.groupby(seg_id)
              .agg(seg_start=(pos_col, "min"),
                   seg_end=(pos_col, "max"),
                   CN_major=(major_col, "first"),
                   CN_minor=(minor_col, "first"),
                   n_bins=(pos_col, "size"))
              .reset_index(drop=True))
    segs["length"] = segs["seg_end"] - segs["seg_start"] + 1
    segs["seg_id"] = np.arange(len(segs))
    return segs[["seg_id","seg_start","seg_end","length","n_bins","CN_major","CN_minor"]]

def _l1_dist(a_major, a_minor, b_major, b_minor):
    return abs(a_major - b_major) + abs(a_minor - b_minor)

def coalesce_same_cn(segs):
    """
    Merge consecutive segments that share the same (CN_major, CN_minor).
    Treat segments as mergeable if they are touching or overlapping.
    """
    if len(segs) <= 1:
        return segs.copy()

    s = segs.sort_values("seg_start").reset_index(drop=True).copy()
    out = []
    cur = s.iloc[0].to_dict()

    for i in range(1, len(s)):
        row = s.iloc[i]
        same_state = (row["CN_major"] == cur["CN_major"]) and (row["CN_minor"] == cur["CN_minor"])
        touching = row["seg_start"] <= cur["seg_end"] + 1  # contiguous or overlap
        if same_state and touching:
            # extend current
            cur["seg_end"] = max(cur["seg_end"], int(row["seg_end"]))
            cur["length"]  = cur["seg_end"] - cur["seg_start"] + 1
            cur["n_bins"]  = int(cur["n_bins"]) + int(row["n_bins"])
        else:
            out.append(cur)
            cur = row.to_dict()
    out.append(cur)

    out = pd.DataFrame(out).reset_index(drop=True)
    out["seg_id"] = np.arange(len(out))
    return out[["seg_id","seg_start","seg_end","length","n_bins","CN_major","CN_minor"]]


def merge_micro_segments(segs, min_bins=5, max_passes=100):
    """
    Merge segments shorter than min_bins into neighbors; after each pass,
    coalesce any newly adjacent equal-CN segments.
    """
    s = segs.sort_values("seg_start").reset_index(drop=True).copy()

    def _merge_once(s):
        if len(s) <= 1:
            return s, False
        short_idx = s.index[s["n_bins"] < min_bins].tolist()
        if not short_idx:
            return s, False

        to_drop = []
        for i in short_idx:
            if i in to_drop or i >= len(s):
                continue
            left = i - 1 if i - 1 >= 0 else None
            right = i + 1 if i + 1 < len(s) else None
            if left is None and right is None:
                continue

            target = None
            # Prefer identical state neighbor
            if left is not None and \
               (s.loc[left, "CN_major"] == s.loc[i, "CN_major"]) and \
               (s.loc[left, "CN_minor"] == s.loc[i, "CN_minor"]):
                target = left
            if right is not None and \
               (s.loc[right, "CN_major"] == s.loc[i, "CN_major"]) and \
               (s.loc[right, "CN_minor"] == s.loc[i, "CN_minor"]):
                if target is None or s.loc[right, "length"] > s.loc[target, "length"]:
                    target = right

            # Else closest L1 distance, tie -> larger length, tie -> left
            if target is None:
                candidates = []
                if left is not None:
                    dL = _l1_dist(s.loc[i,"CN_major"], s.loc[i,"CN_minor"],
                                  s.loc[left,"CN_major"], s.loc[left,"CN_minor"])
                    candidates.append(("L", left, dL, s.loc[left,"length"]))
                if right is not None:
                    dR = _l1_dist(s.loc[i,"CN_major"], s.loc[i,"CN_minor"],
                                  s.loc[right,"CN_major"], s.loc[right,"CN_minor"])
                    candidates.append(("R", right, dR, s.loc[right,"length"]))
                candidates.sort(key=lambda x: (x[2], -x[3], x[0]))
                target = candidates[0][1]

            # Merge i into target
            new_start = min(s.loc[i, "seg_start"], s.loc[target, "seg_start"])
            new_end   = max(s.loc[i, "seg_end"],   s.loc[target, "seg_end"])
            s.loc[target, "seg_start"] = new_start
            s.loc[target, "seg_end"]   = new_end
            s.loc[target, "length"]    = new_end - new_start + 1
            s.loc[target, "n_bins"]    = s.loc[target, "n_bins"] + s.loc[i, "n_bins"]
            to_drop.append(i)

        if not to_drop:
            return s, False

        s = s.drop(index=to_drop).sort_values("seg_start").reset_index(drop=True)
        # Coalesce equal-CN neighbors formed by this pass
        s = coalesce_same_cn(s)
        return s, True

    changed = True
    passes = 0
    while changed and passes < max_passes:
        s, changed = _merge_once(s)
        passes += 1

    # Final coalesce in case last pass only touched boundaries
    s = coalesce_same_cn(s)
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, help="base", default = "/orfeo/scratch/area/lvaleriani/locate/scripts/out")
    parser.add_argument("-s", "--sample", type=str, help="sample", default = "COLO829")

    parser.add_argument("-t", "--type", type=str, help="sample type: cell_line or clinical", default = "cell_line")
    parser.add_argument("-S", "--sigma", type=float, help="prior ploidy", default = 0.9)
    parser.add_argument("-r", "--rho", type=float, help="prior purity", default = 0.9)
    parser.add_argument("-P", "--purity", type=str, help="purity", default = "False")
    parser.add_argument("-p", "--ploidy", type=str, help="ploidy", default = "False")
    parser.add_argument("-v", "--vaf", type=str, help="vaf", default = "False")  
    parser.add_argument("-B", "--bps", type=str, help="breakpoint", default = "False")  
    
    gpu = torch.cuda.is_available()
    
    args = parser.parse_args()
    path = f'{args.base}/{args.sample}/{args.sample}_smooth.csv'
    data, data_input = read_data(path)

    sample_type = args.type
    
    fix_purity = eval(args.purity)
    if fix_purity:
        prior_purity = args.rho
    else: 
        prior_purity = None
        
    fix_ploidy = eval(args.ploidy)
    if fix_ploidy:
        prior_ploidy = args.sigma
    else: 
        prior_ploidy = None
    
    vaf = eval(args.vaf)
    bps = eval(args.bps)
    
    if not bps:
        prior_bps = None
    params = run_hmm(data_input, 
                     fix_purity = fix_purity, 
                     fix_ploidy = fix_ploidy, 
                     vaf = vaf, 
                     bps = prior_bps, 
                     ploidy = prior_ploidy, 
                     purity = prior_purity,
                     sample_type = sample_type,
                     gpu = gpu)
    
    name = f'vaf_{vaf}_ploidy_{fix_ploidy}_purity_{fix_purity}_bps_{bps}'
    out_dir = f'{path}/{args.sample}/{name}'
    os.makedirs(out_dir,exist_ok=True)
    
    if not fix_ploidy:
        inf_ploidy = params['ploidy']
    else:
        inf_ploidy = None

    if not fix_purity:
        inf_purity = params['purity']
    else:
        inf_purity = None
    
    parameters = pd.DataFrame({'inf_purity':inf_purity, 
                               'inf_ploidy':inf_ploidy})
    parameters.to_csv(f'{out_dir}/params.csv', header=True, index=False)

    res = pd.DataFrame({'CN_Major':params["CN_Major"],
            'CN_minor':params["CN_minor"],
            'pos':[i for i in range(len(params["CN_minor"]))]})
    res.to_csv(f'{out_dir}/cna.csv', header=True, index=False)
    
    bins = 10
    segments = segment_allele_specific_cn(res)
    segs_merged = merge_micro_segments(segments, min_bins=bins)
    segments.to_csv(f'{out_dir}/segments.csv', header=True, index=False)
    segs_merged.to_csv(f'{out_dir}/merged_segments_{bins}.csv', header=True, index=False)

    
    
    
