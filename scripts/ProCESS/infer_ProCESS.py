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
                                "dp": data_input["dp_snv"]})
    else:
        locate.initialize_model({"baf":data_input["baf"],
                                "dr":data_input["dr"], 
                                "dp_snp":data_input["dp_snp"], 
                                "vaf": None, 
                                "dp": None})

    ploidy = ploidy if ploidy is not None else 0
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
                            "bp_strength": 3.0,
                            "sample_type":"clinical"})
    
    ll = locate.run(steps = nsteps, param_optimizer = {"lr" : 0.05}, guide_kind="normal")
    params = locate.learned_parameters_Clonal()
    return params


def _per_class_table(y_true, y_pred, labels, *, label_type: str):
    """Build a per-class precision/recall/F1/support table for given labels order."""
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average=None
    )
    tbl = pd.DataFrame({
        label_type: labels,
        "precision": p,
        "recall": r,
        "F1": f1,
        "support": sup.astype(int),
    })
    return tbl

def score_cn_predictions(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    sample_name: str = "Sample",
    pos_col_truth: str = "pos",
    maj_col_truth: str = "major",
    min_col_truth: str = "minor",
    pos_col_pred: str = "pos",
    maj_col_pred: str = "CN_Major",
    min_col_pred: str = "CN_minor",
) -> Dict[str, pd.DataFrame]:
    """
    Compare true allele-specific CN (per-position) vs inferred CN.

    Returns a dict of DataFrames:
      - 'summary' : one-row global metrics
      - 'per_class_total' : per total-CN precision/recall/F1/support
      - 'per_class_pair'  : per (major:minor) precision/recall/F1/support
      - 'confusion_total' : confusion matrix (truth rows vs pred cols) on total CN
      - 'confusion_pair'  : confusion matrix on allele pairs (major:minor)
    """

    # --- prepare truth ---
    truth = truth_df[[pos_col_truth, maj_col_truth, min_col_truth]].copy()
    truth = truth.rename(columns={
        pos_col_truth: "pos",
        maj_col_truth: "t_major",
        min_col_truth: "t_minor"
    })
    
    # ensure integers and sort so major >= minor
    truth[["t_major", "t_minor"]] = truth[["t_major", "t_minor"]].astype(int)
    truth[["t_major", "t_minor"]] = np.sort(truth[["t_major", "t_minor"]].values, axis=1)[:, ::-1]
    truth["t_total"] = truth["t_major"] + truth["t_minor"]
    truth["t_pair"]  = truth["t_major"].astype(str) + ":" + truth["t_minor"].astype(str)

    # --- prepare predictions ---
    pred = pred_df[[pos_col_pred, maj_col_pred, min_col_pred]].copy()
    pred = pred.rename(columns={
        pos_col_pred: "pos",
        maj_col_pred: "p_major",
        min_col_pred: "p_minor"
    })
    pred[["p_major", "p_minor"]] = pred[["p_major", "p_minor"]].astype(int)
    pred[["p_major", "p_minor"]] = np.sort(pred[["p_major", "p_minor"]].values, axis=1)[:, ::-1]
    pred["p_total"] = pred["p_major"] + pred["p_minor"]
    pred["p_pair"]  = pred["p_major"].astype(str) + ":" + pred["p_minor"].astype(str)

    # --- merge on position ---
    merged = truth.merge(pred, on="pos", how="inner")
    if merged.empty:
        raise ValueError("No overlapping positions between truth and predictions.")

    # --- global metrics ---
    allelic_accuracy = (merged["t_pair"] == merged["p_pair"]).mean()
    total_accuracy   = (merged["t_total"] == merged["p_total"]).mean()
    total_mae        = np.abs(merged["t_total"] - merged["p_total"]).mean()

    # labels (ensure stable, meaningful ordering)
    # total CN labels: numeric ascending
    labels_total = sorted(set(merged["t_total"]).union(set(merged["p_total"])))
    
    # pair labels: sort by (total, major, minor) descending major then minor within total
    def _pair_key(s):
        a, b = map(int, s.split(":"))
        return (a + b, a, b)  # total first, then major, then minor
    labels_pair = sorted(set(merged["t_pair"]).union(set(merged["p_pair"])), key=_pair_key)

    # --- per-class tables ---
    per_class_total = _per_class_table(
        merged["t_total"].to_numpy(),
        merged["p_total"].to_numpy(),
        labels_total,
        label_type="totalCN"
    )
    per_class_pair = _per_class_table(
        merged["t_pair"].to_numpy(),
        merged["p_pair"].to_numpy(),
        labels_pair,
        label_type="pair"
    )

    # --- macro averages from the per-class tables (unweighted mean) ---
    per_class_total = per_class_total.query('support > 0')
    precision_total_macro = per_class_total["precision"].mean() if not per_class_total.empty else 0.0
    recall_total_macro    = per_class_total["recall"].mean()    if not per_class_total.empty else 0.0
    f1_total_macro        = per_class_total["F1"].mean()        if not per_class_total.empty else 0.0

    per_class_pair = per_class_pair.query('support > 0')
    precision_pair_macro = per_class_pair["precision"].mean() if not per_class_pair.empty else 0.0
    recall_pair_macro    = per_class_pair["recall"].mean()    if not per_class_pair.empty else 0.0
    f1_pair_macro        = per_class_pair["F1"].mean()        if not per_class_pair.empty else 0.0

    # --- confusion matrices ---
    confusion_total = pd.crosstab(
        merged["t_total"], merged["p_total"],
        rownames=["truth_totalCN"], colnames=["pred_totalCN"]
    ).reindex(index=labels_total, columns=labels_total, fill_value=0)

    confusion_pair = pd.crosstab(
        merged["t_pair"], merged["p_pair"],
        rownames=["truth_pair"], colnames=["pred_pair"]
    ).reindex(index=labels_pair, columns=labels_pair, fill_value=0)

    # --- summary (one row) ---
    summary = pd.DataFrame([{
        "sample": sample_name,
        "n_positions": int(len(merged)),
        "allelic_accuracy": float(allelic_accuracy),
        "total_cn_accuracy": float(total_accuracy),
        "total_cn_mae": float(total_mae),
        "precision_totalCN_macro": float(precision_total_macro),
        "recall_totalCN_macro": float(recall_total_macro),
        "f1_totalCN_macro": float(f1_total_macro),
        "precision_pair_macro": float(precision_pair_macro),
        "recall_pair_macro": float(recall_pair_macro),
        "f1_pair_macro": float(f1_pair_macro),
    }])

    return {
        "summary": summary,
        "per_class_total": per_class_total,
        "per_class_pair": per_class_pair,
        "confusion_total": confusion_total,
        "confusion_pair": confusion_pair,
    }


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
    parser.add_argument("-b", "--base", type=str, help="base", default = "")
    parser.add_argument("-s", "--sim", type=str, help="sample", default = "sim_21")
    
    parser.add_argument("-p", "--ploidy", type=str, help="ploidy", default = "True")
    parser.add_argument("-v", "--vaf", type=str, help="vaf", default = "False")  
    parser.add_argument("-B", "--bps", type=str, help="breakpoint", default = "False")  

    args = parser.parse_args()
    
    sim_dir = args.base + '/' + args.sim
    combinations = [ i for i in os.listdir(sim_dir) if os.path.isdir(sim_dir + '/' + i)]
    
    for comb in combinations:
        print(comb)
        data,data_input = read_data(f'{sim_dir}/{comb}/mirr_smooth_snv.csv')
        ploidy, _ = estimate_ploidy(data, return_details=True)
        
        fix_ploidy = eval(args.ploidy)
        vaf = eval(args.vaf)
        bps = eval(args.bps)
        
        if not bps:
            prior_bps = None
        params = run_hmm(data_input, fix_ploidy=fix_ploidy, vaf=vaf, bps = prior_bps, ploidy = ploidy)
        
        name = f'vaf_{vaf}_ploidy_{fix_ploidy}_bps_{bps}'
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
        parameters.to_csv(f'{out_dir}/params.csv', header=True, index=False)

        res = pd.DataFrame({'CN_Major':params["CN_Major"],
                    'CN_minor':params["CN_minor"],
                    'pos':[i for i in range(len(params["CN_minor"]))]})
        res.to_csv(f'{out_dir}/cna.csv', header=True, index=False)
        
        out_df = score_cn_predictions(data, res, sample_name=f'{args.sim}_{comb}')
        out_df['summary'].to_csv(f'{out_dir}/summary.csv', header=True, index=False)
        out_df['per_class_total'].to_csv(f'{out_dir}/summary_per_class_total.csv', header=True, index=False)
        out_df['per_class_pair'].to_csv(f'{out_dir}/summary_per_class_pair.csv', header=True, index=False)
        
        bins = 10
        segments = segment_allele_specific_cn(res)
        segs_merged = merge_micro_segments(segments, min_bins=bins)
        segments.to_csv(f'{out_dir}/segments.csv', header=True, index=False)
        segs_merged.to_csv(f'{out_dir}/merged_segments_{bins}.csv', header=True, index=False)
        
    
    
