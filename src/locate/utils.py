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
from typing import Dict, Tuple, Optional


def plot_purity_ploidy_hist(draws, bins=50):
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    if "purity" in draws:
        axs[0].hist(draws["purity"].numpy(), bins=bins, density=True)
        axs[0].set_title("Posterior of Purity")
    if "ploidy" in draws:
        axs[1].hist(draws["ploidy"].numpy(), bins=bins, density=True)
        axs[1].set_title("Posterior of Ploidy")
    plt.tight_layout()
    plt.show()
    
def estimate_ploidy(
    df: pd.DataFrame,
    seg_id_col: str = "seg_id",
    include_sex_chroms: bool = False,
    valid_autosomes: Optional[set] = None,
    return_details: bool = False) -> Tuple[float, Optional[pd.DataFrame]]:
    """
    Estimate sample ploidy from a per-variant dataframe annotated with segment IDs.

    Assumes df[seg_id_col] has the form "chr:start:end:major:minor", e.g. "1:207651979:248956422:1:1".
    Ploidy is computed as the length-weighted mean total copy number across unique segments:
        ploidy = sum_i ( (major_i + minor_i) * length_i ) / sum_i length_i

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (one row per variant), containing a column with segment IDs.
    seg_id_col : str, default "seg_id"
        Column name holding the segment identifier string.
    include_sex_chroms : bool, default False
        If False, excludes segments on sex chromosomes (X, Y, chrX, chrY, MT/M).
    valid_autosomes : set or None
        If provided, keep only chromosomes in this set (after stripping optional 'chr' prefix).
        Example: set(map(str, range(1, 23))) for autosomes 1..22.
    return_details : bool, default False
        If True, also return a per-segment dataframe used in the calculation.

    Returns
    -------
    ploidy : float
        Estimated ploidy (length-weighted genome-wide total copy number).
    details_df : Optional[pd.DataFrame]
        If return_details=True, a dataframe with one row per unique segment:
        columns: ['chr','start','end','length','major','minor','total_cn','weight'].
    """
    if seg_id_col not in df.columns:
        raise ValueError(f"Column '{seg_id_col}' not found in dataframe.")

    # Parse seg_id safely with a vectorized regex extraction
    # Expected 5 fields: chr, start, end, major, minor
    pat = r"^([^:]+):(\d+):(\d+):(\d+):(\d+)$"
    parsed = df[seg_id_col].astype(str).str.extract(pat)
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), seg_id_col].unique()[:5]
        raise ValueError(
            "Some seg_id values do not match 'chr:start:end:major:minor'. "
            f"Examples of problematic entries: {list(bad)}"
        )

    parsed.columns = ["chr", "start", "end", "major", "minor"]
    # Attach and drop duplicates so each segment is counted once
    segs = parsed.copy()
    segs[["start", "end", "major", "minor"]] = segs[["start", "end", "major", "minor"]].astype(np.int64)

    # Normalize chromosome labels (e.g., 'chr1' -> '1')
    def norm_chr(c):
        c = str(c)
        c = c[3:] if c.lower().startswith("chr") else c
        return c

    segs["chr_norm"] = segs["chr"].map(norm_chr)

    # Optionally exclude sex/MT chromosomes
    if not include_sex_chroms:
        sex_like = {"x", "y", "mt", "m"}
        segs = segs[~segs["chr_norm"].str.lower().isin(sex_like)]

    # Optionally keep only specified autosomes
    if valid_autosomes is not None:
        segs = segs[segs["chr_norm"].isin(valid_autosomes)]

    # Deduplicate identical segments
    segs = segs.drop_duplicates(subset=["chr", "start", "end", "major", "minor"]).copy()

    # Compute lengths and total copy number
    segs["length"] = (segs["end"] - segs["start"] + 1).clip(lower=0)
    segs = segs[segs["length"] > 0]

    if segs.empty:
        raise ValueError("No valid segments left after filtering; cannot estimate ploidy.")

    segs["total_cn"] = segs["major"] + segs["minor"]
    segs["weight"] = segs["length"].astype(float)

    # Length-weighted mean total copy number
    ploidy = float((segs["total_cn"] * segs["weight"]).sum() / segs["weight"].sum())

    if return_details:
        details = segs.rename(columns={"chr": "chr_raw"})
        details = details[["chr_raw", "chr_norm", "start", "end", "length", "major", "minor", "total_cn", "weight"]]
        details = details.reset_index(drop=True)
        return ploidy, details
    else:
        return ploidy, None
    
    

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
    plot_loss(loss)
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