import pandas as pd
import numpy as np
import random
from scipy.stats import pareto
import os



def check_sum_karyo(k1,k2):
    """_summary_

    Parameters
    ----------
    k1 : _type_
        _description_
    k2 : _type_
        _description_
    """
    
    a1 = int(k1.split(':')[0])
    b1 = int(k1.split(':')[1])
    
    a2 = int(k2.split(':')[0])
    b2 = int(k2.split(':')[1])
    
    return(abs((a1-a2)) + abs((b1-b2)))
  

# size chr21
def simulate_segment(genome_size = 20000000, 
                    segments = 5, 
                    coverage = 50,
                    mu = 0.000003,
                    w = 20,
                    dt = 20, 
                    purity = 1,
                    only_clonal = False,
                    CNA = ["1:0", "1:1", "2:0", "2:1", "2:2"]):
    """_summary_

    Parameters
    ----------
    genome_size : int, optional
        _description_, by default 20000000
    segments : int, optional
        _description_, by default 5
    coverage : int, optional
        _description_, by default 50
    mu : float, optional
        _description_, by default 0.000003
    w : int, optional
        _description_, by default 20
    dt : int, optional
        _description_, by default 20
    purity : int, optional
        _description_, by default 1
    only_clonal : bool, optional
        _description_, by default False
    CNA : list, optional
        _description_, by default ["1:0", "1:1", "2:0", "2:1", "2:2"]

    Returns
    -------
    _type_
        _description_
    """
    
    start = []
    end = [0]
    length = []
    id = []
    cov = []
    snp = []
    snv = []
    Major_1 = []
    minor_1 = []
    Major_2 = []
    minor_2 = []
    l_ccf = []
    l_tp = []
    purity = [purity for i in range(segments)]
    
    
    seg_size = np.random.dirichlet(np.ones(segments), size=1) * genome_size
    seg_size = list(seg_size[0])
    
    ploidy = []
    # num_bins =  genome size
    for i, s in enumerate(seg_size):
        cna_start = end[-1]
        cna_end = cna_start + np.floor(s)
      
        if only_clonal:
          ccf = 0
        else:
          ccf = np.random.choice([i/10 for i in range(0,10,1)], 1)[0]
        
        copy_number_1 = np.random.choice(CNA)
        cna_copy_number_major_1 = int(copy_number_1.split(':')[0])
        cna_copy_number_minor_1 = int(copy_number_1.split(':')[1])
        
        if ccf == 0:
          tp = 'clonal'
          
          copy_number_2 = ''
          cna_copy_number_major_2 = ''
          cna_copy_number_minor_2 = ''
          
        else:
          tp = 'sub-clonal'
          
          cna_idx = CNA.index(copy_number_1)
          copy_number_2 = np.random.choice(CNA[cna_idx:])
          while check_sum_karyo(copy_number_1,copy_number_2) >= 2:
              copy_number_2 = np.random.choice(CNA[cna_idx:])
          
          cna_copy_number_major_2 = int(copy_number_2.split(':')[0])
          cna_copy_number_minor_2 = int(copy_number_2.split(':')[1])
        
        nSNP = int((cna_end - cna_start)/15000) #heterozygous
        nSNV = nSNP #np.random.poisson(size = 1, lam = int((cna_end - cna_start) * mu * w * dt))[0]
        
        tmp_ploidy = (cna_copy_number_major_1+cna_copy_number_minor_1) * (cna_end - cna_start)
        ploidy.append(tmp_ploidy)
        
        start.append(cna_start)
        end.append(cna_end)
        length.append(cna_end - cna_start)
        id.append(i)        
        cov.append(coverage)
        Major_1.append(cna_copy_number_major_1)
        minor_1.append(cna_copy_number_minor_1)
        Major_2.append(cna_copy_number_major_2)
        minor_2.append(cna_copy_number_minor_2)
        snp.append(nSNP)
        snv.append(nSNV) 
        l_ccf.append(ccf)
        l_tp.append(tp)
    
    #pp = np.sum(Major_1+minor_1)/len(Major_1)
    #print(pp)
    final_ploidy = np.sum(ploidy)/genome_size
    #print(final_ploidy)
    df = pd.DataFrame({'id':id, 
                       'start':start, 
                       'end':end[1:], 
                       'len':length, 
                       'coverage':cov,
                       'SNP':snp,
                       'SNV':snv,
                       'purity':purity, 
                       'ploidy':final_ploidy,
                       'Major_1':Major_1,
                       'minor_1':minor_1, 
                       'Major_2':Major_2,
                       'minor_2':minor_2,
                       'ccf':l_ccf, 
                       'type':l_tp})
    return df

def simulate_SNVs_clonal(seg, tail_prop = 0.250):
    """_summary_

    Parameters
    ----------
    seg : _type_
        _description_
    tail_prop : float, optional
        _description_, by default 0.250

    Returns
    -------
    _type_
        _description_
    """
    
    or_nvaf = int(seg.SNV)
    ntail = round(or_nvaf * tail_prop)
    nvaf = or_nvaf - ntail
    
    peaks = get_clonal_peaks(seg.Major_1, seg.minor_1, purity =  seg.purity)
    mix_prop = np.random.uniform(low = 0, high = 1, size = len(peaks)) 
    
    mix_prop = mix_prop / sum(mix_prop)
    peaks_of_mutations = np.random.choice(peaks, size = nvaf, replace = True, p = list(mix_prop))
    
    dp = np.random.poisson(lam = seg.coverage, size = nvaf)
    nv = np.random.binomial(n = seg.coverage, p = peaks_of_mutations, size = nvaf)
    vaf = nv/dp
    
    tail = my_pareto_betabin(ntail)
    dp_tail = np.random.poisson(lam = seg.coverage, size = tail.shape[0])
    nv_tail = np.around(tail * dp_tail[0:len(tail)], 0)
    
    nv_all = np.concatenate([nv, nv_tail])
    dp_all = np.concatenate([dp, dp_tail])
    
    nsnv = dp_all.shape[0]
    pos = np.random.randint(seg.start, seg.end, nsnv)
    
    pks = list(set(peaks_of_mutations))
    df = pd.DataFrame({'segID':[seg.id for _ in range(nsnv)],
                       'pos':pos,
                       'nv':nv_all,
                       'coverage':dp_all,
                       'vaf':np.divide(nv_all, dp_all), 
                       'CN_1':[f'{int(seg.Major_1)}:{int(seg.minor_1)}' for _ in range(nsnv)],
                       'CN_2':['' for _ in range(nsnv)],                       
                       'ccf':['' for _ in range(nsnv)],
                       'purity':[seg.purity for _ in range(nsnv)],
                       'true_peaks':[pks for _ in range(nsnv)],
                       'model':['clonal' for _ in range(nsnv)]
                       })
    return df

    
def get_clonal_peaks(M, m, purity):
    """_summary_

    Parameters
    ----------
    M : _type_
        _description_
    m : _type_
        _description_
    purity : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    multiplicities = [M, m]
    n_tot = sum(multiplicities)
    
    if (M == 2 or m == 2 ):
        multiplicities = [1, 2]
    
    multiplicities = [i for i in multiplicities if i!=0]
    multiplicities = list(set(multiplicities))
    
    peaks = []
    for m in multiplicities:
        peaks.append((m * purity) / (n_tot * purity + 2 * (1 - purity)))
    return peaks


def simulate_SNPs_clonal(seg):
    """_summary_

    Parameters
    ----------
    seg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    nsnp = int(seg.SNP)
    pos = np.random.randint(seg.start, seg.end, nsnp)
    dp = np.random.poisson(seg.coverage, nsnp) 
    purity = seg.purity
    
    # BAF
    exp_baf = ((purity * seg.minor_1) + (1-purity)) / ((purity * (seg.Major_1 + seg.minor_1)) +  (2 * (1-purity)))
    # if seg.minor_1 == 0:
    #         exp_baf = 0.01
    alpha = ((dp - 2) * exp_baf + 1) / (1 - exp_baf)
    sim_baf = np.random.beta(alpha, dp)
    
    # DR
    ploidy = ((2*(1-purity)) + (purity * seg.ploidy))
    exp_dr = ((2 * (1-purity)) + (purity * (seg.Major_1 + seg.minor_1))) / ploidy
    sim_dr = np.random.gamma(shape = exp_dr * np.sqrt(dp) + 1, 
                             scale = 1/np.sqrt(dp))
    
    df =  pd.DataFrame({'segID':[int(seg.id) for _ in range(nsnp)],
                       'pos':pos,
                       'baf':sim_baf,
                       'dr':sim_dr,
                       'cov':dp,
                       'CN_1':[f'{int(seg.Major_1)}:{int(seg.minor_1)}' for _ in range(nsnp)],
                       'CN_2':['' for _ in range(nsnp)],
                       'ccf':['' for _ in range(nsnp)],
                       'purity':[purity for _ in range(nsnp)]})
    return df

def expected_baf_subclonal(seg):
    """_summary_

    Parameters
    ----------
    seg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    nA1 = seg.Major_1
    nB1 = seg.minor_1
    nA2 = seg.Major_2
    nB2 = seg.minor_2
    ccf = seg.ccf
    purity = seg.purity
    
    num = min(nA1 * ccf + nA2 * (1 - ccf), nB1 * ccf + nB2 * (1 - ccf))
    den = purity * ((nA1 + nB1) * ccf + (1 - ccf) * (nA2 + nB2)) + 2 * (1 - purity)
    exp_baf = num / den
    
    return exp_baf
  
def expected_dr_subclonal(seg):
    """_summary_

    Parameters
    ----------
    seg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    na1 = seg.Major_1
    nb1 = seg.minor_1
    na2 = seg.Major_2
    nb2 = seg.minor_2
    ccf = seg.ccf
    purity = seg.purity
    ploidy = ((2*(1-purity)) + (purity * seg.ploidy))
    expected_dr = (2*(1-purity) + purity*(ccf*(na1 + nb1) + (1- ccf)*(na2 + nb2))) / ploidy 
    
    return expected_dr


def simulate_SNPs_subclonal(seg):
    """_summary_

    Parameters
    ----------
    seg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    nsnp = int(seg.SNP)
    pos = np.random.randint(seg.start, seg.end, nsnp)
    dp = np.random.poisson(seg.coverage, nsnp)
    purity = seg.purity
    ccf = seg.ccf
    
    # BAF
    exp_baf = expected_baf_subclonal(seg)
    alpha = ((dp - 2) * exp_baf + 1) / (1 - exp_baf)
    sim_baf = np.random.beta(alpha, dp) 
    
    # DR
    exp_dr = expected_dr_subclonal(seg)
    sim_dr = np.random.gamma(shape = exp_dr * np.sqrt(dp) + 1, scale = 1/np.sqrt(dp))

    
    df =  pd.DataFrame({'segID':[int(seg.id) for _ in range(nsnp)],
                       'pos':pos,
                       'baf':sim_baf,
                       'dr':sim_dr,
                       'cov':dp,
                       'CN_1':[f'{int(seg.Major_1)}:{int(seg.minor_1)}' for _ in range(nsnp)],
                       'CN_2':[f'{int(seg.Major_2)}:{int(seg.minor_2)}' for _ in range(nsnp)],
                       'ccf':[ccf for _ in range(nsnp)],
                       'purity':[purity for _ in range(nsnp)]})
    return df


def my_pareto_betabin(N):
    """_summary_

    Parameters
    ----------
    N : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    
    tail_cutoff = 0.2
    shape = np.random.uniform(size = 1, low = 1, high = 3) # shape of the pareto depends on mutation rate
    scale = .05
    tails = pareto.rvs(b = shape, scale = scale, size = N)
    tails = tails[tails <= tail_cutoff]
    return tails # return vaf of subclonal mutations



def vaf_subclonal(seg, mix_prop, peaks, tail_prop = 0.250):
    """_summary_

    Parameters
    ----------
    seg : _type_
        _description_
    mix_prop : _type_
        _description_
    peaks : _type_
        _description_
    tail_prop : float, optional
        _description_, by default 0.250

    Returns
    -------
    _type_
        _description_
    """
    
    n_muts_sub = seg.SNV
    n_muts_tail_1 = round(n_muts_sub * tail_prop)
    n_muts_tail_2 = round(n_muts_sub * tail_prop)
    #print(n_muts_tail_1, n_muts_tail_2)
    
    #peaks_of_mutations = sample(peaks, n_muts_sub, prob = mix_prop, replace = T)
    peaks_of_mutations = np.random.choice(peaks, size = n_muts_sub, replace = True, p = mix_prop)

    dp_subcl = np.random.poisson(lam = seg.coverage, size = n_muts_sub)
    dp_tail_1 = np.random.poisson(lam = seg.coverage, size = n_muts_tail_1)
    dp_tail_2 = np.random.poisson(lam = seg.coverage, size = n_muts_tail_2)
    
    nv_subcl = np.random.binomial(size = n_muts_sub, n = dp_subcl, p = peaks_of_mutations)
    
    tail_1 = my_pareto_betabin(n_muts_tail_1)
    tail_2 = my_pareto_betabin(n_muts_tail_2)
    
    tail_nv_1 = np.around(tail_1 * dp_tail_1[0:len(tail_1)], 0)
    tail_nv_2 = np.around(tail_2 * dp_tail_2[0:len(tail_2)], 0)
    
    return nv_subcl, dp_subcl, tail_nv_1, dp_tail_1[0:len(tail_1)], tail_nv_2, dp_tail_2[0:len(tail_2)]


# def simulate_SNVs_subclonal(seg):
#     nvaf = int(seg.SNV)
#     ccf = seg.ccf
#     purity = seg.purity
      
#     ks = f'{seg.Major_1}:{seg.minor_1}-{seg.Major_2}:{seg.minor_2}'
#     tsv = f'/Users/lucreziavaleriani/Desktop/MultiSegmenter/sub_clonal_peaks/{ks}/{ccf}-{purity}_peaks.tsv'
#     df = pd.read_csv(tsv, sep = '\t')
#     model = random.choice(list(set(df.model_id)))
#     peaks_df = df.loc[df['model_id'] == model]
#     peaks = list(peaks_df.peak)
    
#     mix_prop =   np.random.uniform(size = len(peaks))
#     mix_prop =  mix_prop / sum(mix_prop)
    
#     nv_subcl, dp_subcl, tail_nv_1, dp_tail_1, tail_nv_2, dp_tail_2 = vaf_subclonal(seg, mix_prop, peaks)
#     nv = np.concatenate([nv_subcl,tail_nv_1,tail_nv_2]) 
#     dp = np.concatenate([dp_subcl,dp_tail_1,dp_tail_2])
    
#     pos = np.random.randint(seg.start, seg.end, nv.shape[0])
    
#     pks = list(set(peaks))
#     df = pd.DataFrame({'segID':[seg.id for _ in range(nv.shape[0])],
#                        'pos':pos,
#                        'nv':nv,
#                        'coverage':dp,
#                        'vaf':np.divide(nv, dp), 
#                        'CN_1':[f'{int(seg.Major_1)}:{int(seg.minor_1)}' for _ in range(nv.shape[0])],
#                        'CN_2':[f'{int(seg.Major_2)}:{int(seg.minor_2)}' for _ in range(nv.shape[0])],
#                        'ccf':[ccf for _ in range(nv.shape[0])],
#                        'purity':[purity for _ in range(nv.shape[0])],
#                        'true_peaks':[pks for _ in range(nv.shape[0])],
#                        'model':[model for _ in range(nv.shape[0])]
#                        })
#     return df




def simulate_data(segs):
    """_summary_

    Parameters
    ----------
    segs : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
  
    SNV = pd.DataFrame()
    SNP = pd.DataFrame()
    for _, row in segs.iterrows():
        if row.type == 'clonal':
            snp = simulate_SNPs_clonal(row)
            snv = simulate_SNVs_clonal(row)
        
        else:
            snp = simulate_SNPs_subclonal(row)
            snv = simulate_SNVs_subclonal(row)
            
        SNP =  pd.concat([SNP,snp], axis=0)
        SNV =  pd.concat([SNV,snv], axis=0)
        
        SNP = SNP.reset_index(drop=True)
        SNV = SNV.reset_index(drop=True)
    return SNP, SNV
  
  
# res_path = '/Users/lucreziavaleriani/Desktop/simulations/subclonal_w_tail'
# for s in range(20):
#     print(s)
#     segs = simulate_segment(coverage = 100, w = 1, dt = 10)
#     SNP, SNV = simulate_data(segs)
#     print('\t', segs)
    
#     os.makedirs(f"{res_path}/sim_{s}/", exist_ok=True)
    
#     segs.to_csv(f"{res_path}/sim_{s}/{s}_segs.tsv", sep = '\t')
#     SNP.to_csv(f"{res_path}/sim_{s}/{s}_snp_sim.tsv", sep = "\t")
#     SNV.to_csv(f"{res_path}/sim_{s}/{s}_snv_sim.tsv", sep = "\t")
    
    
