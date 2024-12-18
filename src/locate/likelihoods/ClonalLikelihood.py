import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from numbers import Number
import numpy as np



class ClonalLikelihood(TorchDistribution):
    """_summary_

    Parameters
    ----------
    TorchDistribution : _type_
        _description_
    """

    def __init__(self,
                 x = None,
                 Major = None,
                 minor = None,
                 tot = None,
                 snp_dp = None,
                 dp = None,
                 scaling_factors = torch.tensor([1.,1.,1.,1.]),
                 purity = 1, 
                 ploidy = 2,
                 batch_shape = None,
                 validate_args = False,
                 has_baf = True,
                 has_dr = True):

        self.x = x
        self.Major = Major
        self.minor = minor
        self.tot = tot
        self.scaling_factors = scaling_factors
        self.purity = purity
        self.ploidy = ploidy
        self.snp_dp = snp_dp
        self.dp = dp
        self.validate_args = validate_args
        self.has_baf = has_baf
        self.has_dr = has_dr
        
        
        batch_shape = torch.Size(batch_shape)
        super(ClonalLikelihood, self).__init__(batch_shape, validate_args=validate_args)


    def log_prob(self, inp):
        """_summary_

        Parameters
        ----------
        inp : _type_
            _description_
        """
        
        dr_lk = 0
        baf_lk = 0
        vaf_lk = 0
    
        # BAF
        if self.has_baf:
            probs = get_baf_probs(self.tot[self.x], self.Major[self.x], self.minor[self.x], self.purity)
            tmp_baf_lk = []
            for cn in probs:
                thr = (sum(cn)/2).detach()
                mirr = 1.01

                b = inp["baf"]
                baf = torch.tensor(np.where(b >= thr, mirr - b, b))

                prob = min(cn)[0]
                alpha = ((self.snp_dp-2) * prob + 1) / (1 - prob)  
                
                baf_lk = dist.Beta(concentration1 = alpha, 
                                    concentration0 = self.snp_dp).log_prob(
                    baf
                    )
                tmp_baf_lk.append(baf_lk) 
            baf_lk = torch.cat(tmp_baf_lk, dim=1)
            

        # DR         
        if self.has_dr:        
            dr = ((2 * (1-self.purity)) + (self.purity * (self.Major[self.x] + self.minor[self.x]))) / (2*(1-self.purity) + (self.purity * self.ploidy))
            dr_lk = dist.Gamma(dr * torch.sqrt(self.snp_dp) + 1, 
                                    torch.sqrt(self.snp_dp)).log_prob(
                inp["dr"]
                )
        
        # VAF
        if self.dp != None and self.vaf != None:
            clonal_peaks = get_clonal_peaks(self.tot[self.x], self.Major[self.x], self.minor[self.x], self.purity)
            
            tmp_vaf_lk = []
            for cn in clonal_peaks:
                thr = (sum(cn)/2).detach()
                mirr = sum(cn).detach()

                vv = inp["vaf"]/inp["dp"]
                vaf = torch.tensor(np.where(vv <= thr, mirr - vv, vv))
                vaf = vaf*inp["dp"]
                vaf = vaf.to(torch.int64)
                vaf[vaf>inp["dp"]] = inp["dp"][vaf>inp["dp"]]
                
                p = max(cn)
                bin_lk = dist.Binomial(total_count = inp["dp"], 
                                        probs = p).log_prob(
                    vaf
                )
                tmp_vaf_lk.append(bin_lk) 
            vaf_lk = torch.cat(tmp_vaf_lk, dim=1)

        tot_lk = self.scaling_factors[0] * baf_lk + self.scaling_factors[1] * dr_lk + self.scaling_factors[2] * vaf_lk 
        return(tot_lk)


def get_baf_probs(tot, Major, minor, purity):
    """_summary_

    Parameters
    ----------
    tot : _type_
        _description_
    Major : _type_
        _description_
    minor : _type_
        _description_
    purity : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    prob_baf = []
    for m,M in zip(minor, Major):
        if m == M:
            num = (purity * m) +  (1 - purity)
            den = (purity * (m + M)) + (2 * (1 - purity))
            prob = num / den
            prob_baf.append([prob])
        else:
            num = (purity * m) +  (1 - purity)
            den = (purity * (M + m)) + (2 * (1 - purity))
            prob_minor = num / den
            prob_Major = 1-prob_minor
            prob_baf.append([prob_minor, prob_Major])
    return prob_baf
            



def get_clonal_peaks(tot, Major, minor, purity):
    """_summary_

    Parameters
    ----------
    tot : _type_
        _description_
    Major : _type_
        _description_
    minor : _type_
        _description_
    purity : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    mult = []
    for i,v in enumerate(Major):
        m = []
        if torch.equal(Major[i], minor[i]):
            m.append(Major[i][0])
        else:
            if minor[i] != 0:
                m.append(Major[i][0])
                m.append(minor[i][0])
            else:
                m.append(Major[i][0])
        if torch.equal(Major[i], torch.tensor([2])) and torch.equal(minor[i], torch.tensor([1])) == False:
            m.append(torch.tensor(1))
        mult.append(m)

    clonal_peaks = []
    for i,c in enumerate(mult):
        p = []
        for m in c:
            cp = m * purity / (tot[i] * purity + 2 * (1 - purity))
            p.append(cp)
        clonal_peaks.append(p)
        
    return clonal_peaks
