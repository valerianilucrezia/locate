import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from numbers import Number



class SubClonalLikelihood(TorchDistribution):
    has_rsample = False

    def __init__(self,
                 x = None,
                 Major_c1 = None,
                 Major_c2 = None,
                 minor_c1 = None,
                 minor_c2 = None,
                 tot_c1 = None,
                 tot_c2 = None,
                 snp_dp = None,
                 scaling_factors = torch.tensor([1.,1.,1.,1.]),
                 purity = 1, 
                 ploidy = 2,
                 ccf = 0,
                 batch_shape = None,
                 validate_args = False):

        self.x = x
        self.Major_c1 = Major_c1,
        self.Major_c2 = Major_c2,
        self.minor_c1 = minor_c1,
        self.minor_c2 = minor_c2,
        self.tot_c1 = tot_c1,
        self.tot_c2 = tot_c2,
        self.scaling_factors = scaling_factors
        self.purity = purity
        self.ploidy = ploidy
        self.ccf = ccf,
        self.snp_dp = snp_dp
        self.validate_args = validate_args
        
        batch_shape = torch.Size(batch_shape)
        super(SubClonalLikelihood, self).__init__(batch_shape, validate_args=validate_args)


    def log_prob(self, inp):
        dr_lk = 0
        baf_lk = 0
        vaf_lk = 0
    
        # BAF
        num = min(self.Major_c1 * self.ccf + self.Major_c2 * (1 - self.ccf), self.minor_c1 * self.ccf + self.minor_c2 * (1 - self.ccf))
        den = purity * ((self.Major_c1 + self.minor_c1) * self.ccf + (1 - self.ccf) * (self.Major_c2 + self.minor_c2)) + 2 * (1 - self.purity)
        exp_baf = num / den
        alpha = ((self.snp_dp - 2) * exp_baf + 1) / (1 - exp_baf)
        
        baf_lk = dist.Beta(concentration1 = alpha, 
                            concentration0 = self.snp_dp).log_prob(
            inp["baf"]
            )
            
        # DR                  
        dr = (2*(1-self.purity) + self.purity*(self.ccf*(self.Major_c1 + self.Major_c2) + (1- self.ccf)*(self.minor_c1 + self.minor_c2))) / self.ploidy 
        dr_lk = dist.Gamma(dr * torch.sqrt(self.snp_dp) + 1, 
                                torch.sqrt(self.snp_dp)).log_prob(
            inp["dr"]
            )
        
                    
        tot_lk = self.scaling_factors[0] * baf_lk + self.scaling_factors[1] * dr_lk + self.scaling_factors[2] * vaf_lk 
        return(tot_lk)



def get_sub_clonal_peaks(tot, Major, minor, purity, ccf):
    # mult = []
    # for i,v in enumerate(Major):
    #     m = []
    #     if torch.equal(Major[i], minor[i]):
    #         m.append(Major[i][0])
    #     else:
    #         if minor[i] != 0:
    #             m.append(Major[i][0])
    #             m.append(minor[i][0])
    #         else:
    #             m.append(Major[i][0])
    #     if torch.equal(Major[i], torch.tensor([2])) and torch.equal(minor[i], torch.tensor([1])) == False:
    #         m.append(torch.tensor(1))
    #     mult.append(m)

    # clonal_peaks = []
    # for i,c in enumerate(mult):
    #     p = []
    #     for m in c:
    #         cp = m * purity / (tot[i] * purity + 2 * (1 - purity))
    #         p.append(cp)
    #     clonal_peaks.append(p)
    return
