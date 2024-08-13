import itertools

import torch
import torch.nn as nn
from torch.distributions import constraints

from locate.models.Model import Model
from locate.likelihoods import ClonalLikelihood

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import infer_discrete, config_enumerate
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings


# Here I created a new class that inherits from dict, it ha an unsqueeze method that unsqueezes all the tensors in the dict 
# It is actually more of a dictionary generator than a proper dictionary

class SqueezableDict(dict):
    
    def __init__(self, *args, **kwargs):
        super(SqueezableDict, self).__init__(*args, **kwargs)
        self.data = [v for v in self.values() if v is not None ][0]
        self.dim = [v for v in self.values() if v is not None ][0].dim
        self.shape = [v for v in self.values() if v is not None ][0].shape
    
    def unsqueeze(self, dim):
        return {k:v.unsqueeze(dim) for k,v in self.items() if v is not None }
    


class Clonal(Model):
    # parameters of the HMM + other stuff specific to CNA calling

    params = {'jumping_prob' : 1e-2,                                              
              'init_probs': torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]), 
              'hidden_dim': 3, 
              "CUDA" : False, 
              "prior_ploidy" : 2, 
              "prior_purity" : 0.9, 
              "fix_purity" : True, 
              "fix_ploidy" : True, 
              "scaling_factors" : torch.tensor([1.,1.,1.,1.]),
              "allele_specific": True}
    
    # This has to be generalized and left empty 
    data_name = set(['baf', 'dr', 'dp_snp', 'vaf', 'dp'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        self._name = "Clonal"
        super().__init__(data_dict, self.data_name)
        
        # internal structure is a SqueezableDict
        self._data = SqueezableDict({k:v for k,v in self._data.items()})
            
    def model(self, i = 1,  *args, **kwargs):
        n_sequences, length = 0, 0
        tot = 0
        
        if self._params["allele_specific"]:
            Major, minor, tot, x = self.get_Major_minor()
            
            self._params["Major"] = Major
            self._params["minor"] = minor
            probs_x = pyro.sample(
                "probs_x",
                dist.Dirichlet((1 - self._params["jumping_prob"]) * torch.eye(x.shape[0]) + self._params["jumping_prob"]).to_event(1),
            )
            
        else:
            probs_x = pyro.sample(
                "probs_x",
                dist.Dirichlet((1 - self._params["jumping_prob"]) * torch.eye(self._params["hidden_dim"]) + self._params["jumping_prob"]).to_event(1),
            )
            x = torch.arange(0, self._params["hidden_dim"]).long()
            tot = x + 1
            minor, Major = None, None
            
        if self._params["fix_purity"]:
            purity = torch.tensor(self._params["prior_purity"])
        else:
            #purity = pyro.sample("purity", dist.Uniform(0.,1.))
            purity = pyro.sample("purity", dist.Beta(4, 2))
            
        if self._data["baf"] is not None:
            has_baf = True
        else:
            has_baf = False
            
        if self._data["dr"] is not None:
            has_dr = True
        else:
            has_dr = False
                    
        if self._params["fix_ploidy"]:
            ploidy = self._params["prior_ploidy"]
        else:
            ploidy = int(pyro.sample("ploidy", dist.Poisson(2)))
            
        if self._data["dp_snp"] != None:
            mean_cov = torch.mean(self._data["dp_snp"].float())
        else:
            mean_cov = None
            
        if self._data["dp"] != None:
            dp = self._data["dp"]
        else:
            dp = None
        
        measures = [i for i in list(self._data.keys()) if self._data[i] != None]
        length, n_sequences  = self._data[measures[0]].shape
        
        
        with pyro.plate("sequences", n_sequences):
            init_logits = self._params["init_probs"].log()
            trans_logits = probs_x.log()
            
            with ignore_jit_warnings():
                obs_dist = ClonalLikelihood(
                 x = x.unsqueeze(-1),
                 Major = Major,
                 minor = minor,
                 tot = tot,
                 snp_dp = mean_cov,
                 dp = dp, 
                 scaling_factors = self._params["scaling_factors"],
                 purity = purity, 
                 ploidy = ploidy,
                 batch_shape = [x.shape[0], length],
                 has_baf = has_baf,
                 has_dr = has_dr
                ).to_event(1)

                hmm_dist = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
            pyro.sample("y", hmm_dist, obs=self._data)
        
    # Autoguide
    def guide(self, *args, **kwargs):
        return AutoDelta(poutine.block(self.model, hide_fn=lambda msg: msg["name"].startswith("x")))

    def get_Major_minor(self):
        combinations = list(itertools.combinations_with_replacement(range( self._params["hidden_dim"]), 2))[1:]
        
        major_alleles = [max(combination) for combination in combinations]
        minor_alleles = [min(combination) for combination in combinations]

        major_allele_tensor = torch.tensor(major_alleles).long()
        minor_allele_tensor = torch.tensor(minor_alleles).long()
        x = torch.tensor(list(range(len(major_alleles)))).long()
        tot = major_allele_tensor + minor_allele_tensor

        return major_allele_tensor, minor_allele_tensor, tot,  x
    
    
    # Model 2 is the same as model 1 but with enumeration, 
    # is it used to get MAP estimates of the states
    @infer_discrete(first_available_dim = -2, temperature=0)
    @config_enumerate
    def model_2(self, learned_params):
        
        n_sequences, length = 0, 0
        minor, Major = None, None
        x = torch.arange(1., self._params["hidden_dim"] + 1)
        tot = x
        
        if self._params["allele_specific"]:
            Major, minor, tot, x = self.get_Major_minor()
            
        probs_x = learned_params['probs_x']
            
        measures = [i for i in list(self._data.keys()) if self._data[i] != None]
        length, n_sequences  = self._data[measures[0]].shape

        if self._params["fix_purity"]:
            purity = self._params["prior_purity"]
        else:
            purity = learned_params['purity']
        
        
        if self._params["fix_ploidy"]:
            ploidy = self._params["prior_ploidy"]
        else:
            ploidy = learned_params['ploidy']
            
                
        with pyro.plate("sequences", n_sequences, dim = -1):
            x = [0]
            for t in pyro.markov(range(length)):

                x_new = pyro.sample(
                "x_{}".format(t),
                dist.Categorical(probs_x[x[t]])
                )
                x.append(x_new)
                
                if self._data["baf"] is not None:
                    num = (purity * minor[x_new]) +  (1 - purity)
                    den = (purity * (Major[x_new] + minor[x_new])) + (2 * (1 - purity))
                    prob = num / den
                    
                    alpha = ((self._data["dp_snp"][t,:]-2) * prob + 1) / (1 - prob)
                    baf_lk = pyro.factor("y_baf_{}".format(t), 
                                         dist.Beta(concentration1 = alpha, 
                                                   concentration0 = self._data["dp_snp"][t,:]).log_prob(
                                                       self._data["baf"][t,:]
                                                       ))
                    
                                           
                if self._data["dr"] is not None:     
                    dr = ((2 * (1-purity)) + (purity * (Major[x_new] + minor[x_new]))) / ploidy
                    dr_lk = pyro.factor("y_dr_{}".format(t), 
                                        dist.Gamma(dr * torch.sqrt(self._data["dp_snp"][t,:]) + 1, 
                                                   torch.sqrt(self._data["dp_snp"][t,:])).log_prob(
                                                       self._data["dr"][t,:]
                                                       ))
                
                
                if self._data["vaf"] is not None:
                    clonal_peaks = get_clonal_peaks(tot[x_new], Major[x_new], minor[x_new], purity)
                    
                    tmp_vaf_lk = []
                    for j,cn in enumerate(clonal_peaks):
                        tmp_peak = 0.0
                        for i,p in enumerate(cn):
                            bin_lk = pyro.factor(f"y_vaf__{t}_{i}_{j}", dist.Binomial(self._data["dp"][t,:], 
                                    p).log_prob(self._data["vaf"][t,:].to(torch.int64)))
                    
            return x
        
        
def get_clonal_peaks(tot, Major, minor, purity):
    if Major.dim() == 0:
        Major = Major.unsqueeze(0).unsqueeze(1)
        minor = minor.unsqueeze(0).unsqueeze(1)
        tot = tot.unsqueeze(0).unsqueeze(1)
    
    
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
