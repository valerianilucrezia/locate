import itertools
import torch

from locate.models.Model import Model
from locate.likelihoods import ClonalLikelihood
from locate.priors.ploidy import PloidyPrior
from locate.priors.purity import PurityPrior

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import infer_discrete, config_enumerate
from pyro.infer.autoguide import AutoDelta, init_to_sample
from pyro.util import ignore_jit_warnings

from torch.nn.functional import softmax


class SqueezableDict(dict):
    """_summary_

    Parameters
    ----------
    dict : _type_
        _description_
    """
    
    def __init__(self, *args, **kwargs):
        super(SqueezableDict, self).__init__(*args, **kwargs)
        self.data = [v for v in self.values() if v is not None ][0]
        self.dim = [v for v in self.values() if v is not None ][0].dim
        self.shape = [v for v in self.values() if v is not None ][0].shape
    
    def unsqueeze(self, dim):
        return {k:v.unsqueeze(dim) for k,v in self.items() if v is not None }
    


class Clonal(Model):
    """_summary_

    Parameters
    ----------
    Model : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    params = {'jumping_prob' : 1e-2,                                              
            'init_probs': torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 
            'hidden_dim': 3, 
            "CUDA" : False, 
            "prior_ploidy" : 2, 
            "prior_purity" : 0.9, 
            "fix_purity" : True, 
            "fix_ploidy" : True, 
            "scaling_factors" : torch.tensor([1.,1.,1.]),
            "allele_specific": True,
            "prior_bp": None,
            "lambda_cn": 1.0,          # ΔCN penalty slope used to shape the prior
            "alpha_conc": 50.0,        # global Dirichlet concentration (higher = stronger prior)
            "alpha_self_boost": 3.0,   # extra mass on the diagonal
            "alpha_dip_boost": 0.5,    # extra mass on diploid column (if present)
            "bp_strength": 3.0, 
            "sample_type": "cell_line"}
    
    # This has to be generalized and left empty 
    data_name = set(['baf', 'dr', 'dp_snp', 'vaf', 'dp'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        self._name = "Clonal"
        super().__init__(data_dict, self.data_name)
        
        # internal structure is a SqueezableDict
        self._data = SqueezableDict({k:v for k,v in self._data.items()})
        

    def _diploid_index(self, Major, minor):
        if Major is None or minor is None:
            return None
        is_dip = (Major == 1) & (minor == 1)
        idx = torch.nonzero(is_dip, as_tuple=False)
        return int(idx[0,0]) if idx.numel() > 0 else None

    def _dirichlet_alpha_from_kernel(self, tot, dip_idx, K):
        """Return α (K x K) > 0 used as Dirichlet concentration for each row."""
        lam = float(self._params["lambda_cn"])
        conc0 = float(self._params["alpha_conc"])
        self_boost = float(self._params["alpha_self_boost"])
        dip_boost  = float(self._params["alpha_dip_boost"])

        tot = tot.float()                               # (K,)
        dcn = torch.cdist(tot.unsqueeze(1), tot.unsqueeze(1), p=1).squeeze(-1)  # (K,K)
        kernel = torch.exp(-lam * dcn)                  # penalize big ΔCN
        kernel = kernel + torch.eye(K) * self_boost     # sticky diagonal

        if dip_idx is not None and dip_boost > 0:
            kernel[:, dip_idx] = kernel[:, dip_idx] + dip_boost

        # scale to desired total concentration per row
        alpha = conc0 * (kernel / kernel.sum(-1, keepdim=True))
        # strictly positive
        return torch.clamp(alpha, min=1e-6)
    
    def _apply_breakpoint_prior(self, base_logits, prior_bp, strength):
        """
        base_logits: (K,K); prior_bp: (T-1,)
        Return (T-1,K,K) logits that encourage switches at high prior_bp[t].
        """
        K = base_logits.size(0)
        eye = torch.eye(K, device=base_logits.device)
        off = (1 - eye) / (K - 1)
        # mask with zero row-sum so overall scale stays reasonable:
        M = off - eye
        Tm1 = prior_bp.numel()
        L = base_logits.unsqueeze(0).expand(Tm1, K, K).clone()
        for t in range(Tm1):
            L[t] = L[t] + float(prior_bp[t]) * strength * M
        return L
            
    def model(self, i = 1,  *args, **kwargs):
        """_summary_

        Parameters
        ----------
        i : int, optional
            _description_, by default 1
        """
        
        n_sequences, length = 0, 0
        tot = 0
        
        if self._params["allele_specific"]:
            Major, minor, tot, x = self.get_Major_minor()
            self._params["Major"] = Major
            self._params["minor"] = minor
            K = x.shape[0]
            # probs_x = pyro.sample(
            #     "probs_x",
            #     dist.Dirichlet((1 - self._params["jumping_prob"]) * torch.eye(x.shape[0]) + self._params["jumping_prob"]).to_event(1),
            # )
            
        else:
            # probs_x = pyro.sample(
            #     "probs_x",
            #     dist.Dirichlet((1 - self._params["jumping_prob"]) * torch.eye(self._params["hidden_dim"]) + self._params["jumping_prob"]).to_event(1),
            # )
            x = torch.arange(0, self._params["hidden_dim"]).long()
            tot = x + 1
            minor, Major = None, None
            #K = self._params["hidden_dim"]
            K = x.shape[0]
            
        has_baf = self._data["baf"] is not None
        has_dr = self._data["dr"] is not None
        dp = self._data.get('dp', None)

        purity = self._params["prior_purity"] if self._params["fix_purity"] else PurityPrior(self._params["sample_type"]).sample() 
        ploidy = self._params["prior_ploidy"] if self._params["fix_ploidy"] else PloidyPrior(self._params["sample_type"]).sample()
        mean_cov = torch.mean(self._data["dp_snp"].float()) if self._data["dp_snp"] is not None else None
        
        measures = [i for i in list(self._data.keys()) if self._data[i] is not None]
        length, n_sequences  = self._data[measures[0]].shape
        
        # --- transitions: learn base via Dirichlet; modulate if prior_bp provided ---
        dip_idx = self._diploid_index(Major, minor) if Major is not None else None
        alpha = self._dirichlet_alpha_from_kernel(tot, dip_idx, K)            # (K,K)
        probs_x = pyro.sample("probs_x", dist.Dirichlet(alpha).to_event(1))               # (K,K), each row simplex
        base_logits = probs_x.log()                                            # (K,K)

        prior_bp = self._params.get("prior_bp", None)
        if prior_bp is not None:
            #assert prior_bp.numel() == (length - 1), "prior_bp must have length T-1"
            trans_logits = self._apply_breakpoint_prior(
                base_logits,
                prior_bp=prior_bp,
                strength=float(self._params["bp_strength"]),
            )  # (T-1,K,K)
        else:
            trans_logits = base_logits  # stationary
        init_logits = self._params["init_probs"].log()
        
        with pyro.plate("sequences", n_sequences):
            #init_logits = self._params["init_probs"].log()
            #trans_logits = probs_x.log()
            
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
                batch_shape = [K, length],
                has_baf = has_baf,
                has_dr = has_dr
                ).to_event(1)

                hmm_dist = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
            pyro.sample("y", hmm_dist, obs=self._data)
        
    # Autoguide
    def guide(self, *args, **kwargs):
        return AutoDelta(model = poutine.block(
                                self.model, 
                                hide=['mixture_idx', 'mixture_idx_purity']
                                ), 
                        init_loc_fn = self.my_init_fn)

    def my_init_fn(self, site):
        if site["name"] == "probs_x":
            # mean of Dirichlet(alpha) is alpha / alpha.sum(-1, keepdim=True)
            Major, minor, tot, x = self.get_Major_minor() if self._params["allele_specific"] \
                                else (None, None, torch.arange(0, self._params["hidden_dim"]) + 1, None)
            K = tot.shape[0]
            dip_idx = self._diploid_index(Major, minor) if Major is not None else None
            alpha = self._dirichlet_alpha_from_kernel(tot, dip_idx, K)
            return (alpha / alpha.sum(-1, keepdim=True)).detach()
        return init_to_sample(site)

    def get_Major_minor(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        
        combinations = list(itertools.combinations_with_replacement(range(self._params["hidden_dim"]), 2))[1:]
        
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
        """_summary_

        Parameters
        ----------
        learned_params : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        
        n_sequences, length = 0, 0
        minor, Major = None, None
        x = torch.arange(1., self._params["hidden_dim"] + 1)
        tot = x

        if self._params["allele_specific"]:
            Major, minor, tot, x = self.get_Major_minor()
            
        #probs_x = learned_params['probs_x']
        purity = self._params["prior_purity"] if self._params["fix_purity"] else learned_params['purity']
        ploidy = self._params["prior_ploidy"] if self._params["fix_ploidy"] else learned_params['ploidy']
            
        measures = [i for i in list(self._data.keys()) if self._data[i] != None]
        length, n_sequences  = self._data[measures[0]].shape
        K = x.shape[0]
        
        # learned transitions
        base_logits = learned_params['probs_x'].log()   # (K,K)
        prior_bp = self._params.get("prior_bp", None)
        if prior_bp is not None:
            trans_logits_T = self._apply_breakpoint_prior(
                base_logits, prior_bp, float(self._params["bp_strength"])
            )  # (T-1,K,K)
        else:
            trans_logits_T = None
            base_probs = softmax(base_logits, dim=-1)
        
        with pyro.plate("sequences", n_sequences, dim = -1):
            x = [0]
            
            for t in pyro.markov(range(length)):
                if trans_logits_T is None:
                    p_row = base_probs[x[t]]
                else:
                    p_row = softmax(trans_logits_T[t-1, x[t]] if t > 0 else base_logits[x[t]], dim=-1)
                x_new = pyro.sample(f"x_{t}", dist.Categorical(p_row))
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
                    dr = ((2 * (1-purity)) + (purity * (Major[x_new] + minor[x_new]))) / (2*(1-purity) + (purity * ploidy))
                    dr_lk = pyro.factor("y_dr_{}".format(t), 
                                        dist.Gamma(dr * torch.sqrt(self._data["dp_snp"][t,:]) + 1, torch.sqrt(self._data["dp_snp"][t,:])).log_prob(
                                            self._data["dr"][t,:]))
                
                
                if "vaf" in self._data.keys() and self._data['vaf'] is not None:
                    # i could add here pareto stuff
                    clonal_peaks = get_clonal_peaks(tot[x_new], Major[x_new], minor[x_new], purity)
                    
                    #tmp_vaf_lk = []
                    for j,cn in enumerate(clonal_peaks):
                        for i,p in enumerate(cn):
                            bin_lk = pyro.factor(f"y_vaf_{t}_{i}_{j}", 
                                                dist.Binomial(self._data["dp"][t,:], p).log_prob(
                                                    self._data["vaf"][t,:].to(torch.int64)))
                    
            return x

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


