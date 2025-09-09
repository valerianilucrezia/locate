import pyro
import torch

from pyro.infer import SVI, infer_discrete
from pyro import poutine

from tqdm import trange

from locate.likelihoods import ClonalLikelihood
from locate.utils import retrieve_params
from locate.stopping_criteria import all_stopping_criteria
from locate.likelihoods.utils_likelihood import export_switch


from pyro.infer.autoguide import AutoDelta, AutoNormal, init_to_sample
from pyro.infer import SVI, Trace_ELBO, Predictive


class LOCATE:
    """_summary_
    """
    
    def __init__(self, model = None, optimizer = None, loss = None, inf_type = SVI, CUDA = False):
        self._model_fun = model
        self._optimizer = optimizer
        self._loss = loss
        self._model = None
        self._inf_type = inf_type
        self._model_trained = None
        self._guide_trained = None
        self._loss_trained = None
        self._model_string = None
        self._CUDA = CUDA
        
        if self._CUDA:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.params_history = {}


    def __repr__(self):
        if self._model is None:
            dictionary = {"Model": self._model_fun,
                    "Optimizer": self._optimizer,
                    "Loss": self._loss,
                    "Inference strategy": self._inf_type
                    }
        else:
            dictionary = {"Model" : self._model_fun,
                    "Data" : self._model._data,
                    "Model params": self._model._params,
                    "Optimizer" :self._optimizer,
                    "Loss" : self._loss,
                    "Inference strategy" :  self._inf_type
                    }

        return "\n".join("{}:\t{}".format(k, v) for k, v in dictionary.items())

    def initialize_model(self, data):
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_
        """
        
        assert self._model_fun is not None
        if self._CUDA:
            data = {k: v.cuda() for k, v in data.items() if torch.is_tensor(v)}
        
        self._model = self._model_fun(data)
        self._model_string = type(self._model).__name__

    def set_optimizer(self, optimizer):
        """_summary_

        Parameters
        ----------
        optimizer : _type_
            _description_
        """
        
        self._optimizer = optimizer

    def set_model(self, model):
        """_summary_

        Parameters
        ----------
        model : _type_
            _description_
        """
        
        self._model_fun = model

    def set_loss(self, loss):
        """_summary_

        Parameters
        ----------
        loss : _type_
            _description_
        """
        
        self._loss = loss

    def set_model_params(self, param_dict):
        """_summary_

        Parameters
        ----------
        param_dict : _type_
            _description_
        """
        
        if self._CUDA:
            param_dict['CUDA'] = True
            param_dict = {k: v.cuda() if torch.is_tensor(v) else v for k, v in param_dict.items()}

        self._model.set_params(param_dict)

    def run(self, steps, 
            param_optimizer = {'lr' : 0.05}, 
            e = 0.01, 
            patience = 5, 
            param_loss = None, 
            seed = 3, 
            callback=None,
            guide_kind: str = "delta"):
        """ This function runs the inference of non-categorical parameters

          This function performs a complete inference cycle for the given tuple(model, optimizer, loss, inference modality).
          For more info about the parameters for the loss and the optimizer look at `Optimization <http://docs.pyro.ai/en/stable/optimization.html>`_.
          and `Loss <http://docs.pyro.ai/en/stable/inference_algos.html>`_.

          Not all the the combinations Optimize-parameters and Loss-parameters have been tested, so something may
          not work (please open an issue on the GitHub page).


          Args:
              steps (int): Number of steps
              param_optimizer (dict):  A dictionary of paramaters:value for the optimizer
              param_loss (dict): A dictionary of paramaters:value for the loss function
              seed (int): seed to be passed to  pyro.set_rng_seed
              MAP (bool): Perform learn a Delta distribution over the outer layer of the model graph
              verbose(bool): show loss for each step, if false the functions just prints a progress bar
              BAF(torch.tensor): if provided use BAF penalization in the loss

          Returns:
              list: loss (divided by sample size) for each step
          """

        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        model = self._model.model
        
        # NEW: choose guide
        if guide_kind == "normal":
            # Use a flexible guide to get posterior uncertainty
            guide = AutoNormal(model = poutine.block(model,  hide=['mixture_idx', 'mixture_idx_purity']), init_loc_fn=getattr(self._model, "my_init_fn", init_to_sample))
        else:
            # default: your existing AutoDelta guide
            guide = self._model.guide(None)

        optim = self._optimizer(param_optimizer)
        elbo = self._loss(**param_loss) if param_loss is not None else self._loss()
        svi = self._inf_type(model, guide, optim, elbo)

        num_observations = 0
        num_segments = 0
        measures = [i for i in list(self._model._data.keys()) if self._model._data[i] != None]

        num_observations += self._model._data[measures[0]].shape[1]
        num_segments += self._model._data[measures[0]].shape[0] 

        loss = [None] * steps

        #loss[0] = svi.step(i = 1) / (num_observations * num_segments)
        loss[0] = svi.step(i = 1)
        elb = loss[0]
        new_w = retrieve_params()

        t = trange(steps, desc='Bar desc', leave=True)

        conv = 0
        for step in t:
            t.set_description('ELBO: {:.9f}  '.format(elb))
            t.refresh()

            #loss[step] = svi.step(i = step + 1) / (num_observations * num_segments)
            loss[step] = svi.step(i = step + 1)
            elb = loss[step]

            old_w, new_w = new_w, retrieve_params()
            
            if callback is not None:
                callback(model, guide, step)
            

            stop, diff_params = all_stopping_criteria(old_w, new_w, e, step)
            self.params_history.update({step : diff_params})

            if stop:
                if (conv == patience):
                    print('Reached convergence', flush = True)
                    break
                else:
                    conv = conv + 1
            else:
                conv = 0
                
        self._model_trained = model
        self._guide_trained = guide
        self._loss_trained = loss
        return loss, num_observations

    def learned_parameters_Clonal(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        params = self._guide_trained()
        map_states =  self._model.model_2(self._model, learned_params = params)
        
        if self._model._params["allele_specific"]:
            discrete_params = {"CN_Major" : self._model._params["Major"][torch.tensor(map_states)[1:].long()], 
                            "CN_minor" : self._model._params["minor"][torch.tensor(map_states)[1:].long()]}
        else:
            discrete_params = {"CN_tot" : torch.tensor(map_states)[1:]}
            
        if self._CUDA:
            trained_params_dict = {i : params[i].cpu().detach().numpy() for i in params}
            discrete_params = {i : discrete_params[i].cpu().detach().numpy() for i in discrete_params}
        else:
            trained_params_dict = {i : params[i].detach().numpy() for i in params}
            discrete_params = {i : discrete_params[i].detach().numpy() for i in discrete_params}

        all_params =  {**trained_params_dict, **discrete_params}
        return all_params



    def learned_parameters_SubClonal(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        params = self._guide_trained()
        map_states =  self._model.model_2(self._model, learned_params = params)
        
        if self._model._params["allele_specific"]:
            discrete_params = {"CN_Major" : self._model._params["Major"][torch.tensor(map_states)[1:].long()], 
                            "CN_minor" : self._model._params["minor"][torch.tensor(map_states)[1:].long()]}
        else:
            discrete_params = {"CN_tot" : torch.tensor(map_states)[1:]}
            
        if self._CUDA:
            trained_params_dict = {i : params[i].cpu().detach().numpy() for i in params}
            discrete_params = {i : discrete_params[i].cpu().detach().numpy() for i in discrete_params}
        else:
            trained_params_dict = {i : params[i].detach().numpy() for i in params}
            discrete_params = {i : discrete_params[i].detach().numpy() for i in discrete_params}

        all_params =  {**trained_params_dict,**discrete_params}

        return all_params
    
    def posterior_draws(self, num_samples=500, seed=3, sites=None):
        """
        Draw posterior samples from the trained guide for selected sites.
        Default sites capture purity, ploidy, and transitions (probs_x).
        Returns a dict of tensors keyed by site name; leading dim = num_samples.
        """
        assert self._model_trained is not None and self._guide_trained is not None, \
            "Run .run(...) first."

        pyro.set_rng_seed(seed)
        if sites is None:
            # Adjust to the exact names your model samples.
            sites = ["purity", "ploidy", "probs_x"]

        predictive = Predictive(self._model_trained,
                                guide=self._guide_trained,
                                num_samples=num_samples,
                                return_sites=sites)
        draws = predictive(i=1)  # keep your model's signature
        # detach & move to cpu
        for k in list(draws.keys()):
            if torch.is_tensor(draws[k]):
                draws[k] = draws[k].detach().cpu()
        return draws
    
    def cn_posterior_from_draws(self, draws):
        """
        Build per-locus posterior over hidden states and total CN from posterior parameter draws.
        Compatible with model_2 returning:
        - list (optionally length T+1 with a dummy first element),
        - tensor (T,),
        - tensor (T, N).

        Returns:
        state_probs: (T, K)  tensor with P(z_t = k | data)
        totcn_probs: (T, Cmax+1) tensor with P(TotalCN = c | data)
        state_map:   (T,)   consensus path from the median draw (majority over sequences if N>1)
        """
        import torch

        # --- shapes from data ---
        measures = [k for k, v in self._model._data.items() if v is not None]
        T, N = self._model._data[measures[0]].shape  # N may be 1

        # --- state space & CN mapping ---
        if self._model._params["allele_specific"]:
            Major = self._model._params["Major"].cpu()
            minor = self._model._params["minor"].cpu()
            K = Major.shape[0]
            tot_vec = (Major + minor).cpu()  # (K,)
        else:
            K = self._model._params["hidden_dim"]
            tot_vec = (torch.arange(0, K) + 1).cpu()  # (K,)

        # --- containers for posteriors aggregated across draws (and sequences) ---
        state_counts = torch.zeros(T, K, dtype=torch.float32)
        max_tot = int(tot_vec.max().item())
        totcn_counts = torch.zeros(T, max_tot + 1, dtype=torch.float32)

        # --- figure out #draws S from any available site ---
        probs_samples = draws.get("probs_x", None)
        purity_samples = draws.get("purity", None)
        ploidy_samples = draws.get("ploidy", None)
        S = None
        for s in (probs_samples, purity_samples, ploidy_samples):
            if torch.is_tensor(s):
                S = s.shape[0]
                break
        if S is None:
            S = 1  # degenerate case (e.g., AutoDelta)

        median_idx = S // 2
        state_map = None  # will set from the median draw

        # --- helper to normalize model_2 output to shape (T, N_current) ---
        def normalize_path(path):
            """
            Convert model_2 output to LongTensor of shape (T, N').
            Accepts list/tuple or tensor; trims dummy first element if length T+1.
            """
            if isinstance(path, (list, tuple)):
                # Elements might be tensors or ints; stack or tensor-ize
                if isinstance(path[0], torch.Tensor):
                    path = torch.stack(path, dim=0)
                else:
                    path = torch.tensor(path)
            # Now path is a tensor
            path = path.detach().cpu().long()
            # If there is a dummy at the front (common in earlier code): length T+1 and first entry 0
            if path.dim() == 1:
                if path.numel() == T + 1:
                    path = path[1:]
                # shape (T,) -> make (T,1)
                path = path.view(T, 1)
            elif path.dim() == 2:
                if path.size(0) == T + 1:
                    path = path[1:, :]
                assert path.size(0) == T, f"Expected T={T}, got {path.size(0)}"
            else:
                raise ValueError(f"Unexpected model_2 path dim: {path.dim()}")
            return path  # (T, N')

        # --- iterate posterior draws ---
        for s in range(S):
            learned_params = {}
            if probs_samples is not None:
                learned_params["probs_x"] = probs_samples[s]
            if purity_samples is not None and not self._model._params["fix_purity"]:
                learned_params["purity"] = purity_samples[s]
            if ploidy_samples is not None and not self._model._params["fix_ploidy"]:
                learned_params["ploidy"] = ploidy_samples[s]

            with torch.no_grad():
                path_out = self._model.model_2(self._model, learned_params=learned_params)

            path = normalize_path(path_out)  # (T, N')
            Np = path.size(1)

            # accumulate per-locus counts across sequences for this draw
            # use bincount per time to be robust
            for t in range(T):
                counts_t = torch.bincount(path[t], minlength=K).to(torch.float32)  # (K,)
                state_counts[t] += counts_t
                # map to total CN
                tot_t = tot_vec[path[t]]  # (N',)
                tot_counts_t = torch.bincount(tot_t, minlength=max_tot + 1).to(torch.float32)  # (Cmax+1,)
                totcn_counts[t] += tot_counts_t

            # store a consensus path for the median draw (majority across sequences)
            if s == median_idx:
                state_map = torch.mode(path, dim=1).values  # (T,)

        # normalize to probabilities (avoid divide-by-zero)
        state_probs = state_counts / state_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        totcn_probs = totcn_counts / totcn_counts.sum(dim=1, keepdim=True).clamp_min(1.0)

        return state_probs, totcn_probs, state_map







