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
        
        self._device = torch.device('cuda' if CUDA else 'cpu')
        if CUDA:
            torch.set_default_device('cuda')
        else:
            torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float32)  
 
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
        
        # Add CUDA information to data
        if 'params' not in data:
            data['params'] = {}
        data['params']['CUDA'] = self._CUDA
        
        # Initialize model
        self._model = self._model_fun(data)
        self._model_string = type(self._model).__name__
        
        # Ensure device consistency
        if self._model.device != self._device:
            print(f"Warning: Model device {self._model.device} differs from LOCATE device {self._device}")
            self._device = self._model.device

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
        """Set model parameters with proper parameter propagation.
        
        Args:
            param_dict: Dictionary of parameters to update
        """
        assert self._model is not None, "Model must be initialized before setting parameters"
        
        # Update model parameters
        if hasattr(self._model, 'set_params'):
            self._model.set_params(param_dict)
        else:
            raise AttributeError(f"Model {type(self._model)} doesn't have set_params method")

    def run(self, steps, 
            param_optimizer = {'lr' : 0.05}, 
            e = 0.01, 
            patience = 5, 
            param_loss = None, 
            seed = 3, 
            callback=None,
            guide_kind = "delta"):
        
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

    
        # Setup
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        
        # Pre-compute model and data dimensions
        measures = [k for k, v in self._model._data.items() if v is not None]
        first_measure = self._model._data[measures[0]]
        num_observations = first_measure.shape[1]
        num_segments = first_measure.shape[0]
        
        # Setup model and guide
        model = self._model.model
        if guide_kind == "normal":
            guide = AutoNormal(
                model=poutine.block(model, hide=['mixture_idx', 'mixture_idx_purity', 'mixture_idx_ploidy']),
                init_loc_fn=getattr(self._model, "my_init_fn", init_to_sample)
            ).to(self._device)
        else:
            guide = self._model.guide(None)
            if hasattr(guide, 'to'):
                guide = guide.to(self._device)
                
        # Setup optimizer and loss
        optim = self._optimizer(param_optimizer)
        elbo = self._loss(**param_loss) if param_loss is not None else self._loss()
        svi = self._inf_type(model, guide, optim, elbo)
        
        # Pre-allocate loss history
        loss = torch.zeros(steps, device=self._device)
        
        # Initial step
        loss[0] = svi.step(i=1)
        elb = loss[0].item()
        new_w = retrieve_params()
        
        # Progress bar
        t = trange(steps, desc='ELBO: {:.9f}'.format(elb), leave=True)
        
        # Optimization loop with early stopping
        conv = 0
        for step in t:
            # Update progress bar
            t.set_description('ELBO: {:.9f}'.format(elb))
            
            # Optimization step
            loss[step] = svi.step(i=step + 1)
            elb = loss[step].item()
            
            # Parameter updates
            old_w, new_w = new_w, retrieve_params()
            
            # Optional callback
            if callback is not None:
                callback(model, guide, step)
            
            # Convergence check
            stop, diff_params = all_stopping_criteria(old_w, new_w, e, step)
            self.params_history[step] = diff_params
            
            if stop:
                if conv >= patience:
                    print('Reached convergence')
                    break
                conv += 1
            else:
                conv = 0
                
        # Store trained model components
        self._model_trained = model
        self._guide_trained = guide
        self._loss_trained = loss.cpu().numpy() if self._CUDA else loss.numpy()
        
        return self._loss_trained, num_observations
        
    def learned_parameters_Clonal(self):
        """Optimized parameter retrieval for Clonal model with proper device handling."""
        # Get parameters and ensure they're on the correct device
        params = {k: v.to(self._device) if torch.is_tensor(v) else v 
                for k, v in self._guide_trained().items()}
        
        # Set default device before discrete inference
        if self._CUDA:
            # Extract device index for CUDA device
            device_idx = 0 
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(device_idx)
        
        try:
            # Run discrete inference
            map_states = self._model.model_2(self._model, learned_params=params)
            # Helper for device-aware numpy conversion
            to_numpy = lambda x: x.cpu().detach().numpy() if self._CUDA else x.detach().numpy()
            
            # Process states based on model type
            if self._model._params["allele_specific"]:
                states = torch.tensor(map_states, device=self._device)[1:].long()
                Major = self._model._params["Major"].to(self._device)
                minor = self._model._params["minor"].to(self._device)
                
                discrete_params = {
                    "CN_Major": to_numpy(Major[states]),
                    "CN_minor": to_numpy(minor[states])
                }
            else:
                states = torch.tensor(map_states, device=self._device)[1:]
                discrete_params = {
                    "CN_tot": to_numpy(states)
                }
            
            # Convert parameters to numpy with proper device handling
            trained_params_dict = {k: to_numpy(v) for k, v in params.items()}
            
            return {**trained_params_dict, **discrete_params}
        
        finally:
            # Restore original device
            if self._CUDA:
                torch.cuda.set_device(original_device)

    # def learned_parameters_Clonal(self):
    #     """Optimized parameter retrieval for Clonal model."""
    #     params = self._guide_trained()
    #     map_states = self._model.model_2(self._model, learned_params=params)
        
    #     # Pre-compute device transfer
    #     to_numpy = lambda x: x.cpu().detach().numpy() if self._CUDA else x.detach().numpy()
        
    #     if self._model._params["allele_specific"]:
    #         states = torch.tensor(map_states, device=self._device)[1:].long()
    #         discrete_params = {
    #             "CN_Major": to_numpy(self._model._params["Major"][states]),
    #             "CN_minor": to_numpy(self._model._params["minor"][states])
    #         }
    #     else:
    #         discrete_params = {
    #             "CN_tot": to_numpy(torch.tensor(map_states, device=self._device)[1:])
    #         }
        
    #     # Convert parameters to numpy efficiently
    #     trained_params_dict = {k: to_numpy(v) for k, v in params.items()}
        
    #     return {**trained_params_dict, **discrete_params}


    def posterior_draws(self, num_samples=500, seed=3, sites=None):
        """
        Draw posterior samples from the trained guide for selected sites.
        Default sites capture purity, ploidy, and transitions (probs_x).
        Returns a dict of tensors keyed by site name; leading dim = num_samples.
        """

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
    
    