import pyro
import torch
import matplotlib.pyplot as plt
import numpy as np
from pyro.infer import Predictive
from collections import defaultdict

class ClonalModelChecker:
    """
    A class to perform model checking for Clonal models trained with LOCATE
    """
    def __init__(self, locate_instance):
        """
        Initialize with a trained LOCATE instance
        
        Parameters
        ----------
        locate_instance : LOCATE
            A trained LOCATE instance with a Clonal model
        """
        self.locate = locate_instance
        self.model = locate_instance._model.guide
        
    def track_parameters(self, steps=400, param_optimizer={"lr": 0.05}):
        """
        Run inference while tracking parameter values
        
        Parameters
        ----------
        steps : int
            Number of inference steps
        param_optimizer : dict
            Optimizer parameters
            
        Returns
        -------
        dict
            Dictionary of parameter traces and losses
        """
        # Initialize parameter storage
        param_traces = defaultdict(list)
        losses = []
        
        # Set optimizer parameters
        if param_optimizer is not None:
            for key, value in param_optimizer.items():
                self.locate.optimizer_params[key] = value
        
        # Create optimizer and SVI instance
        optimizer = self.locate.optimizer(self.locate.optimizer_params)
        svi = pyro.infer.SVI(self.model.model, self.model.guide, optimizer, loss=self.locate.loss())
        
        # Run inference and track parameters
        for i in range(steps):
            loss = svi.step()
            losses.append(loss)
            
            # Store current parameter values
            for name, value in pyro.get_param_store().items():
                param_traces[name].append(value.detach().clone())
            
            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}/{steps} - Loss: {loss:.4f}")
        
        return {"param_traces": param_traces, "losses": losses}
    
    def plot_parameter_traces(self, param_traces):
        """
        Plot parameter traces over iterations
        
        Parameters
        ----------
        param_traces : dict
            Dictionary of parameter traces from track_parameters
            
        Returns
        -------
        figs : list
            List of matplotlib figures
        """
        figs = []
        for param_name, values in param_traces.items():
            # Convert list of tensors to a stacked tensor
            if isinstance(values[0], torch.Tensor):
                param_shape = values[0].shape
                
                # Handle different parameter shapes
                if len(param_shape) == 0:  # scalar
                    plt.figure(figsize=(10, 6))
                    plt.plot([v.item() for v in values])
                    plt.xlabel("Iteration")
                    plt.ylabel(param_name)
                    plt.title(f"Parameter {param_name} over iterations")
                    figs.append(plt.gcf())
                
                elif len(param_shape) == 1:  # vector
                    plt.figure(figsize=(12, 6))
                    stacked = torch.stack(values)
                    for j in range(param_shape[0]):
                        plt.plot(stacked[:, j].cpu().numpy(), label=f"dim {j}")
                    plt.xlabel("Iteration")
                    plt.ylabel(param_name)
                    plt.title(f"Parameter {param_name} over iterations")
                    plt.legend()
                    figs.append(plt.gcf())
                
                elif len(param_shape) == 2:  # matrix
                    # Plot the final matrix as a heatmap
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Plot the evolution of selected cells in the matrix
                    stacked = torch.stack(values).cpu().numpy()
                    n_iters = len(values)
                    
                    # Select some cells to track
                    for i in range(min(5, param_shape[0])):
                        for j in range(min(5, param_shape[1])):
                            if i == j:  # Only diagonal elements for simplicity
                                axes[0].plot(stacked[:, i, j], label=f"({i},{j})")
                    
                    axes[0].set_xlabel("Iteration")
                    axes[0].set_ylabel("Value")
                    axes[0].set_title(f"Selected cells from {param_name}")
                    axes[0].legend()
                    
                    # Plot final matrix as heatmap
                    im = axes[1].imshow(values[-1].cpu())
                    plt.colorbar(im, ax=axes[1])
                    axes[1].set_title(f"Final {param_name} matrix")
                    
                    figs.append(fig)
        
        return figs
    
    def posterior_predictive_check(self, num_samples=10):
        """
        Perform posterior predictive checks
        
        Parameters
        ----------
        num_samples : int
            Number of posterior samples to generate
            
        Returns
        -------
        dict
            Dictionary of posterior samples and predictive samples
        """
        # Get the trained guide
        guide = self.model.guide
        
        # Get posterior samples
        predictive = Predictive(guide, num_samples=num_samples)
        posterior_samples = predictive()
        
        # Generate data from the posterior predictive distribution
        model_with_posterior = lambda *args, **kwargs: self.model.model_2(posterior_samples)
        predictive_model = Predictive(model_with_posterior, posterior_samples, num_samples=num_samples)
        predicted_data = predictive_model()
        
        return {"posterior_samples": posterior_samples, "predicted_data": predicted_data}
    
    def plot_predictive_checks(self, posterior_check_results):
        """
        Plot posterior predictive checks
        
        Parameters
        ----------
        posterior_check_results : dict
            Results from posterior_predictive_check
            
        Returns
        -------
        figs : list
            List of matplotlib figures
        """
        figs = []
        predicted_data = posterior_check_results["predicted_data"]
        
        # Plot comparisons for each data type
        for key in self.model._data.keys():
            if self.model._data[key] is not None and key in predicted_data:
                # Extract observed data
                observed = self.model._data[key]
                
                # Create a figure for this data type
                fig = plt.figure(figsize=(12, 6))
                
                # Plot based on data type
                if key == "baf":
                    # Plot observed BAF as histogram
                    plt.hist(observed.cpu().flatten().numpy(), bins=30, alpha=0.7, label="Observed", density=True)
                    
                    # Plot a few predicted samples
                    for i in range(min(3, len(predicted_data[key]))):
                        plt.hist(predicted_data[key][i].cpu().flatten().numpy(), bins=30, alpha=0.3, 
                                density=True, label=f"Predicted {i+1}")
                
                elif key == "dr":
                    # Plot observed vs predicted DR 
                    obs_flat = observed.cpu().flatten().numpy()
                    
                    plt.plot(obs_flat, 'k-', alpha=0.7, label="Observed")
                    
                    # Plot a few predicted samples
                    for i in range(min(3, len(predicted_data[key]))):
                        pred_flat = predicted_data[key][i].cpu().flatten().numpy()
                        plt.plot(pred_flat, alpha=0.3, label=f"Predicted {i+1}")
                
                plt.title(f"Posterior Predictive Check: {key}")
                plt.legend()
                figs.append(fig)
        
        return figs
    
    def evaluate_model_fit(self):
        """
        Evaluate overall model fit using multiple metrics
        
        Returns
        -------
        dict
            Dictionary of fit metrics
        """
        # Run posterior predictive checks
        ppc_results = self.posterior_predictive_check(num_samples=20)
        
        # Calculate metrics for each data type
        metrics = {}
        
        for key in self.model._data.keys():
            if self.model._data[key] is not None and key in ppc_results["predicted_data"]:
                observed = self.model._data[key].cpu()
                predicted = ppc_results["predicted_data"][key].cpu()
                
                # Calculate mean absolute error
                mae = torch.mean(torch.abs(observed - predicted.mean(dim=0))).item()
                
                # Calculate correlation
                obs_flat = observed.flatten()
                pred_flat = predicted.mean(dim=0).flatten()
                
                # Remove NaNs for correlation calculation
                valid_idx = torch.logical_and(~torch.isnan(obs_flat), ~torch.isnan(pred_flat))
                if valid_idx.sum() > 0:
                    corr = torch.corrcoef(
                        torch.stack([obs_flat[valid_idx], pred_flat[valid_idx]])
                    )[0, 1].item()
                else:
                    corr = float('nan')
                
                metrics[key] = {
                    "mae": mae,
                    "correlation": corr
                }
        
        return metrics

# Usage example
def run_model_checking(locate_instance):
    """
    Run comprehensive model checking on a trained LOCATE instance
    
    Parameters
    ----------
    locate_instance : LOCATE
        A LOCATE instance with a trained Clonal model
        
    Returns
    -------
    checker : ClonalModelChecker
        The model checker instance with results
    """
    # Initialize checker
    checker = ClonalModelChecker(locate_instance)
    
    # Run parameter tracking (can skip if already trained)
    print("Tracking parameters during training...")
    tracking_results = checker.track_parameters(steps=400)
    
    # Plot parameter traces
    print("Plotting parameter traces...")
    trace_figs = checker.plot_parameter_traces(tracking_results["param_traces"])
    
    # Run posterior predictive checks
    print("Performing posterior predictive checks...")
    ppc_results = checker.posterior_predictive_check()
    
    # Plot predictive checks
    print("Plotting predictive checks...")
    ppc_figs = checker.plot_predictive_checks(ppc_results)
    
    # Evaluate model fit
    print("Evaluating model fit...")
    fit_metrics = checker.evaluate_model_fit()
    print("Fit metrics:", fit_metrics)
    
    return checker

# Example implementation in your code:
"""
locate = l.LOCATE(CUDA = False)
locate.set_model(Clonal)
locate.set_optimizer(ClippedAdam)
locate.set_loss(TraceEnum_ELBO)
locate.initialize_model({"baf": data_input["baf"],
    "dr": data_input["dr"],
    "dp_snp": data_input["dp_snp"]
})
locate.set_model_params({"jumping_prob" : 1e-6,
    "fix_purity": False,
    "fix_ploidy": True,
    "prior_purity": purity,
    "prior_ploidy": ploidy,
    "scaling_factors": [1,1,1],
    "hidden_dim":4,
    'init_probs': torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])})

# Run inference
ll = locate.run(steps = 400, param_optimizer = {"lr" : 0.05})

# Run model checking
checker = run_model_checking(locate)
"""