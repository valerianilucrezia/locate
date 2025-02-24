import matplotlib.pyplot as plt
import pyro
from collections import defaultdict

class PosteriorTracker:
    def __init__(self, param_names):
        """
        Initialize a tracker for posterior parameters
        
        Parameters
        ----------
        param_names : list
            List of parameter names to track
        """
        self.param_values = defaultdict(list)
        self.param_names = param_names
        self.iterations = []
        self.current_iter = 0
    
    def __call__(self, model, guide, iteration):
        """
        Callback function to be used with pyro.infer.SVI
        """
        self.current_iter += iteration + 1
        self.iterations.append(self.current_iter)
        
        # Get the posterior (guide) params
        posterior_params = {}
        for name, value in pyro.get_param_store().items():
            posterior_params[name] = value.detach().clone()
            #print(value)
            #print(len(value))
        
        # Save values for tracked parameters
        for param_name in self.param_names:
            param_name = 'AutoDelta.'+param_name
            if param_name in posterior_params:
                self.param_values[param_name].append(posterior_params[param_name].cpu())
    
    def plot(self, figsize=(5, 5)):
        """
        Plot the tracked parameter values over iterations
        """
        n_params = len(self.param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
        
        if n_params == 1:
            axes = [axes]
        
        for i, param_name in enumerate(self.param_names):
            param_name = 'AutoDelta.'+param_name
            ax = axes[i]
            values = self.param_values[param_name]
            
            # Handle different parameter shapes
            if len(values) == 0:
                ax.text(0.5, 0.5, f"No data for {param_name}", 
                        ha='center', va='center', transform=ax.transAxes)
                continue
            
            param_shape = values[0].shape
            
            if len(param_shape) == 0:  # scalar parameter
                ax.plot(self.iterations, [v.item() for v in values], marker='o')
                ax.set_ylabel(param_name)
            
            elif len(param_shape) == 1:  # vector parameter
                for j in range(param_shape[0]):
                    # ax.plot(self.iterations, [v[j].item() for v in values], 
                    #         marker='o', label=f"dim {j}")
                    ax.hist([v[j].item() for v in values])
                ax.set_ylabel(param_name)
                ax.set_xlim(0,1)
                ax.legend()
            
            elif len(param_shape) == 2:  # matrix parameter
                # For matrices, plot as heatmap at specific iterations
                # Just show the last iteration for brevity
                if values:
                    last_val = values[-1]
                    im = ax.imshow(last_val.numpy(), aspect='auto')
                    ax.set_title(f"{param_name} at iteration {self.current_iter}")
                    plt.colorbar(im, ax=ax)
            
            else:
                ax.text(0.5, 0.5, f"Cannot plot {param_name} with shape {param_shape}", 
                        ha='center', va='center', transform=ax.transAxes)
        
        plt.xlabel("Iteration")
        plt.tight_layout()
        return
