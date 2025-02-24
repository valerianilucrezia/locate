import torch
import pyro
import pyro.distributions as dist

class PloidyPrior:
    def __init__(self, sample_type="cell_line"):
        """
        Implements a flexible ploidy distribution for different sample types.
        
        Parameters:
        sample_type: str, one of "cell_line", "clinical"
        """
        self.sample_type = sample_type
        
        if sample_type == "cell_line":
            # More flexible mixture for cell lines
            self.weights = torch.tensor([0.35, 0.25, 0.25, 0.15])
            self.mus = torch.tensor([2.0, 3.0, 4.0, 5.0])
            # Increasing variance for higher ploidies
            self.sigmas = torch.tensor([0.15, 0.2, 0.25, 0.3])
            
        elif sample_type == "clinical":
            # More conservative mixture for primary tumors
            self.weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
            self.mus = torch.tensor([2.0, 3.0, 4.0, 5.0])
            self.sigmas = torch.tensor([0.15, 0.2, 0.25, 0.3])
            
            
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
    def sample(self, sample_shape=(1,)):
        """
        Sample from the ploidy distribution
        
        Parameters:
        sample_shape: tuple, shape of samples to generate
        
        Returns:
        samples: tensor of ploidy samples
        """
        with pyro.plate("samples", sample_shape[0] if len(sample_shape) > 0 else 1):
            # Sample mixture component
            mixture_idx = pyro.sample(
                "mixture_idx",
                dist.Categorical(self.weights)
            )
            
            # Sample ploidy from selected component
            ploidy = pyro.sample(
                "ploidy",
                dist.Normal(
                    self.mus[mixture_idx],
                    self.sigmas[mixture_idx]
                )
            )
            
        return ploidy