import torch
import pyro
import pyro.distributions as dist

class PurityPrior:
    def __init__(self, sample_type="cell_line"):
        """
        Implements prior for tumor purity/cellular fraction
        
        Parameters:
        sample_type: str, either "cell_line" or "clinical"
        """
        self.sample_type = sample_type
        
        if sample_type == "cell_line":
            # Single beta distribution for cell lines
            self.alpha = torch.tensor(50.0)
            self.beta = torch.tensor(2.0)
            
        elif sample_type == "clinical":
            # Mixture model for clinical samples
            self.weights = torch.tensor([0.4, 0.4, 0.2])  # High, medium, low purity
            self.alphas = torch.tensor([15.0, 8.0, 2.0])
            self.betas = torch.tensor([3.0, 8.0, 8.0])
            
    def sample(self, sample_shape=(1,)):
        """Sample from the purity prior"""
        with pyro.plate("sp", sample_shape[0] if len(sample_shape) > 0 else 1):
            if self.sample_type == "cell_line":
                purity = pyro.sample(
                    "purity",
                    dist.Beta(self.alpha, self.beta)
                )
            else:
                # Sample mixture component
                mixture_idx = pyro.sample(
                    "mixture_idx_purity",
                    dist.Categorical(self.weights),
                     infer={"enumerate": "parallel"} 
                )
                
                # Sample purity from selected beta component
                purity = pyro.sample(
                    "purity",
                    dist.Beta(
                        self.alphas[mixture_idx],
                        self.betas[mixture_idx]
                    )
                )
                
        return purity