import torch
import pyro
import pyro.distributions as dist

def gamma_from_mean_var(mean: torch.Tensor, var: torch.Tensor):
    """
    Positive-support alternative for ploidy. Returns (concentration k, rate β)
    for Gamma(k, β) with the requested mean and variance.
    Mean = k/β, Var = k/β^2  ->  k = m^2/v, β = m/v
    """
    eps = torch.finfo(torch.float32).eps
    m = torch.clamp(mean, min=eps)
    v = torch.clamp(var, min=eps)
    k = (m * m) / v
    rate = m / v
    return k, rate


class PloidyPrior:
    def __init__(self, 
                sample_type="cell_line",
                estimate: float | None = None,
                variance: float | None = None,
                family: str = "gamma"):
        
        """
        Flexible ploidy prior.

        Modes:
        - Informative (if estimate & variance given):
            * family="gamma": Gamma(concentration, rate) matched to mean/var
              (positive support; good default for ploidy)
            * family="normal": Normal(mean, std) matched to mean/var
        - Empirical mixture (fallback): same as your current implementation.

        Args
        ----
        sample_type: "cell_line" | "clinical"
        estimate: mean ploidy to center the prior on (e.g., 2.7)
        variance: variance around the estimate (e.g., 0.15^2)
        family: "gamma" or "normal"
        """
        
        self.sample_type = sample_type
        self.estimate = estimate
        self.variance = variance
        self.family = family
        
        if (estimate is None) or (variance is None):
            # fallback mixture you already had
            if sample_type == "cell_line":
                self.weights = torch.tensor([0.35, 0.25, 0.25, 0.15])
                self.mus = torch.tensor([2.0, 3.0, 4.0, 5.0])
                self.sigmas = torch.tensor([0.15, 0.2, 0.25, 0.3])
            elif sample_type == "clinical":
                self.weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
                self.mus = torch.tensor([2.0, 3.0, 4.0, 5.0])
                self.sigmas = torch.tensor([0.15, 0.2, 0.25, 0.3])
            self.weights = self.weights / self.weights.sum()
            
    def sample(self, sample_shape=(1,)):
        with pyro.plate("ploidy_plate", sample_shape[0] if len(sample_shape) > 0 else 1):
            # Informative prior if provided
            if (self.estimate is not None) and (self.variance is not None):
                m = torch.tensor(float(self.estimate))
                v = torch.tensor(float(self.variance))
                if self.family.lower() == "gamma":
                    k, rate = gamma_from_mean_var(m, v)
                    ploidy = pyro.sample("ploidy", dist.Gamma(k, rate))
                elif self.family.lower() == "normal":
                    ploidy = pyro.sample("ploidy", dist.Normal(m, torch.sqrt(v)))
                else:
                    raise ValueError("family must be 'gamma' or 'normal'")
                return ploidy

            # Otherwise: your mixture prior
            mixture_idx = pyro.sample(
                "mixture_idx_ploidy",
                dist.Categorical(self.weights),
                infer={"enumerate": "parallel"},
            )
            ploidy = pyro.sample(
                "ploidy",
                dist.Normal(self.mus[mixture_idx], self.sigmas[mixture_idx])
            )
            return ploidy

        
    # def sample(self, sample_shape=(1,)):
    #     """
    #     Sample from the ploidy distribution
        
    #     Parameters:
    #     sample_shape: tuple, shape of samples to generate
        
    #     Returns:
    #     samples: tensor of ploidy samples
    #     """
    #     with pyro.plate("samples", sample_shape[0] if len(sample_shape) > 0 else 1):
    #         # Sample mixture component
    #         mixture_idx = pyro.sample(
    #             "mixture_idx",
    #             dist.Categorical(self.weights),
    #              infer={"enumerate": "parallel"} 
    #         )
            
    #         # Sample ploidy from selected component
    #         ploidy = pyro.sample(
    #             "ploidy",
    #             dist.Normal(
    #                 self.mus[mixture_idx],
    #                 self.sigmas[mixture_idx]
    #             )
    #         )
            
    #     return ploidy
    
