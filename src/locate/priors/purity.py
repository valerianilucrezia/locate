import torch
import pyro
import pyro.distributions as dist

def beta_from_mean_var(mean: torch.Tensor, var: torch.Tensor):
    """
    Given mean m in (0,1) and variance v with 0 < v < m(1-m),
    return (alpha, beta) for a Beta(alpha, beta).
    """
    eps = torch.finfo(torch.float32).eps
    m = torch.clamp(mean, eps, 1 - eps)
    max_var = m * (1 - m) - eps
    v = torch.clamp(var, min=eps, max=max_var)

    t = m * (1 - m) / v - 1.0
    alpha = m * t
    beta = (1 - m) * t
    return alpha, beta


class PurityPrior:
    def __init__(
        self,
        sample_type: str = "cell_line",
        *,
        estimate= None,
        variance= None,
    ):
        """
        Tumor purity prior with optional informative Beta(m, v).

        Modes:
        - Informative (if estimate & variance given): Beta with alpha/beta
          computed so that mean=estimate and var=variance (INCOMMON-style).
        - Empirical defaults (fallback): your previous cell_line / clinical priors.
        """
        self.sample_type = sample_type
        self.estimate = estimate
        self.variance = variance
        
        if (estimate is None) or (variance is None):
            if sample_type == "cell_line":
                self.alpha = torch.tensor(50.0)
                self.beta = torch.tensor(2.0)
            elif sample_type == "clinical":
                self.weights = torch.tensor([0.4, 0.4, 0.2])  # high / mid / low
                self.alphas = torch.tensor([15.0, 8.0, 2.0])
                self.betas  = torch.tensor([3.0,  8.0, 8.0])

    def sample(self, sample_shape=(1,)):
        with pyro.plate("purity_plate", sample_shape[0] if len(sample_shape) > 0 else 1):
            # Informative Beta if provided
            if (self.estimate is not None) and (self.variance is not None):
                m = torch.tensor(float(self.estimate))
                v = torch.tensor(float(self.variance))
                alpha, beta = beta_from_mean_var(m, v)
                purity = pyro.sample("purity", dist.Beta(alpha, beta))
                return purity

            # Otherwise: your defaults
            if self.sample_type == "cell_line":
                purity = pyro.sample("purity", dist.Beta(self.alpha, self.beta))
            else:
                mixture_idx = pyro.sample(
                    "mixture_idx_purity",
                    dist.Categorical(self.weights),
                    infer={"enumerate": "parallel"},
                )
                purity = pyro.sample(
                    "purity",
                    dist.Beta(self.alphas[mixture_idx], self.betas[mixture_idx])
                )
            return purity
        
    # def sample(self, sample_shape=(1,)):
    #     """Sample from the purity prior"""
    #     with pyro.plate("sp", sample_shape[0] if len(sample_shape) > 0 else 1):
    #         if self.sample_type == "cell_line":
    #             purity = pyro.sample(
    #                 "purity",
    #                 dist.Beta(self.alpha, self.beta)
    #             )
    #         else:
    #             # Sample mixture component
    #             mixture_idx = pyro.sample(
    #                 "mixture_idx_purity",
    #                 dist.Categorical(self.weights),
    #                  infer={"enumerate": "parallel"} 
    #             )
                
    #             # Sample purity from selected beta component
    #             purity = pyro.sample(
    #                 "purity",
    #                 dist.Beta(
    #                     self.alphas[mixture_idx],
    #                     self.betas[mixture_idx]
    #                 )
    #             )
                
    #     return purity
