import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLayer(nn.Module):
    def __init__(self,
        input_dim: int,
        prototypes: int | nn.Parameter,
        features: int | nn.Parameter,
        prototype_init=None,
        feature_init=None,
        approximate_sharpness=13
    ):
        super().__init__()

        if isinstance(features, int):
            self.features = nn.Parameter(torch.empty(features, input_dim))
        elif isinstance(features, nn.Parameter):
            self.features = features
        else:
            raise ValueError("features must be int or nn.Parameter")

        if isinstance(prototypes, int):
            self.prototypes = nn.Parameter(torch.empty(prototypes, input_dim))
        elif isinstance(prototypes, nn.Parameter):
            self.prototypes = prototypes
        else:
            raise ValueError("prototypes must be int or nn.Parameter")

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.theta = nn.Parameter(torch.ones(1))

        self.prototype_init = prototype_init
        self.feature_init = feature_init
        self.approximate_sharpness = approximate_sharpness

        self.reset_parameters()

    def reset_parameters(self):
        if self.feature_init is not None:
            self.feature_init(self.features)
        if self.prototype_init is not None:
            self.prototype_init(self.prototypes)

        #nn.init.uniform_(self.alpha, 0.04, 0.12)
        #nn.init.uniform_(self.beta, 0.001, 0.004)
        #nn.init.uniform_(self.theta, 0.7, 1.0)

    # Shifted indicator since we need a differentiable signal > 0 but binary mask is not such a thing.
    def indicator(self, x):
        sigma = (torch.tanh(self.approximate_sharpness * x) + 1) * 0.5
        weighted = x * sigma
        return weighted, sigma

    # Ignorematch with Product intersections
    def forward(self, x):
        B = x.size(0)
        P = self.prototypes.size(0)

        A = x @ self.features.T                    # [B, F]
        Pi = self.prototypes @ self.features.T     # [P, F]

        weighted_A, sigma_A = self.indicator(A)
        weighted_Pi, sigma_Pi = self.indicator(Pi)

        theta_val = self.theta.item()
        alpha_val = self.alpha.item()
        beta_val = self.beta.item()

        # theta * (weighted_A @ weighted_Pi.T)
        result = torch.addmm(
            torch.empty(B, P, device=x.device, dtype=x.dtype),
            weighted_A, weighted_Pi.T,
            beta=0, alpha=theta_val
        )

        # - alpha * (weighted_A @ (1 - sigma_Pi).T)
        result = torch.addmm(
            result,
            weighted_A, (1 - sigma_Pi).T,
            beta=1, alpha=-alpha_val
        )

        # beta * ((1 - sigma_A) @ weighted_Pi.T)
        result = torch.addmm(
            result,
            (1 - sigma_A), weighted_Pi.T,
            beta=1, alpha=-beta_val
        )

        return result
