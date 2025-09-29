import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.trainutils import count_parameters_layerwise

class TverskyLayer(nn.Module):
    def __init__(self, input_dim: int, num_prototypes: int, num_features: int, use_cached_forward=True):
        super().__init__()

        self.features = nn.Parameter(torch.empty(num_features, input_dim))  # Feature bank
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, input_dim))  # Prototypes
        self.alpha = nn.Parameter(torch.zeros(1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.zeros(1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.zeros(1))  # General scale
        self.use_cached_forward = use_cached_forward

        if use_cached_forward:
            self.register_buffer('cached_matrix', None)
            self.register_buffer('cached_proto_sum', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.features, -.27, 1)
        torch.nn.init.uniform_(self.prototypes, -.27, 1)

        # Recommended by paper
        #torch.nn.init.uniform_(self.alpha, 0, 2)
        #torch.nn.init.uniform_(self.beta, 0, 2)
        #torch.nn.init.uniform_(self.theta, 0, 2)

    def forward(self, x):
        x_features = x @ self.features.T
        x_present = F.softplus(x_features, beta=10)
        x_weighted = x_features * x_present

        if not self.use_cached_forward or self.training:
            proto_features = self.prototypes @ self.features.T
            proto_present = F.softplus(proto_features, beta=10)
            proto_weighted = proto_features * proto_present
            proto_sum = proto_weighted.sum(dim=1)

            fused_matrix = (self.theta + self.beta) * proto_weighted - self.alpha * proto_present
        else:
            if self.cached_matrix is None:
                with torch.no_grad():
                    proto_features = self.prototypes @ self.features.T
                    proto_present = F.softplus(proto_features, beta=10)
                    proto_weighted = proto_features * proto_present
                    self.cached_proto_sum = proto_weighted.sum(dim=1)
                    self.cached_matrix = (self.theta + self.beta) * proto_weighted - self.alpha * proto_present

            fused_matrix = self.cached_matrix
            proto_sum = self.cached_proto_sum

        result = x_weighted @ fused_matrix.T

        x_sum = rearrange(x_weighted.sum(dim=1), 'b -> b 1')
        proto_sum_broadcast = rearrange(proto_sum, 'p -> 1 p')

        result = result - self.alpha * x_sum - self.beta * proto_sum_broadcast

        return result # [batch, num_prototypes]
