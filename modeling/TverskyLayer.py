import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.trainutils import count_parameters_layerwise

class TverskyLayer(nn.Module):
    def __init__(self, input_dim: int, num_prototypes: int, num_features: int):
        super().__init__()

        self.features = nn.Parameter(torch.empty(num_features, input_dim))  # Feature bank
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, input_dim))  # Prototypes
        self.alpha = nn.Parameter(torch.zeros(1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.zeros(1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.zeros(1))  # General scale

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.features, -.27, 1)
        torch.nn.init.uniform_(self.prototypes, -.27, 1)

        # Recommended by paper
        #torch.nn.init.uniform_(self.alpha, 0, 2)
        #torch.nn.init.uniform_(self.beta, 0, 2)
        #torch.nn.init.uniform_(self.theta, 0, 2)

    def forward(self, x):
        batch_size = x.shape[0]

        x_features = torch.matmul(x, self.features.T)  # [batch, features]
        p_features = torch.matmul(self.prototypes, self.features.T)  # [prototypes, features]
        x_present = F.relu(x_features)  # [batch, features]
        p_present = F.relu(p_features)  # [prototypes, features]

        # Reformulated to avoid materializing large tensors:
        # Original: (x_features * p_features * both_present).sum(dim=2)
        # where both_present = x_present * p_present (broadcasted to [batch, prototypes, features])
        # Equivalent: sum_f(x_features[f] * p_features[f] * x_present[f] * p_present[f])
        # = sum_f((x_features[f] * x_present[f]) * (p_features[f] * p_present[f]))
        # = (x_features * x_present) @ (p_features * p_present).T

        x_weighted = x_features * x_present  # [batch, features]
        p_weighted = p_features * p_present  # [prototypes, features]
        common = torch.matmul(x_weighted, p_weighted.T)  # [batch, prototypes]

        # Original: (x_features * x_only).sum(dim=2)
        # where x_only = x_present * (1 - p_present) (broadcasted)
        # = sum_f(x_features[f] * x_present[f] * (1 - p_present[f]))
        # = sum_f(x_features[f] * x_present[f]) - sum_f(x_features[f] * x_present[f] * p_present[f])
        # = x_weighted.sum(1) - (x_weighted @ p_present.T)

        x_weighted_sum = x_weighted.sum(dim=1, keepdim=True)  # [batch, 1]
        x_p_interaction = torch.matmul(x_weighted, p_present.T)  # [batch, prototypes]
        x_distinctive = x_weighted_sum - x_p_interaction  # [batch, prototypes]

        # Original: (p_features * p_only).sum(dim=2)
        # where p_only = p_present * (1 - x_present) (broadcasted)
        # = sum_f(p_features[f] * p_present[f] * (1 - x_present[f]))
        # = sum_f(p_features[f] * p_present[f]) - sum_f(p_features[f] * p_present[f] * x_present[f])
        # = p_weighted.sum(1) - (x_present @ p_weighted.T)

        p_weighted_sum = p_weighted.sum(dim=1).unsqueeze(0)  # [1, prototypes]
        x_p_weighted_interaction = torch.matmul(x_present, p_weighted.T)  # [batch, prototypes]
        p_distinctive = p_weighted_sum - x_p_weighted_interaction  # [batch, prototypes]

        return self.theta * common - self.alpha * x_distinctive - self.beta * p_distinctive
