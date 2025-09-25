import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.trainutils import count_parameters_layerwise

class TverskyLayer(nn.Module):
    def __init__(self, input_dim: int, num_prototypes: int, num_features: int):
        super().__init__()

        self.features = nn.Parameter(torch.rand(num_features, input_dim) * 2 - 1)   # Feature bank
        self.prototypes = nn.Parameter(torch.rand(num_prototypes, input_dim) * 2 - 1)  # Prototypes

        # Zeros seemed to give better early on training dynamics
        self.alpha = nn.Parameter(torch.zeros(1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.zeros(1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.zeros(1))  # General scale

    def forward(self, x):
        batch_size, input_dim = x.shape
        num_prototypes = self.prototypes.shape[0]

        # Expand for all pairwise comparisons
        x_expanded = x.unsqueeze(1).expand(-1, num_prototypes, -1)  # [batch, num_prototypes, input_dim]
        proto_expanded = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_prototypes, input_dim]

        # Feature activations for inputs and prototypes
        x_features = torch.einsum('bpi,fi->bpf', x_expanded, self.features)  # [batch, num_prototypes, num_features]
        p_features = torch.einsum('bpi,fi->bpf', proto_expanded, self.features)  # [batch, num_prototypes, num_features]

        # a · fk > 0
        x_present = torch.nn.functional.relu(x_features)
        p_present = torch.nn.functional.relu(p_features)

        # Set operations
        both_present = x_present * p_present  # A ∩ B
        x_only = x_present * (1 - p_present)  # A - B
        p_only = p_present * (1 - x_present)  # B - A

        # Feature measures
        common = (x_features * p_features * both_present).sum(dim=2)  # [batch, num_prototypes]
        x_distinctive = (x_features * x_only).sum(dim=2)  # [batch, num_prototypes]
        p_distinctive = (p_features * p_only).sum(dim=2)  # [batch, num_prototypes]

        # Tversky similarity for all pairs
        return self.theta * common - self.alpha * x_distinctive - self.beta * p_distinctive
