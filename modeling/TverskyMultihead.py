import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def tversky_multihead_similarity(x, features, prototypes, theta, alpha, beta, n_heads):
    batch_size, total_dim = x.shape
    d_head = total_dim // n_heads

    # Multihead expansion
    x = rearrange(x, 'b (h d) -> b h d', h=n_heads)                    # [batch, heads, d_head]
    features = rearrange(features, 'f (h d) -> h f d', h=n_heads)      # [heads, features, d_head]
    prototypes = rearrange(prototypes, 'p (h d) -> h p d', h=n_heads)  # [heads, prototypes, d_head]

    # Full features (hidden * features per head, proto * features per head)
    x_features = torch.einsum('bhd,hfd->bhf', x, features)             # [batch, heads, features]
    p_features = torch.einsum('hpd,hfd->hpf', prototypes, features)    # [heads, prototypes, features]

    # Presence masking
    x_present = F.relu(x_features)                                     # [batch, heads, features]
    p_present = F.relu(p_features)                                     # [heads, prototypes, features]
    x_weighted = x_features * x_present                                # [batch, heads, features]
    p_weighted = p_features * p_present                                # [heads, prototypes, features]

    # BMM to avoid [batch, heads, prototypes, features] materialization
    x_weighted_h = x_weighted.transpose(0, 1)                         # [heads, batch, features]
    p_weighted_h = p_weighted.transpose(1, 2)                         # [heads, features, prototypes]

    # Original: torch.einsum('bhf,hpf->bhp', x_weighted, p_weighted) would require broadcasting
    # where intermediate tensors expand to [batch, heads, prototypes, features] for element-wise multiply
    # Equivalent: sum_f(x_weighted[b,h,f] * p_weighted[h,p,f]) for each (b,h,p)
    # = sum_f((x_weighted[b,h,f]) * (p_weighted[h,p,f]))
    # = x_weighted[b,h,:] @ p_weighted[h,:,p] for each head h
    # = torch.bmm([heads, batch, features], [heads, features, prototypes])
    common = torch.bmm(x_weighted_h, p_weighted_h).transpose(0, 1)    # [batch, heads, prototypes]

    # Same idea, avoid [batch, heads, prototypes, features] materialization
    x_weighted_sum = x_weighted.sum(dim=2, keepdim=True)              # [batch, heads, 1]
    p_present_h = p_present.transpose(1, 2)                           # [heads, features, prototypes]

    # Original: torch.einsum('bhf,hpf->bhp', x_weighted, p_present)
    # where p_present would broadcast to [batch, heads, prototypes, features]
    # Equivalent: sum_f(x_weighted[b,h,f] * p_present[h,p,f]) for each (b,h,p)
    # = x_weighted[b,h,:] @ p_present[h,:,p] for each head h
    # = torch.bmm([heads, batch, features], [heads, features, prototypes])
    x_p_interaction = torch.bmm(x_weighted_h, p_present_h).transpose(0, 1)  # [batch, heads, prototypes]
    x_distinctive = x_weighted_sum - x_p_interaction                   # [batch, heads, prototypes]

    # And again same to avoid [batch, heads, prototypes, features] materialization
    p_weighted_sum = p_weighted.sum(dim=2).unsqueeze(0)               # [1, heads, prototypes]
    x_present_h = x_present.transpose(0, 1)                           # [heads, batch, features]

    # Original: torch.einsum('bhf,hpf->bhp', x_present, p_weighted)
    # where x_present would broadcast to [batch, heads, prototypes, features]
    # Equivalent: sum_f(x_present[b,h,f] * p_weighted[h,p,f]) for each (b,h,p)
    # = x_present[b,h,:] @ p_weighted[h,:,p] for each head h
    # = torch.bmm([heads, batch, features], [heads, features, prototypes])
    p_x_interaction = torch.bmm(x_present_h, p_weighted_h).transpose(0, 1)  # [batch, heads, prototypes]
    p_distinctive = p_weighted_sum - p_x_interaction                   # [batch, heads, prototypes]

    theta = rearrange(theta, 'h 1 -> 1 h 1')
    alpha = rearrange(alpha, 'h 1 -> 1 h 1')
    beta = rearrange(beta, 'h 1 -> 1 h 1')

    tversky_out = theta * common - alpha * x_distinctive - beta * p_distinctive  # [batch, heads, prototypes]
    tversky_out = rearrange(tversky_out, 'b h d -> b (h d)') # [batch, heads * prototypes]
    return tversky_out

class TverskyMultihead(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, num_prototypes: int, num_features: int):
        super().__init__()

        self.features = nn.Parameter(torch.empty(num_features, hidden_dim))  # Feature bank
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_dim))  # Prototypes
        self.alpha = nn.Parameter(torch.zeros(n_heads, 1))  # scale for a_distinctive
        self.beta = nn.Parameter(torch.zeros(n_heads, 1))   # Scale for b_distinctive
        self.theta = nn.Parameter(torch.zeros(n_heads, 1))  # General scale
        self.n_heads = n_heads

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.features, -.27, 1)
        torch.nn.init.uniform_(self.prototypes, -.27, 1)

    def forward(self, x):
        batch_size, features = x.shape
        return tversky_multihead_similarity(x, self.features, self.prototypes, self.theta, self.alpha, self.beta, self.n_heads)
