import torch
import torch.nn as nn

class TimeDependentMLP(nn.Module):
    """
    A unified MLP that handles time embeddings and conditional inputs automatically.
    Replaces: MLPInstFlexible
    """
    def __init__(self, z_dim=2, cond_dim=0, width=256, depth=4, output_dim=2):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        # Time embedding: Projects scalar t -> width
        self.time_proj = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU()
        )

        input_dim = z_dim + width + cond_dim
        layers = [nn.Linear(input_dim, width), nn.SiLU()]
        
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
            
        self.net = nn.Sequential(*layers, nn.Linear(width, output_dim))

    def forward(self, z, t, c=None):
        # 1. Handle Time Broadcasting
        if t.dim() == 0: t = t.view(1, 1)
        elif t.dim() == 1: t = t.view(-1, 1)
        
        # 2. Handle Condition Broadcasting
        if c is not None and c.dim() == 1:
            c = c.view(-1, 1)
            
        # 3. Embed & Concatenate
        t_emb = self.time_proj(t)
        
        inputs = [z, t_emb]
        if c is not None:
            # Broadcast c to match z if necessary (e.g. for single condition)
            if c.shape[0] != z.shape[0]:
                c = c.expand(z.shape[0], -1)
            inputs.append(c)
            
        h = torch.cat(inputs, dim=-1)
        return self.net(h)