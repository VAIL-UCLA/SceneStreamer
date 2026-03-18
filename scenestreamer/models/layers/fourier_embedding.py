"""
Credit: https://github.com/rainmaker22/SMART/blob/a329361b63082359be56c9bfaa7e76336c19115f/smart/layers/fourier_embedding.py
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.square(nn.functional.relu(x))


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int, is_v7=None) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                ) for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = sum(categorical_embs)
            else:
                raise ValueError('Both continuous_inputs and categorical_embs are None')
        else:
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            # Warning: if your data are noisy, don't use learnable sinusoidal embedding
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            # continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            continuous_embs = None
            for i in range(self.input_dim):
                out = self.mlps[i](x[:, i])
                if continuous_embs is None:
                    continuous_embs = out
                else:
                    continuous_embs += out
            x = continuous_embs
            if categorical_embs is not None:
                x = x + sum(categorical_embs)
        return self.to_out(x)
