"""
Audio→cross-attention projector.

CLAP gives a single 512-dim vector per clip. SD 1.5's UNet expects
encoder_hidden_states of shape (B, N, 768). We learn an MLP that maps the
audio vector to N learned tokens of dim 768, used in place of text embeddings.
"""

import torch
import torch.nn as nn


class AudioProjector(nn.Module):
    def __init__(self, audio_dim: int = 512, num_tokens: int = 8, hidden_dim: int = 768, mlp_dim: int = 1024):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_tokens * hidden_dim),
        )
        self.null_token = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.normal_(self.null_token, std=0.02)

    def forward(self, audio_emb: torch.Tensor) -> torch.Tensor:
        """audio_emb: (B, audio_dim) → (B, num_tokens, hidden_dim)"""
        out = self.net(audio_emb).view(-1, self.num_tokens, self.hidden_dim)
        return out

    def null(self, batch_size: int, device, dtype) -> torch.Tensor:
        return self.null_token.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
