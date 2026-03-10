import torch
import torch.nn as nn
from .attention import SelfAttention

class TransformerBlock(nn.Module):

    def __init__(self, embed_size):
        super().__init__()

        self.attention = SelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )

        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):

        attention = self.attention(x)
        x = self.norm1(attention + x)

        forward = self.feed_forward(x)
        out = self.norm2(forward + x)

        return out