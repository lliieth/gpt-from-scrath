import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, embed_size):
        super().__init__()

        self.embed_size = embed_size

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = torch.matmul(Q, K.transpose(-2, -1))
        attention = attention / (self.embed_size ** 0.5)

        weights = F.softmax(attention, dim=-1)

        out = torch.matmul(weights, V)

        return self.fc_out(out)