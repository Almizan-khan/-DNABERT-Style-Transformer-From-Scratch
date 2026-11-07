import math
import torch
from torch import nn

class DNAEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        return self.dropout(self.layer_norm(self.token_embedding(input_ids)))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query, self.key, self.value = [nn.Linear(hidden_size, hidden_size) for _ in range(3)]
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, D = x.size()
        if mask is not None and not mask.bool().any(dim=1).all():
            raise ValueError('Attention requires at least one unmasked token per sequence')
        Q = self.query(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        slopes = torch.tensor([2**(-8*(i+1)/self.num_heads) for i in range(self.num_heads)], device=x.device)
        rel_pos = (torch.arange(L, device=x.device)[None, :] - torch.arange(L, device=x.device)[:, None]).abs()
        scores -= (slopes[:, None, None] * rel_pos).unsqueeze(0)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), V).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.output(out)

class EncoderBlock(nn.Module):
    def __init__(self, h, heads, inter, drop):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.attn = MultiHeadSelfAttention(h, heads, drop)
        self.ffn = nn.Sequential(nn.Linear(h, inter), nn.GELU(), nn.Dropout(drop), nn.Linear(inter, h))

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class DNABERT2ForMLM(nn.Module):
    def __init__(self, v, h, heads, inter, layers, drop):
        super().__init__()
        self.embedding = DNAEmbedding(v, h, drop)
        self.layers = nn.ModuleList([EncoderBlock(h, heads, inter, drop) for _ in range(layers)])
        self.mlm_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, v))

    def forward(self, ids, mask):
        x = self.embedding(ids)
        for layer in self.layers: x = layer(x, mask)
        return self.mlm_head(x)
