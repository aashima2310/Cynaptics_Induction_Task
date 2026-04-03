import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model ,head_size,block_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("mask" , torch.tril(torch.ones(block_size , block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape   # batch, block_size, d_model
        
        Q = self.query(x)   # (B, T, head_size)
        K = self.key(x)     # (B, T, head_size)
        V = self.value(x)   # (B, T, head_size)
        
        # Attention scores
        head_size = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / (head_size ** 0.5)  

        # Causal mask
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Softmax
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
    
        # Multiply by V
        output = weights @ V
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,block_size):
        super().__init__()
        self.head_size = d_model // n_heads            # head-size for each head which will go through attention
        self.heads = nn.ModuleList([SingleHeadAttention(d_model ,self.head_size,block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

