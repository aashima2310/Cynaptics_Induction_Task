import torch.nn as nn
from src.attention import MultiHeadAttention
from src.ffn import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads, block_size):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, block_size)
        self.ffn = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x + self.dropout(self.attention(self.ln1(x)))   # attention
        x = x + self.dropout(self.ffn(self.ln2(x)))      # feedforward
        return x
