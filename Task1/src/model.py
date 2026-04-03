import torch
import torch.nn as nn
from src.block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, n_heads, n_layers):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, block_size) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T    = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_final(x)
        logits  = self.lm_head(x)
        return logits
