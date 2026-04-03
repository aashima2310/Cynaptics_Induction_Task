# All Hyperparameters 

import torch

vocab_size = 1000   # custom BPE vocab
d_model    = 256
n_heads    = 8
n_layers   = 6
block_size = 128
batch_size = 64
max_iters  = 10000
EVAL_EVERY = 200
LR         = 3e-4
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
