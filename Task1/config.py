# All Hyperparameters 

import torch

VOCAB_SIZE = 1000
D_MODEL    = 256
N_HEADS    = 8
N_LAYERS   = 6
BLOCK_SIZE = 128
BATCH_SIZE = 64
MAX_ITERS  = 10000
EVAL_EVERY = 200
LR         = 3e-4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
