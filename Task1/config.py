# All Hyperparameters 

vocab_size  = 50257
d_model     = 256
n_heads    = 8
n_layers    = 6
block_size  = 128
batch_size  = 64
max_iters   = 10000
EVAL_EVERY  = 200
LR          = 3e-4
device      = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'