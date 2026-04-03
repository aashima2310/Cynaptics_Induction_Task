# Task 1 — GPT Shakespeare: Pretraining from Scratch

## Overview
A decoder-only GPT-style Transformer built entirely from scratch using PyTorch,
trained on Tiny Shakespeare to generate Shakespeare-style text.
No pretrained weights — every component is hand-written.

---

## Setup & Dataset
```bash
pip install torch tokenizers
mkdir data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---

## How to Train
```bash
python train.py
```

## How to Generate
```bash
python generate.py
```

---

## Sample Output
```
ROMEO:
What say you to this matter?
JULIET:
I know not what to say, my lord,
But I will speak the truth.
```

---

## Architecture
```
Token + Positional Embeddings
       ↓
Transformer Block × 6
  └── LayerNorm → Masked MHA → Residual
  └── LayerNorm → FFN → Residual
       ↓
LayerNorm → Linear → logits
```

---

## Hyperparameters

| Parameter     | Value  |
|---------------|--------|
| d_model       | 256    |
| n_heads       | 8      |
| n_layers      | 6      |
| block_size    | 128    |
| batch_size    | 64     |
| vocab_size    | 1000   |
| learning rate | 3e-4   |
| max_iters     | 10000  |
| dropout       | 0.2    |

---

## Tokenizer
Custom BPE tokenizer trained on Shakespeare data.
Vocab size 1000 — better than character level (65),
smaller than GPT2 (50257).
