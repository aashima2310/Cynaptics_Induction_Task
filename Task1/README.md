# Task 1 — GPT Shakespeare (Pretraining from Scratch)

## Setup
pip install torch transformers

## Download Dataset
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## How to Train
python train.py

## How to Generate
python generate.py

## Sample Output
To be, gom wowh willst lals rothem youtie beth.
INRENCHOLIOMIS:
Somo cer, t CKng an, lead;
PA Lonoby mbur, l lyonou.

## Model Hyperparameters
| Parameter  | Value |
|------------|-------|
| d_model    | 256   |
| n_heads    | 8     |
| n_layers   | 6     |
| block_size | 128   |
| batch_size | 64    |
| vocab_size | 50257 |
| lr         | 3e-4  |
| iterations | 10000 |

## Architecture
- Decoder-only Transformer (GPT style)
- Token + Positional Embeddings
- 6 x Transformer Blocks
- Masked Self-Attention + Feed Forward
- Final LayerNorm + Linear head
