import torch
from src.tokenizer import train_tokenizer, load_tokenizer
import os

def load_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    # train tokenizer if not already trained
    if not os.path.exists('shakespeare-vocab.json'):
        tokenizer = train_tokenizer(file_path, vocab_size=1000)
    else:
        tokenizer = load_tokenizer()

    # encode full text
    encoded    = tokenizer.encode(text).ids
    vocab_size = tokenizer.get_vocab_size()

    print(f"Vocab size     : {vocab_size}")
    print(f"Encoded length : {len(encoded)}")

    return encoded, tokenizer, vocab_size


def get_batches(encoded, block_size, batch_size):
    data            = torch.tensor(encoded)
    start_positions = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in start_positions])
    y = torch.stack([data[i+1 : i+block_size+1] for i in start_positions])
    return x, y
