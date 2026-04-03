import torch
from transformers import GPT2Tokenizer

def load_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    # load pretrained GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # encode full text
    encoded = tokenizer.encode(text)
    vocab_size = tokenizer.vocab_size

    return encoded, tokenizer, vocab_size


def get_batches(encoded, block_size, batch_size):
    data = torch.tensor(encoded)
    start_positions = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in start_positions])
    y = torch.stack([data[i+1 : i+block_size+1] for i in start_positions])
    return x, y