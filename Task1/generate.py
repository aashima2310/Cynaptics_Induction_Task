import torch
import torch.nn.functional as F
from src.data import load_data
from src.model import GPT
import config

def generate(model, tokenizer, seed_text, max_tokens=500):
    encoded = tokenizer.encode(seed_text)
    x = torch.tensor(encoded).unsqueeze(0).to(config.device)

    for _ in range(max_tokens):
        x_crop = x[:, -config.block_size:]
        logits  = model(x_crop)
        logits  = logits[:, -1, :]
        probs   = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    decoded = tokenizer.decode(x[0].tolist())
    return decoded


if __name__ == "__main__":
    encoded, tokenizer, vocab_size = load_data('data/input.txt')

    model = GPT(
        vocab_size = config.vocab_size,
        d_model    = config.d_model,
        block_size = config.block_size,
        n_heads    = config.n_heads,
        n_layers   = config.n_layers
    ).to(config.device)

    model.load_state_dict(torch.load('best_model.pt', map_location=config.device))
    model.eval()

    output = generate(model, tokenizer, "To be", max_tokens=500)
    print(output)