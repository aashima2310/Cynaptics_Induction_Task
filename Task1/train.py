import torch
import torch.nn.functional as F
import math
from src.data import load_data, get_batches
from src.model import GPT
import config

# load data
encoded, tokenizer, vocab_size = load_data('data/input.txt')

# split 90% train 10% val
split      = int(0.9 * len(encoded))
train_data = encoded[:split]
val_data   = encoded[split:]

# init model
model = GPT(
    vocab_size = config.VOCAB_SIZE,
    d_model    = config.D_MODEL,
    block_size = config.BLOCK_SIZE,
    n_heads    = config.N_HEADS,
    n_layers   = config.N_LAYERS
).to(config.DEVICE)

# optimizer and scheduler
optimizer     = torch.optim.AdamW(model.parameters(), lr=config.LR)
scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.MAX_ITERS)
best_val_loss = float('inf')

# training loop
for step in range(config.MAX_ITERS):
    model.train()
    x, y = get_batches(train_data, config.BLOCK_SIZE, config.BATCH_SIZE)
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    logits  = model(x)
    B, T, C = logits.shape
    loss    = F.cross_entropy(logits.view(B*T, C), y.view(B*T))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % config.EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batches(val_data, config.BLOCK_SIZE, config.BATCH_SIZE)
            x_val, y_val = x_val.to(config.DEVICE), y_val.to(config.DEVICE)
            val_logits   = model(x_val)
            B, T, C      = val_logits.shape
            val_loss     = F.cross_entropy(val_logits.view(B*T, C), y_val.view(B*T))
            perplexity   = math.exp(val_loss.item())

        print(f"step {step:4d} | train {loss.item():.4f} | val {val_loss.item():.4f} | ppl {perplexity:.2f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  → best model saved!")

print("Training complete!")
