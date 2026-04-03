import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from DataLoader import load_alpaca_dataset

# hyperparameters
BLOCK_SIZE  = 512
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 2e-5
EVAL_EVERY  = 100
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# load formatted dataset
print("Loading dataset...")
train_data = load_alpaca_dataset(split="train")
val_data   = load_alpaca_dataset(split="test")

# load pretrained GPT-2
print("Loading GPT-2...")
tokenizer             = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token   = tokenizer.eos_token
model                 = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').to(DEVICE)

print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# tokenize dataset
def tokenize(dataset):
    all_ids = []
    for example in dataset:
        encoded = tokenizer(
            example['text'],
            truncation    = True,
            max_length    = BLOCK_SIZE,
            padding       = 'max_length',
            return_tensors= 'pt'
        )
        all_ids.append(encoded['input_ids'].squeeze(0))
    return torch.stack(all_ids)

print("Tokenizing train data...")
train_ids = tokenize(train_data)
print("Tokenizing val data...")
val_ids   = tokenize(val_data)

# create dataloaders
train_loader = DataLoader(TensorDataset(train_ids), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_ids),   batch_size=BATCH_SIZE)

# optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS * len(train_loader))

best_val_loss = float('inf')

# fine-tuning loop
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for step, (batch,) in enumerate(train_loader):
        batch  = batch.to(DEVICE)
        output = model(input_ids=batch, labels=batch)
        loss   = output.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        if step % EVAL_EVERY == 0:
            print(f"epoch {epoch+1} | step {step:4d} | loss {loss.item():.4f}")

    # validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for (batch,) in val_loader:
            batch          = batch.to(DEVICE)
            output         = model(input_ids=batch, labels=batch)
            total_val_loss += output.loss.item()

    avg_train = total_train_loss / len(train_loader)
    avg_val   = total_val_loss   / len(val_loader)
    print(f"\nepoch {epoch+1} | avg train {avg_train:.4f} | avg val {avg_val:.4f}\n")

    # save best model
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        model.save_pretrained('gpt2-alpaca')
        tokenizer.save_pretrained('gpt2-alpaca')
        print(f"  → best model saved!")

print("Training complete!")
