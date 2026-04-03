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
To be an air abused
Sixt the commission.

First Servingman:
He's to bear them.

Second Servingman:
Sir, I pray, I think 'twas grieve no less.

First Servingman:
She shall be so so: 'tis led as three sins yet
uspicensied in his steal.

Second Servingman:
Binheees:
But in their dogs tempt, and dartell your general;
And not the Volsces are bared on
An over sound a blows for you;
For you, mistress, as I will go hunt
In your cannot come.

First Servingman:
I would endure it were as it was a groan,
His fault but two years a sisterhority;
So true, that therefore I have stay to you.

First Gentleman:
Who'll not till you do, money?

Second Watchman:
Hapster no house of you?

Second Servingman:
My lord, ere she looks upon me
Your hungry's apparet.

CORIOLANUS:
This is as your limitiest woman
That Angelo; let you be battle.

CORIOLANUS:
I tongues, consudges, credit no first.

CORIOLANUS:
Ay, do feel 't.

CORIOLANUS:
I should not wrong aundred, for it.

CORIOLANUS:
That will not so contrive: I
From love a grave as I jot upon thee,
As I have poison'd my hand of my ancient:
So was breathed; though crown'd, man
Of my letters adversition, when 'tis
To stole jealous queen, and my highness cannibble.

First Citizen:
No, he's one
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

## Loss on training
step 9600 | train 2.4423 | val 3.5417 | ppl 34.53
step 9800 | train 2.4648 | val 3.4633 | ppl 31.92
Training complete!

training loss = 2.4648
val_loss = 3.4633
