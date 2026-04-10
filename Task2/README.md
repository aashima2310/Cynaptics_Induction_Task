# Task 2 — Supervised Fine-Tuning of GPT-2 on Alpaca

## Overview
Fine-tuning pretrained GPT-2 base (124M parameters) from HuggingFace
on the Stanford Alpaca dataset (~52K examples) to create an
instruction-following conversational assistant.

```
Pretraining  → learn language from scratch       (Task 1)
Fine-tuning  → adapt pretrained model to task    (Task 2)
```

---

## Project Structure
```
Task2/
├── DataLoader.py    ← dataset loading + prompt formatting
├── train.py         ← fine-tuning loop
├── generate.py      ← inference + interactive script
└── README.md
```

---

## Setup
```bash
pip install torch transformers datasets
```

## How to Fine-Tune
```bash
python train.py
```

## How to Generate
```bash
python generate.py
```

---

## Prompt Template

With input:
```
Below is an instruction that describes a task, paired with an input
that provides further context. Write a response that appropriately
completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

Without input:
```
Below is an instruction that describes a task. Write a response
that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
```

---

## Hyperparameters

| Parameter     | Value        | Explanation                    |
|---------------|--------------|--------------------------------|
| model         | GPT-2 (124M) | pretrained base model          |
| block_size    | 256          | max sequence length            |
| batch_size    | 16           | sequences per step             |
| epochs        | 2            | fine-tuning passes             |
| learning rate | 2e-5         | small lr preserves weights     |
| optimizer     | AdamW        | weight decay regularization    |
| scheduler     | Cosine       | smooth lr decay                |
| dataset       | tatsu-lab/alpaca | 52K instruction examples   |

---

## Training Results

| Step | Loss  |
|------|-------|
| 0    | 8.46  |
| 100  | 0.65  |
| 500  | 0.73  |
| 1000 | 0.56  |
| 2900 | 0.61  |

Final loss after all epochs : 
epoch 2 | avg train 0.6494 | avg val 0.6264

---

## Sample Outputs

**Instruction:** Explain what gravity is in simple terms
```
Response: Gravity is the force that exists between two objects
proportional to their masses and distance. It keeps us on the
ground and planets in orbit around the sun.
```

## Sample video

https://drive.google.com/file/d/132z5yvy7Pv3FAO6xPHeGwl1Ub1UiKPha/view?usp=sharing

---

## Key Concepts

**Why fine-tuning works:**
GPT-2 pretrained on 40GB of internet text already understands
English grammar and reasoning. Fine-tuning teaches it the
instruction-response format without relearning language.

**Why small learning rate (2e-5):**
Preserves pretrained weights. Too high = catastrophic forgetting.

**Why labels=batch:**
GPT-2 is causal LM — predicts next token. Same tensor as input
and labels lets model compute loss by shifting internally.
