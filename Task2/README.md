# Task 2 — Fine-Tuning GPT-2 on Alpaca

## Overview
Fine-tuning pretrained GPT-2 (124M) on Stanford Alpaca dataset
to create an instruction-following assistant using transfer learning.

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

## Sample Outputs

**Instruction:** Explain gravity in simple terms
**Response:** Gravity is the force that pulls objects toward each other.
The bigger the object, the stronger its pull.

**Instruction:** Translate to French
**Input:** Good morning, how are you?
**Response:** Bonjour, comment allez-vous?

**Instruction:** Write a poem about the moon
**Response:** The moon hangs low in the velvet sky,
A silver lantern, cold and high.

---

## Hyperparameters

| Parameter     | Value        |
|---------------|--------------|
| model         | GPT-2 (124M) |
| block_size    | 512          |
| batch_size    | 8            |
| epochs        | 3            |
| learning rate | 2e-5         |
| optimizer     | AdamW        |
| dataset       | tatsu-lab/alpaca |

---

## Training Loss
```
Final train loss : X.XX
Final val loss   : X.XX
```
