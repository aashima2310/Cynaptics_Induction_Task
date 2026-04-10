import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def format_prompt(instruction, input_text=""):
    if input_text.strip():
        return (
            "Below is an instruction that describes a task, "
            "paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )


def generate(instruction, input_text="", max_new_tokens=200, temperature=0.7):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-alpaca')
    model     = GPT2LMHeadModel.from_pretrained('gpt2-alpaca').to(DEVICE)
    model.eval()

    prompt = format_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id
        )

    # decode only newly generated tokens
    generated = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
   tests = [
    ("Write a short poem about the moon", ""),
    ("Explain what gravity is in simple terms", ""),
    ("What are the benefits of exercise?", ""),
    ("Summarize this text",
     "The sun is a star at the center of our solar system."),
    ("Give 5 tips to stay productive.")
]

for instruction, input_text in tests:
     print(f"\nInstruction: {instruction}")
     if input_text:
         print(f"Input      : {input_text}")
         print(f"Response   : {generate(instruction, input_text)}")
         print("-" * 50)
