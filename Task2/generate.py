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


def generate(model, tokenizer, instruction, input_text="", max_new_tokens=200, temperature=0.7):
    prompt = format_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_new_tokens      = max_new_tokens,
            temperature         = temperature,
            do_sample           = True,
            pad_token_id        = tokenizer.eos_token_id,
            repetition_penalty  = 1.3,
            no_repeat_ngram_size= 3,
        )

    generated = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def save_outputs(model, tokenizer, tests, filename='sample_outputs.txt'):
    with open(filename, 'w') as f:
        for instruction, input_text in tests:
            response = generate(model, tokenizer, instruction, input_text)
            f.write(f"Instruction: {instruction}\n")
            if input_text:
                f.write(f"Input: {input_text}\n")
            f.write(f"Response: {response}\n")
            f.write("-" * 50 + "\n")
            print(f"Instruction: {instruction}")
            print(f"Response: {response}")
            print("-" * 50)
    print(f"\nOutputs saved to {filename}!")


def interactive_mode(model, tokenizer):
    print("\n" + "="*50)
    print("GPT-2 Alpaca — Interactive Mode")
    print("Type 'quit' to exit")
    print("="*50 + "\n")

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() == 'quit':
            print("Goodbye!")
            break
        input_text = input("Input (press Enter to skip): ").strip()
        print("\nGenerating response...")
        response = generate(model, tokenizer, instruction, input_text)
        print(f"\nResponse: {response}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    print("Loading model...")
    tokenizer             = GPT2Tokenizer.from_pretrained('gpt2-alpaca')
    tokenizer.pad_token   = tokenizer.eos_token
    model                 = GPT2LMHeadModel.from_pretrained('gpt2-alpaca').to(DEVICE)
    model.eval()
    print("Model loaded!\n")

    tests = [
        ("Write a short poem about the moon", ""),
        ("Explain what gravity is in simple terms", ""),
        ("What are the benefits of exercise?", ""),
        ("Summarize this text",
         "The sun is a star at the center of our solar system."),
        ("Give me 3 tips for better sleep", ""),
    ]

    save_outputs(model, tokenizer, tests)

    interactive_mode(model, tokenizer)
