from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(file_path, vocab_size=1000):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files      = [file_path],
        vocab_size = vocab_size,
        min_frequency = 2
    )
    tokenizer.save_model('.', 'shakespeare')
    print(f"Tokenizer trained! vocab size = {vocab_size}")
    return tokenizer

def load_tokenizer():
    tokenizer = ByteLevelBPETokenizer(
        'shakespeare-vocab.json',
        'shakespeare-merges.txt'
    )
    return tokenizer

if __name__ == "__main__":
    train_tokenizer('data/input.txt', vocab_size=1000)
