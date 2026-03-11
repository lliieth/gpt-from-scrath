class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def train(self, text):
        tokens = text.split()
        unique_tokens = sorted(set(tokens))
        self.vocab = {tok: i for i, tok in enumerate(unique_tokens)}
        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(tok, 0) for tok in text.split()]

    def decode(self, ids):
        return " ".join(self.inverse_vocab.get(i, "<unk>") for i in ids)
