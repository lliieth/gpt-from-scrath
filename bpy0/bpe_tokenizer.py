class BPETokenizer:
    def __init__(self):
        self.vocab = {}

    def train(self, text):
        tokens = text.split()
        self.vocab = {w:i for i,w in enumerate(set(tokens))}

    def encode(self, text):
        return [self.vocab.get(w,0) for w in text.split()]

    def decode(self, ids):
        inv = {v:k for k,v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)
