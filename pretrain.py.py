import torch
import torch.nn as nn
import torch.optim as optim

from model.transformer import TransformerBlock

class SimpleGPT(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = TransformerBlock(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        x = self.embedding(x)
        x = self.transformer(x)
        logits = self.fc(x)

        return logits


def train():

    vocab_size = 100
    embed_size = 64

    model = SimpleGPT(vocab_size, embed_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    dummy_input = torch.randint(0, vocab_size, (4, 10))
    dummy_target = torch.randint(0, vocab_size, (4, 10))

    for epoch in range(10):

        optimizer.zero_grad()

        output = model(dummy_input)

        loss = loss_fn(output.view(-1, vocab_size), dummy_target.view(-1))

        loss.backward()
        optimizer.step()

        print("Epoch:", epoch, "Loss:", loss.item())


if __name__ == "__main__":
    train()