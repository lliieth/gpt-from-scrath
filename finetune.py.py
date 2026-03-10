import torch
import torch.nn as nn
import torch.optim as optim

from training.pretrain import SimpleGPT


def finetune():

    vocab_size = 100
    embed_size = 64

    model = SimpleGPT(vocab_size, embed_size)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    dummy_input = torch.randint(0, vocab_size, (4, 10))
    dummy_target = torch.randint(0, vocab_size, (4, 10))

    for epoch in range(5):

        optimizer.zero_grad()

        output = model(dummy_input)

        loss = loss_fn(output.view(-1, vocab_size), dummy_target.view(-1))

        loss.backward()
        optimizer.step()

        print("Finetune Epoch:", epoch, "Loss:", loss.item())


if __name__ == "__main__":
    finetune()