import mmap
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import pandas as pd


def plot_loss_curves():
    # save the loss curve
    # Create the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(iterations, training_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(iterations, testing_losses, label='Testing Loss', color='orange', linestyle='--', marker='x')

    # Add labels, title, and legend
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses Over Iterations')
    plt.legend()

    # Show grid
    plt.grid(True)

    # Save the plot as a PDF
    plt.savefig('../visualization/training_losses_{}.pdf'.format(save_name), format='pdf')

    # Display the plot
    plt.show()

def get_random_block(split):
    file_name = input_file_dir + 'open_web_train.txt' if split == 'train' else input_file_dir + 'open_web_test.txt'
    with open(file_name, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            block_decode = block.decode('utf-8', errors='ignore').replace('\r', '')

            data = torch.tensor(encode(block_decode), dtype=torch.long)

    return data


def get_batch(split):
    data = get_random_block(split)
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # to device
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        qk_result = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        qk_result = qk_result.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        qk_result = F.softmax(qk_result, dim=-1)
        qk_result = self.dropout(qk_result)

        value = self.value(x)
        output = qk_result @ value
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)

        return output


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffoward_inner_size * n_embd),
            nn.ReLU(),
            nn.Linear(ffoward_inner_size * n_embd, n_embd),
            nn.Dropout(dropout),)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.ffoward = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual
        y = self.attention(x)
        x = self.ln1(x + y)
        y = self.ffoward(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_blocks)])

        self.final_layer = nn.LayerNorm(n_embd)
        self.output_head = nn.Linear(n_embd, vocab_size)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
            B, T = index.shape
            token_embd = self.token_embedding_table(index)
            pos_embd = self.position_embedding_table(torch.arange(T, device=device))
            x = token_embd + pos_embd
            x = self.blocks(x)
            x = self.final_layer(x)
            logits  = self.output_head(x)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


if __name__ == '__main__':
    # time the program
    start_time = time.time()

    # data for the plot
    training_losses = []
    testing_losses = []
    iterations = []
    last_iter = 0

    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # settings and hyperparameters
    block_size = 64
    batch_size = 128
    learning_rate = 3e-4
    max_iteration = 1000
    eval_iterations = 100
    n_embd = 384
    n_head = 8
    n_blocks = 8
    dropout = 0.2
    ffoward_inner_size = 4

    model_name = 'open_web'
    loss_file_dir = '../model_losses/' + model_name + '_losses.csv'
    input_file_dir = '../processed_data/'
    save_name = '{}_model'.format(model_name)

    print(device)

    chars = ''
    with open(input_file_dir + '{}_vocab.txt'.format(model_name), 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))

    vocab_size = len(chars)

    # letter-wise tokenizer
    string_to_int = {character: i for i, character in enumerate(chars)}
    int_to_string = {i: character for i, character in enumerate(chars)}
    encode = lambda s: [string_to_int[c] for c in s]
    decode = lambda l: ''.join([int_to_string[i] for i in l])

    # create our model
    model = GPTLanguageModel(vocab_size)

    # check if we have trained model
    if os.path.exists('../saved_model/' + save_name + '.pkl'):
        # load model parameters
        print('found existing model parameter. loading now...')
        with open('../saved_model/' + save_name + '.pkl', 'rb') as f:
            model = pickle.load(f)
        print('model loaded!')

        # load losses
        saved_loss_df = pd.read_csv(loss_file_dir)
        iterations = saved_loss_df['iterations'].tolist()
        last_iter = iterations[-1] + 1
        training_losses = saved_loss_df['train_losses'].tolist()
        testing_losses = saved_loss_df['test_losses'].tolist()

    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # beginning of the training iterations
    for iter in range(max_iteration):
        if iter % eval_iterations == 0 or iter == max_iteration-1:
            losses = estimate_loss()
            print(f"step: {iter + last_iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
            training_losses.append(losses['train'].item())
            testing_losses.append(losses['val'].item())
            iterations.append(iter + last_iter)

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save the model
    print('saving the model')
    with open('../saved_model/' + save_name + '.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('model saved')

    # show the time of training
    end_time = time.time()
    print('time takes to train {}: '.format(save_name), end_time - start_time)

    # save the losses to csv
    loss_dict = {'iterations': iterations, 'train_losses': training_losses, 'test_losses': testing_losses}
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(loss_file_dir, index=False)

    # plot and save the visualization of training:
    plot_loss_curves()