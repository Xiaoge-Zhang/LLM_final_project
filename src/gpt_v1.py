import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time

def get_batch(split):
    data = trian_data if split == 'train' else val_data
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

    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # settings and hyperparameters
    training_ratio = 0.8
    block_size = 64
    batch_size = 128
    learning_rate = 3e-4
    max_iteration = 3000
    eval_iterations = 100
    n_embd = 384
    n_head = 8
    n_blocks = 10
    dropout = 0.2
    ffoward_inner_size = 4

    save_name = 'wizard_of_oz_model'
    input_file_name = 'wizard_of_oz.txt'
    prompt = 'Hello! Can you see me?'

    training = True

    # read in the trainining data
    with open('../data/{}'.format(input_file_name), 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(device)

    # letter-wise tokenizer
    string_to_int = {character: i for i, character in enumerate(chars)}
    int_to_string = {i: character for i, character in enumerate(chars)}
    encode = lambda s: [string_to_int[c] for c in s]
    decode = lambda l: ''.join([int_to_string[i] for i in l])

    # training and testing split
    data = torch.tensor(encode(text), dtype=torch.long)
    training_size = int(training_ratio * len(data))
    trian_data = data[:training_size]
    val_data = data[training_size:]

    # create our model
    model = GPTLanguageModel(vocab_size)
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if training:
        # beginning of the training iterations
        for iter in range(max_iteration):
            if iter % eval_iterations == 0 or iter == max_iteration-1:
                losses = estimate_loss()
                print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

            # sample a batch of data
            xb, yb = get_batch('train')

            # evaluate the loss
            logits, loss = model.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # save the model
        torch.save(m.state_dict(), '../saved_model/{}.pth'.format(save_name))
        # generate chars based on the query context
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
        print('prompt: ', prompt)
        print('anwser from the model: ', generated_chars[len(prompt):])

    else:
        # Load the trained model and anwer the prompt
        model = GPTLanguageModel(vocab_size).to(device)
        checkpoint_path = '../saved_model/{}.pth'.format(save_name)  # Path to the saved model
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        # generate chars based on the query context
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
        print('prompt: ', prompt)
        print('anwser from the model: ', generated_chars[len(prompt):])