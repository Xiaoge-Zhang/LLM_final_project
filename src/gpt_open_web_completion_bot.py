import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os


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


# select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# settings and hyperparameters, should stay the same with gpt_open_web.py
block_size = 64
batch_size = 128
n_embd = 384
n_head = 8
n_blocks = 8
dropout = 0.2
ffoward_inner_size = 4

# determine the vocab size
model_name = 'open_web'
max_completion_length = 150

vocab_file = '../processed_data/' + model_name + '_vocab.txt'
model_dir = '../saved_model/{}_model.pkl'.format(model_name)

# read in the vocab
chars = ''
with open(vocab_file, 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

# tokenizer
string_to_int = {character: i for i, character in enumerate(chars)}
int_to_string = {i: character for i, character in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# create model
model = GPTLanguageModel(vocab_size)

# check if we have trained model
if os.path.exists(model_dir):
    # load model parameters
    print('found existing model parameter. loading now...')
    with open(model_dir, 'rb') as f:
        model = pickle.load(f)
    print('model loaded!')
else:
    exit('no model found!')

# start the chatbot
m = model.to(device)

# tokenizer
string_to_int = {character: i for i, character in enumerate(chars)}
int_to_string = {i: character for i, character in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=max_completion_length)[0].tolist())
    print(f'Completion:\n{generated_chars}')
