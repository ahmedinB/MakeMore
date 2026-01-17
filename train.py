# this is a simple bigram model 

with open ('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
   
# print('length of dataset :', len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

# print("".join(chars))
# print(vocab_size)
    
stoi = { ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: "".join([itos[c] for c in s])

# print(encode("Hi There"))
# print(decode([20, 47, 1, 32, 46, 43, 56, 43]))

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.tensor(encode(text), dtype=torch.long, device = device)
# print(data.shape)

n = int(len(data)*0.9)
train_data = data[:n]
test_data = data[n:]

eval_iter = 500
max_iters = 5000
learning_rate = 1e-3
torch.manual_seed(1337)
context_size = 8
batch_size = 4
n_embd = 32

def get_batch(split):
    
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - context_size, (batch_size,), device = device)
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    
    return x,y


xb, yb = get_batch("train")
# print(xb)
# print(yb)
    

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        
        if targets == None:
            loss = None
        else:
                        
            B, T, C = logits.shape
            
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -context_size:]
            
            logits, loss = self(idx_cond)
            
            logits = logits[:,-1,:] # becomes [B C]
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=-1)
        
        return idx


    
m = BigramLanguageModel().to(device)
logits, loss = m(xb, yb)

# print(logits.shape)
# print(loss)
    
# idx = torch.zeros((1,1), dtype= torch.long)
# print("".join(decode(m.generate(idx, max_new_tokens= 100)[0].tolist())))
    
    
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

batch_size = 32

for steps in range (max_iters):
    xb, yb = get_batch("train")
    
    logits, loss = m(xb, yb) 
    optimizer.zero_grad(set_to_none= True)
    
    loss.backward()
    optimizer.step()
    if steps % eval_iter == 0:
        print("Eval epoc", steps, " loss", loss.item())
    
    
    
print(decode(m.generate(torch.zeros((1,1), dtype= torch.long, device = device), max_new_tokens= 100)[0].tolist()))
    
    
# Eval epoc 0  loss 4.531844615936279
# Eval epoc 500  loss 2.6418449878692627
# Eval epoc 1000  loss 2.582864284515381
# Eval epoc 1500  loss 2.558906078338623
# Eval epoc 2000  loss 2.478076219558716
# Eval epoc 2500  loss 2.5636518001556396
# Eval epoc 3000  loss 2.4831295013427734
# Eval epoc 3500  loss 2.474890947341919
# Eval epoc 4000  loss 2.5053882598876953
# Eval epoc 4500  loss 2.    `

# Wa theye h nt
# In; irken turiey
# F ot

# WA. hit fag'sovee owas d t g nd y le I ataronghore'
# Wal ove y b

