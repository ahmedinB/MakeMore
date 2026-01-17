# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 01:42:35 2025

@author: A
"""

# this is a tiny GPT model 

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
learning_rate = 3e-4
torch.manual_seed(1337)
context_size = 8
batch_size = 4
n_embd = 32
dropout = 0.2

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


class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**(-0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim= -1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
         
        return out

# =============================================================================
# head run 1
# Eval epoc 499  loss 2.#684253692626953
# Eval epoc 999  loss 2.613590955734253
# Eval epoc 1499  loss 2.3154666423797607
# Eval epoc 1999  loss 2.4868252277374268
# Eval epoc 2499  loss 2.464097738265991
# Eval epoc 2999  loss 2.4927334785461426
# Eval epoc 3499  loss 2.275904417037964
# Eval epoc 3999  loss 2.455901861190796
# Eval epoc 4499  loss 2.3168857097625732
# Eval epoc 4999  loss 2.4807181358337402
# 
# Wa theye hant
# In; irken turiemin ot
# Tes. I thy, wiove wow sheat ghed yol buratigon hofr'd ghanve hor
# =============================================================================

# =============================================================================
# run 2
# Eval epoc 499  loss 2.684253692626953
# Eval epoc 999  loss 2.613590955734253
# Eval epoc 1499  loss 2.3154666423797607
# Eval epoc 1999  loss 2.4868252277374268
# Eval epoc 2499  loss 2.464097738265991
# Eval epoc 2999  loss 2.4927334785461426
# Eval epoc 3499  loss 2.275904417037964
# Eval epoc 3999  loss 2.455901861190796
# Eval epoc 4499  loss 2.3168857097625732
# Eval epoc 4999  loss 2.4807181358337402
# 
# Wa theye hant
# In; irken turiemin ot
# Tes. I thy, wiove wow sheat ghed yol buratigon hofr'd ghanve hor
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out  
    
        
    
# =============================================================================
# Multi Head Attention run
# Eval epoc 499  loss 2.6202938556671143
# Eval epoc 999  loss 2.5890953540802
# Eval epoc 1499  loss 2.2738037109375
# Eval epoc 1999  loss 2.405195474624634
# Eval epoc 2499  loss 2.400883197784424
# Eval epoc 2999  loss 2.3948771953582764
# Eval epoc 3499  loss 2.22015380859375
# Eval epoc 3999  loss 2.350111961364746
# Eval epoc 4499  loss 2.181793689727783
# Eval epoc 4999  loss 2.309291362762451
# 
# Wa theye, bous all
# Srent?
# 
# Temiorot
# Tes.
# 
# EN:
# ag's the ow shea agund youre-; tareng, Co'
# Wal ove hou
# =============================================================================
    


class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), 
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
            )
    def forward(self, x):
        return self.net(x)
    
# =============================================================================
# FeedForward run
# Eval epoc 499  loss 2.5791025161743164
# Eval epoc 999  loss 2.5848851203918457
# Eval epoc 1499  loss 2.2409965991973877
# Eval epoc 1999  loss 2.359457015991211
# Eval epoc 2499  loss 2.38368821144104
# Eval epoc 2999  loss 2.3437695503234863
# Eval epoc 3499  loss 2.1599738597869873
# Eval epoc 3999  loss 2.345202684402466
# Eval epoc 4499  loss 2.1283066272735596
# Eval epoc 4999  loss 2.3344078063964844
# 
# Wa theye hant
# In; irrent?-
# Tem
# Thound know thy g
# =============================================================================
 
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x +  self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# =============================================================================
# Block - not residual learning
# Eval epoc 499  loss 3.095425605773926
# Eval epoc 999  loss 2.8292391300201416
# Eval epoc 1499  loss 2.5278396606445312
# Eval epoc 1999  loss 2.610002279281616
# Eval epoc 2499  loss 2.596346855163574
# Eval epoc 2999  loss 2.6077661514282227
# Eval epoc 3499  loss 2.3445701599121094
# Eval epoc 3999  loss 2.4707343578338623
# Eval epoc 4499  loss 2.2884631156921387
# Eval epoc 4999  loss 2.4743802547454834
# 
# Waf heye hant
# Inwerr in that Rin otndes. I thy gishete of shear gund you burl agen hor sum banv-We b
# =============================================================================
# =============================================================================
# Blocks with residual learning
# Eval epoc 499  loss 2.382868766784668
# Eval epoc 999  loss 2.394176721572876
# Eval epoc 1499  loss 1.9831843376159668
# Eval epoc 1999  loss 2.1649539470672607
# Eval epoc 2499  loss 2.193105697631836
# Eval epoc 2999  loss 2.069242477416992
# Eval epoc 3499  loss 1.9237247705459595
# Eval epoc 3999  loss 2.176154851913452
# Eval epoc 4499  loss 1.854046106338501
# Eval epoc 4999  loss 2.060460090637207
# 
# Wad here hand
# In; in he that Rive
# Bram of blesag's the of she the nigence I and of horr'd grant-fapr
# =============================================================================

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
# =============================================================================
#         self.sa_head = Head(n_embd)
# =============================================================================
# =============================================================================
#         self.sa_head = MultiHeadAttention(4, n_embd//4)
# =============================================================================
# =============================================================================
#         self.ffwd = FeedForward(n_embd)
# =============================================================================
        self.blocks = nn.Sequential(
            Block(n_embd, n_head= 4),
            Block(n_embd, n_head= 4),
            Block(n_embd, n_head= 4),
            nn.LayerNorm(n_embd)
            )
        
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
# =============================================================================
#         x = self.sa_head(x)
#         x = self.ffwd(x)
# =============================================================================
        x = self.blocks(x)
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
    if (steps+1) % eval_iter == 0:
        print("Eval epoc", steps, " loss", loss.item())
    
    
    
    
print(decode(m.generate(torch.zeros((1,1), dtype= torch.long, device = device), max_new_tokens= 100)[0].tolist()))
    
    
    

