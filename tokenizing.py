class CharTokenizer:
    def __init__(self, texts, indent_spaces=4):
        self.indent_spaces = indent_spaces

        # Special tokens (fixed IDs)
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<indent>"] # padding, beginning of sequence, end of sequence, indent
        self.stoi = {tok: i for i, tok in enumerate(self.special_tokens)} # map string to index
        self.itos = {i: tok for tok, i in self.stoi.items()} # map index to string

        # Collect characters
        chars = set() # set of unique characters
        for text in texts: 
            chars.update(text) # add all of the unique characters

        # Assign IDs
        offset = len(self.stoi) # offset by 4, ie numbers taken up by special tokens already
        for i, ch in enumerate(sorted(chars)): 
            self.stoi[ch] = i + offset # map the characters to index
            self.itos[i + offset] = ch # reverse map

        self.vocab_size = len(self.stoi)

    def encode(self, text, add_special_tokens=True):
        ids = []

        if add_special_tokens:
            ids.append(self.stoi["<bos>"])

        i = 0
        while i < len(text):
            # Handle indentation (only at line start)
            if text[i] == " ":
                count = 0
                while i < len(text) and text[i] == " ":
                    count += 1
                    i += 1
                # you kinda reverse engineer from the amount of spaces how many indents there are

                while count >= self.indent_spaces: # when count bigger than 4 it counts as an indent
                    ids.append(self.stoi["<indent>"]) # add token for indent
                    count -= self.indent_spaces # reduce count by 4

                # leftover spaces
                ids.extend([self.stoi[" "]] * count) # add remaining spaces as formatting spaces basically
            else:
                ids.append(self.stoi[text[i]])
                i += 1

        if add_special_tokens:
            ids.append(self.stoi["<eos>"])

        return ids

    def decode(self, ids):
        text = "" #initialize string
        for i in ids:
            token = self.itos.get(i, "") # get the token from ids (the index)
            if token == "<bos>" or token == "<eos>" or token == "<pad>":
                continue
            elif token == "<indent>":
                text += " " * self.indent_spaces #. add 4 spaces if there was an indent
            else:
                text += token #just add the token to the string
        return text


import pandas as pd
df = pd.read_csv("code_bug_fix_pairs.csv")

import re
def clean_code_logic(text):
    if not isinstance(text, str):
            return ""

    marker = "# Sample ID"
    index = text.find(marker)

    if index == -1:
        return text.strip()

    return text[:index].strip()

# --- Step 1: Clean the DataFrame first ---
print("Cleaning data and building custom vocabulary...")

df['buggy_clean'] = df['buggy_code'].apply(clean_code_logic)
df['fixed_clean'] = df['fixed_code'].apply(clean_code_logic)

# --- Step 2: Gather cleaned tokens into a list ---
# Using .tolist() is much faster than iterrows()
texts = df['buggy_clean'].tolist() + df['fixed_clean'].tolist()

print(f"Collected {len(texts)} cleaned code snippets.")



tokenizer = CharTokenizer(texts)

print("Vocab size:", tokenizer.vocab_size)
print(list(tokenizer.stoi.items()))


sample = df.iloc[0]["buggy_clean"]
print("ORIGINAL:")
print(repr(sample))

encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("ENCODED:")
print(encoded[:50])  # print first 50 tokens
print("DECODED:")
print(repr(decoded))


assert decoded == sample
print("Yippee reversiblityy")

import torch
from torch.utils.data import Dataset



class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range (0,len(token_ids) - max_length, stride):
           input_chunk = token_ids[i:i+max_length]
           target_chunk = token_ids[i+1:i+max_length+1]

           self.input_ids.append(torch.tensor(input_chunk))
           self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt,batch_size = 4,max_length =256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer = CharTokenizer(texts)

    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader

dataloader = create_dataloader_v1(df.iloc[0]["buggy_clean"], batch_size=8, max_length=4, stride=4,shuffle=False,drop_last=False,num_workers=0,)

data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("inputs\n",inputs)
print("targets\n",targets)

vocab_size = tokenizer.vocab_size
output_dim = 256 * 2

torch.manual_seed(123)
import torch
import torch.nn as nn

# --- Step 1: Embedding layer ---
output_dim = 512  # embedding dimension
embedding_layer = nn.Embedding(
    num_embeddings=tokenizer.vocab_size,
    embedding_dim=output_dim,
    padding_idx=tokenizer.stoi["<pad>"]  # ignores <pad> token in training
)

# --- Step 2: Simple Transformer block ---
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ff_hidden_dim=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)  # to (seq_len, batch_size, embed_dim) for attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x.transpose(0, 1)  # back to (batch_size, seq_len, embed_dim)

# --- Step 3: Test embeddings with Transformer ---
max_length = 4
dataloader = create_dataloader_v1(
    df.iloc[0]["buggy_clean"],
    batch_size=4,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("inputs\n", inputs)
print("targets\n", targets)

# Get token embeddings
token_embeddings = embedding_layer(inputs)
print("Token embeddings shape:", token_embeddings.shape)  # (batch_size, seq_len, embed_dim)

# Apply Transformer
transformer_block = SimpleTransformerBlock(embed_dim=output_dim)
transformer_out = transformer_block(token_embeddings)
print("Transformer output shape:", transformer_out.shape)  # same shape as embeddings


# --- Step 4: Add next-token prediction head ---
class SimpleCodeSLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads=8, ff_hidden_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.stoi["<pad>"])
        self.transformer = SimpleTransformerBlock(embed_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)  # output logits for each token

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        x = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        x = self.transformer(x)        # (batch_size, seq_len, embed_dim)
        logits = self.fc_out(x)        # (batch_size, seq_len, vocab_size)
        return logits

# --- Step 5: Instantiate model ---
vocab_size = tokenizer.vocab_size
embed_dim = 512
model = SimpleCodeSLM(vocab_size, embed_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Step 6: Simple forward + loss example ---
inputs, targets = next(iter(dataloader))
logits = model(inputs)  # (batch_size, seq_len, vocab_size)

# For CrossEntropyLoss we need (batch_size*seq_len, vocab_size) and targets as (batch_size*seq_len)
loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
print("Initial loss:", loss.item())

# --- Step 7: Backward pass example ---
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Step done. Model updated.")


import torch
from torch.utils.data import DataLoader

# --- Parameters ---
batch_size = 8
max_length = 64   # sequence length
stride = 32       # overlapping stride
epochs = 3        # small number for testing
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Step 1: Combine all cleaned code into one big string ---
all_text = "\n".join(df['buggy_clean'].tolist() + df['fixed_clean'].tolist())

# --- Step 2: Create DataLoader ---
dataloader = create_dataloader_v1(
    all_text, 
    batch_size=batch_size, 
    max_length=max_length, 
    stride=stride,
    shuffle=True
)

# --- Step 3: Instantiate model ---
vocab_size = tokenizer.vocab_size
embed_dim = 512
model = SimpleCodeSLM(vocab_size, embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# --- Step 4: Training loop ---
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (batch_size, seq_len, vocab_size)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed | Average loss: {total_loss / len(dataloader):.4f}")

print("Training finished!")


def generate_fixed_code(model, tokenizer, buggy_code, max_gen_len=200, device="cpu"):
    """
    Generate a suggested fixed code snippet from buggy code.
    """
    model.eval()
    with torch.no_grad():
        # Encode the input
        input_ids = torch.tensor(tokenizer.encode(buggy_code)).unsqueeze(0).to(device)  # (1, seq_len)
        generated_ids = input_ids.clone()  # start with input
        
        for _ in range(max_gen_len):
            logits = model(generated_ids)  # (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # take the last token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # greedy
            if next_token_id.item() == tokenizer.stoi["<eos>"]:
                break  # stop at end-of-sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        # Decode
        output_code = tokenizer.decode(generated_ids.squeeze().tolist())
        return output_code
    

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example buggy snippet
buggy_snippet = """
def add_numbers(a, b):
print(a + b)
"""

fixed_suggestion = generate_fixed_code(model, tokenizer, buggy_snippet, max_gen_len=100, device=device)
print("=== BUGGY CODE ===")
print(buggy_snippet)
print("=== MODEL SUGGESTION ===")
print(fixed_suggestion)


print("suggestion complete")