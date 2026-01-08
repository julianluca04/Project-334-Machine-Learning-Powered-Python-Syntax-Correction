import pandas as pd
import re
import torch
from collections import Counter

# 1. Configuration
FILE_PATH = 'code_bug_fix_pairs.csv'
MAX_LENGTH = 128 

# 2. Setup
df = pd.read_csv(FILE_PATH)

def clean_code_logic(text):
    """Removes the # Sample ID comment and everything after it."""
    if not isinstance(text, str): return ""
    return re.sub(r'\s*#\s*Sample ID.*', '', text, flags=re.IGNORECASE | re.DOTALL).strip()

def custom_tokenize(text):
    """
    Tokenizes text and replaces invisible whitespace with readable symbols.
    """
    if not isinstance(text, str): return []
    
    # Define the symbols
    NWLN = "[NWLN]"  # Symbol for Newline
    INDT = "[INDENT]" # Symbol for 4-space indent
    SPC = "[SPC]"    # Symbol for a single space
    
    # Pattern looks for: Newlines, 4-spaces, single spaces, words, or symbols
    pattern = r"\n| {4}| |[\w']+|[^\w\s]"
    raw_tokens = re.findall(pattern, text)
    
    # Swap the invisible characters for our readable symbols
    mapped_tokens = []
    for t in raw_tokens:
        if t == "\n":
            mapped_tokens.append(NWLN)
        elif t == "    ":
            mapped_tokens.append(INDT)
        elif t == " ":
            mapped_tokens.append(SPC)
        else:
            mapped_tokens.append(t)
            
    return mapped_tokens

# --- Step 1: Clean and Gather Tokens ---
print("Building custom vocabulary with special symbols...")
df['buggy_clean'] = df['buggy_code'].apply(clean_code_logic)
df['fixed_clean'] = df['fixed_code'].apply(clean_code_logic)

all_tokens = []
for col in ['buggy_clean', 'fixed_clean']:
    for cell in df[col]:
        all_tokens.extend(custom_tokenize(cell))

# Get unique tokens and build the vocabulary
unique_tokens = sorted(list(set(all_tokens)))
vocab_list = ["<PAD>", "<UNK>"] + unique_tokens
token_to_id = {token: i for i, token in enumerate(vocab_list)}

# --- Step 2: Save Vocabulary Lookup CSV ---
# Now your vocabulary CSV will actually show [NWLN] and [INDENT]
vocab_df = pd.DataFrame(vocab_list, columns=['Token']).reset_index().rename(columns={'index': 'ID'})
vocab_df = vocab_df[['ID', 'Token']]
vocab_df.to_csv('custom_vocab_symbols.csv', index=False)
print("Success: 'custom_vocab_symbols.csv' created.")

# --- Step 3: Convert to Numeric Lists for .pt File ---
def encode_and_pad(text, max_len):
    tokens = custom_tokenize(text)
    ids = [token_to_id.get(t, 1) for t in tokens]
    
    # Truncate or Pad
    ids = ids[:max_len]
    padding_len = max_len - len(ids)
    ids = ids + ([0] * padding_len)
    
    # 1 for data, 0 for padding
    mask = ([1] * (max_len - padding_len)) + ([0] * padding_len)
    return ids, mask

input_ids_list, mask_list, labels_list = [], [], []

for _, row in df.iterrows():
    in_ids, in_mask = encode_and_pad(row['buggy_clean'], MAX_LENGTH)
    out_ids, _ = encode_and_pad(row['fixed_clean'], MAX_LENGTH)
    
    input_ids_list.append(in_ids)
    mask_list.append(in_mask)
    labels_list.append(out_ids)

# --- Step 4: Generate the .pt Binary File ---
training_data = {
    "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
    "attention_mask": torch.tensor(mask_list, dtype=torch.long),
    "labels": torch.tensor(labels_list, dtype=torch.long)
}
torch.save(training_data, "processed_training_data.pt")
print("Success: 'processed_training_data.pt' created!")

# --- Step 5: Final Human-Readable Verification CSV ---
verification_data = []
for idx, row in df.iterrows():
    b_tokens = custom_tokenize(row['buggy_clean'])
    verification_data.append({
        'Row': idx,
        'Visible_Tokens': " | ".join(b_tokens), # Separated by pipes for clarity
        'Token_IDs': [token_to_id.get(t, 1) for t in b_tokens]
    })

pd.DataFrame(verification_data).to_csv('visible_token_check.csv', index=False)
print("Success: 'visible_token_check.csv' created.")




# --- NEW Step 6: Reconstruction Test ---
print("\n--- RECONSTRUCTION TEST ---")

def decode_ids_to_code(ids_tensor):
    rebuilt = []
    for tid in ids_tensor.tolist():
        if tid == 0: break # Stop at padding
        word = id_to_token.get(tid, "<UNK>")
        
        # Convert symbols back to actual whitespace
        if word == "[NWLN]": rebuilt.append("\n")
        elif word == "[INDENT]": rebuilt.append("    ")
        elif word == "[SPC]": rebuilt.append(" ")
        else: rebuilt.append(word)
    return "".join(rebuilt)

# Test Row 0
test_ids = training_data["input_ids"][0]
print(f"Row 0 Reconstructed:\n{decode_ids_to_code(test_ids)}")