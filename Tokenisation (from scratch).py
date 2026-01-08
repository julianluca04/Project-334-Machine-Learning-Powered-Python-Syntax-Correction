import pandas as pd
import json
from collections import Counter

df = pd.read_csv("code_bug_fix_pairs.csv")


def build_char_vocab(df):
    all_text = "".join(df["buggy_code"].astype(str)) + "".join(df["fixed_code"].astype(str))
    unique_chars = sorted(list(set(all_text)))
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for char in unique_chars:
        if char not in vocab:
            vocab[char] = len(vocab)      
    return vocab

vocab = build_char_vocab(df)

#Converter
def char_to_ids(text, vocab):
    text = str(text) 
    return [vocab["<SOS>"]] + [vocab.get(c, vocab["<UNK>"]) for c in text] + [vocab["<EOS>"]]

print("Converting code to character ID")
df["buggy_ids"] = df["buggy_code"].apply(lambda x: char_to_ids(x, vocab))
df["fixed_ids"] = df["fixed_code"].apply(lambda x: char_to_ids(x, vocab))

#Docs
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=4)

vocab_df = pd.DataFrame(list(vocab.items()), columns=['Character', 'ID'])
vocab_df.to_csv('vocabulary_from_scratch.csv', index=False)

#Final Print Statements
print(f"Vocabulary size: {len(vocab)}")
print("vocab.json and vocabulary_from_scratch.csv are saved")