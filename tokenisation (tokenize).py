# tokenisation (incorrect)

import pandas as pd
import tokenize
import io
from collections import Counter
import json

df = pd.read_csv("code_bug_fix_pairs.csv")
all_code_data = df["buggy_code"].tolist() + df["fixed_code"].tolist()

def python_tokenize(code):
    tokens = []
    try:
        token_gen = tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline)
        for toknum, tokval, start, end, line in token_gen:
            if toknum in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT}:
                continue
            if toknum == tokenize.NAME:
                tokens.append(tokval)
            elif toknum == tokenize.STRING:
                tokens.append("<STR>") 
            elif toknum == tokenize.NUMBER:
                tokens.append("<NUM>")
            elif toknum == tokenize.NEWLINE or toknum == tokenize.NL:
                tokens.append("<NEWLINE>")
            elif toknum == tokenize.INDENT:
                tokens.append("<INDENT>")
            elif toknum == tokenize.DEDENT:
                tokens.append("<DEDENT>")
            else:
                tokens.append(tokval)
    except Exception:
        tokens = [t for t in code.split() if not t.startswith('#')]
    return tokens

# step 2

def build_vocab_from_csv(df, max_vocab_size=2000):
    all_tokens = []
    #Combine columns
    for col in ["buggy_code", "fixed_code"]:
        for code_str in df[col]:
            all_tokens.extend(python_tokenize(str(code_str)))
    
    #dictionary
    counts = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for word, _ in counts.most_common(max_vocab_size):
        if word not in vocab:
            vocab[word] = len(vocab)
            
    return vocab

df = pd.read_csv("code_bug_fix_pairs.csv")
vocab = build_vocab_from_csv(df)

#Converter

def text_to_ids(code_str, vocab):
    tokens = python_tokenize(str(code_str))
    return [vocab["<SOS>"]] + [vocab.get(t, vocab["<UNK>"]) for t in tokens] + [vocab["<EOS>"]]

df["buggy_ids"] = df["buggy_code"].apply(lambda x: text_to_ids(x, vocab))
df["fixed_ids"] = df["fixed_code"].apply(lambda x: text_to_ids(x, vocab))

# save vocab
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

vocab_df = pd.DataFrame(list(vocab.items()), columns=['Token', 'ID'])
vocab_df.to_csv('vocabulary.csv', index=False)

print("Vocabulary and token IDs saved.")