# !pip install transformers
# !pip install torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# ---- helper function for one string ----
def tokenize_text(text):
    enc = tokenizer(text, add_special_tokens=True)
    # <s>  — Start-of-sequence token.
    # </s> — End-of-sequence token.
    # <pad> — Padding token.
    # <unk> — Unknown token..
    # Ġ  — Space-marker prefix (RoBERTa / byte-BPE tokenizers).
    # etc...
    ids = enc["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    return ids, toks

# ---- apply to buggy_code ----
df["buggy_ids_tokens"] = df["buggy_code"].apply(tokenize_text)
df["buggy_ids"] = df["buggy_ids_tokens"].apply(lambda x: x[0])
df["buggy_tokens"] = df["buggy_ids_tokens"].apply(lambda x: x[1])

# ---- apply to fixed_code ----
df["fixed_ids_tokens"] = df["fixed_code"].apply(tokenize_text)
df["fixed_ids"] = df["fixed_ids_tokens"].apply(lambda x: x[0])
df["fixed_tokens"] = df["fixed_ids_tokens"].apply(lambda x: x[1])

# ---- build vocabularies  ----
buggy_vocab = sorted({tok for toks in df["buggy_tokens"] for tok in toks})
fixed_vocab = sorted({tok for toks in df["fixed_tokens"] for tok in toks})

pd.DataFrame({"token": buggy_vocab}).to_csv("buggy_vocab_hf.csv", index=False)
pd.DataFrame({"token": fixed_vocab}).to_csv("fixed_vocab_hf.csv", index=False)
