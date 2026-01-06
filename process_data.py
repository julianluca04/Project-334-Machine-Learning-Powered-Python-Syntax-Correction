import pandas as pd
import re


# ---------- 1. Load CSV ----------
df = pd.read_csv("code_bug_fix_pairs.csv").dropna(subset=["buggy_code", "fixed_code"])


# ---------- 2. Tokeniser for code ----------
# split into: identifiers/keywords/numbers + single symbols
TOKEN_PATTERN = r"[A-Za-z_][A-Za-z_0-9]*|\d+|==|!=|<=|>=|[^\s]"

def tokenize_code(code: str):
    return re.findall(TOKEN_PATTERN, code)

df["buggy_tokens"] = df["buggy_code"].apply(tokenize_code)
df["fixed_tokens"] = df["fixed_code"].apply(tokenize_code)

