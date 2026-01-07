import pandas as pd
import re


# ---------- 1. Load CSV ----------
df = pd.read_csv("code_bug_fix_pairs.csv").dropna(subset=["buggy_code", "fixed_code"])




