from datasets import load_dataset
import pandas as pd

import re

def remove_def_return_and_fix_indentation(text):
    if not isinstance(text, str):
        return text

    # Step 1: remove def / return lines
    lines = [
        line for line in text.splitlines()
        if not re.match(r"^\s*(def|return)\b", line)
    ]

    # Keep empty-only result safe
    if not any(line.strip() for line in lines):
        return ""

    # Step 2: compute minimum indentation of remaining lines
    indents = [
        len(line) - len(line.lstrip())
        for line in lines
        if line.strip()  # ignore blank lines
    ]
    min_indent = min(indents) if indents else 0

    # Step 3: dedent by exactly that amount
    dedented = [
        line[min_indent:] if len(line) >= min_indent else line
        for line in lines
    ]

    return "\n".join(dedented)


print("Loading full BuggedPythonLeetCode dataset...")

dataset = load_dataset(
    "NeuroDragon/BuggedPythonLeetCode",
    split="train"
)

print("Fields in the dataset:", dataset.column_names)

records = []
for sample in dataset:
    bug = sample.get("bugged_code")
    fix = sample.get("original_code")  # fixed code

    if not bug or not fix:
        continue

    records.append({
        "bugged_code": bug,
        "fixed_code": fix
    })



df = pd.DataFrame(records)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].map(remove_def_return_and_fix_indentation)

# Save CSV without quotation marks wrapping code snippets
df.to_csv(
    "bug_fix_pairs.csv",
    index=False,
    quoting=2,         # QUOTE_NONE
    escapechar="\\",     # escape special characters
    lineterminator="\n"  # correct argument name
)





print(f"Saved {len(records):,} bug-fix pairs to bug_fix_pairs.csv")


