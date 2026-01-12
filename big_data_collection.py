from datasets import load_dataset
import pandas as pd

print("🔄 Loading full BuggedPythonLeetCode dataset...")

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

# Save CSV without quotation marks wrapping code snippets
df.to_csv(
    "bug_fix_pairs.csv",
    index=False,
    quoting=2,           # QUOTE_NONE
    escapechar="\\",     # escape special characters
    lineterminator="\n"  # correct argument name
)

print(f"💾 Saved {len(records):,} bug-fix pairs to bug_fix_pairs.csv")

