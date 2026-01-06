import csv
import io
import tokenize
from pathlib import Path

INPUT_CSV = Path("code_bug_fix_pairs.csv")
OUTPUT_CSV = Path("code_bug_fix_pairs_tokenized.csv")

def py_tokenize(code: str) -> str:
    """
    Tokenize Python code using stdlib tokenizer.
    Returns tokens as a space-separated string (CSV-friendly).
    Falls back to whitespace split if tokenization fails.
    """
    toks = []
    try:
        reader = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(reader):
            if tok.type in (
                tokenize.ENCODING,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                continue
            toks.append(tok.string)
    except tokenize.TokenError:
        toks = code.strip().split()
    return " ".join(toks)

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as fin, \
         OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError("Could not read CSV header.")

        # Ensure required columns exist
        for col in ("buggy_code", "fixed_code"):
            if col not in reader.fieldnames:
                raise ValueError(f"Missing required column '{col}'. Found: {reader.fieldnames}")

        fieldnames = list(reader.fieldnames) + ["buggy_tokens", "fixed_tokens"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            buggy = row.get("buggy_code", "") or ""
            fixed = row.get("fixed_code", "") or ""
            row["buggy_tokens"] = py_tokenize(buggy)
            row["fixed_tokens"] = py_tokenize(fixed)
            writer.writerow(row)

    print(f"Done! Wrote {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
