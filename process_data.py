import pandas as pd
import tokenize
import io

INPUT_CSV = "code_bug_fix_pairs.csv"
OUTPUT_CSV = "code_bug_fix_pairs_tokenized.csv"

def py_tokenize(code: str) -> list[str]:
    """
    Tokenize Python code using Python's own lexer.
    Skips whitespace/newlines/indent/dedent/endmarkers.
    Falls back to whitespace split if the code is too broken to tokenize.
    """
    tokens: list[str] = []
    try:
        reader = io.StringIO(str(code)).readline
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
            tokens.append(tok.string)
    except tokenize.TokenError:
        tokens = str(code).strip().split()
    return tokens

def main() -> None:
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"buggy_code", "fixed_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    print("Tokenizing buggy_code and fixed_code ...")
    df["buggy_tokens"] = df["buggy_code"].apply(py_tokenize)
    df["fixed_tokens"] = df["fixed_code"].apply(py_tokenize)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Wrote {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
