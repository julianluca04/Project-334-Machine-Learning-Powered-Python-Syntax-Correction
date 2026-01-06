import pandas as pd
import torch
from transformers import AutoTokenizer

# 1. Configuration (Keep everything in one place)
MODEL_NAME = "Salesforce/codet5-small"
MAX_LENGTH = 128
PREFIX = "fix python: "
FILE_PATH = 'code_bug_fix_pairs.csv'

# 2. Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df = pd.read_csv(FILE_PATH)

def prepare_and_save_dataset(dataframe):
    print(f"Tokenizing {len(dataframe)} pairs...")

    # OPTIMIZATION: Pass the whole list to the tokenizer (Batching)
    # This is much faster than a 'for' loop
    inputs = [PREFIX + str(code) for code in dataframe['buggy_code'].tolist()]
    targets = [str(code) for code in dataframe['fixed_code'].tolist()]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    # Tokenize targets
    # Note: Modern transformers use 'text_target' instead of the 'with' block
    labels = tokenizer(
        text_target=targets, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )["input_ids"]

    # 3. Handle Padding in Labels
    # Important: Replace padding token id (0) with -100 
    # so the loss function ignores them during training
    labels[labels == tokenizer.pad_token_id] = -100

    # 4. Save
    training_data = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels
    }
    
    torch.save(training_data, "processed_training_data.pt")
    print("Success: Binary training file 'processed_training_data.pt' created!")

# Run the process
if __name__ == "__main__":
    prepare_and_save_dataset(df)