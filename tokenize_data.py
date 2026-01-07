import pandas as pd
import torch
from transformers import AutoTokenizer


# 1. Configuration
MODEL_NAME = "Salesforce/codet5-small"
MAX_LENGTH = 256
PREFIX = "fix python: "
FILE_PATH = 'code_bug_fix_pairs.csv'

# 2. Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df = pd.read_csv(FILE_PATH)

def prepare_and_save_dataset(dataframe):
    # --- Part A: Save Vocabulary to CSV ---
    print("Generating Vocabulary CSV...")
    vocab = tokenizer.get_vocab()
    # Sort by ID so the CSV is organized
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_df = pd.DataFrame(sorted_vocab, columns=['Token', 'ID'])
    vocab_df.to_csv('tokenizer_vocabulary.csv', index=False)
    print("Success: 'tokenizer_vocabulary.csv' created.")

    # --- Part B: Tokenize and Save .pt File ---
    print(f"Tokenizing {len(dataframe)} pairs...")

    inputs = [PREFIX + str(code) for code in dataframe['buggy_code'].tolist()]
    targets = [str(code) for code in dataframe['fixed_code'].tolist()]

    # Batch Tokenization
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    labels = tokenizer(
        text_target=targets, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )["input_ids"]

    # Replace padding (usually 0) with -100 so the model doesn't learn to predict it
    labels[labels == tokenizer.pad_token_id] = -100

    # Package into a dictionary
    training_data = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels
    }
    
    # Save as binary for M1 GPU efficiency
    torch.save(training_data, "processed_training_data.pt")
    print("Success: Binary training file 'processed_training_data.pt' created!")

if __name__ == "__main__":
    prepare_and_save_dataset(df)