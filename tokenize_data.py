import pandas as pd
from transformers import AutoTokenizer
import torch



def save_tokenized_data():
    all_inputs = []
    all_labels = []

    for i in range(len(df)):
        # Tokenize pair
        inputs = tokenizer("fix: " + df['buggy_code'][i], truncation=True, padding='max_length', max_length=128)
        labels = tokenizer(df['fixed_code'][i], truncation=True, padding='max_length', max_length=128)
        
        all_inputs.append(inputs['input_ids'])
        all_labels.append(labels['input_ids'])

    # Save as a PyTorch file
    training_data = {
        "input_ids": torch.tensor(all_inputs),
        "labels": torch.tensor(all_labels)
    }
    torch.save(training_data, "processed_training_data.pt")
    print("Success: Binary training file created!")






# 1. Load your local CSV
df = pd.read_csv('code_bug_fix_pairs.csv')

# 2. Use a tokenizer designed for code logic
# 'Salesforce/codet5-small' is lightweight and excellent for M1 Max hardware
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

def preprocess_function(buggy_code, fixed_code):
    # We prefix the input so the model knows the task
    input_text = "fix python: " + buggy_code
    
    # Tokenize input (Buggy)
    model_inputs = tokenizer(
        input_text, 
        max_length=128, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    # Tokenize labels (Fixed)
    # The model learns to transform input_ids into these labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            fixed_code, 
            max_length=128, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

    return model_inputs["input_ids"], labels["input_ids"]

# Example: Process the first pair
input_ids, label_ids = preprocess_function(df['buggy_code'][0], df['fixed_code'][0])

print(f"Tokenized Input (First 10 IDs): {input_ids[0][:10]}")
print(f"Decoded back to text: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")






save_tokenized_data()