import pandas as pd
import os
import re

# 1. Setup paths
input_file = 'buggy_dataset/train.pkl' # Adjust based on your download
output_dir = 'processed_data'
os.makedirs(f"{output_dir}/buggy", exist_ok=True)
os.makedirs(f"{output_dir}/fixed", exist_ok=True)

def clean_code(code):
    """Optional: Basic cleaning to focus on syntax as you requested."""
    # This keeps the code logic but you can add string-stripping regex here
    return code.strip()

# 2. Load the data
print("Loading dataset...")
df = pd.read_pickle(input_file)

# 3. Process and Save
print(f"Processing {len(df)} snippets...")
for i, row in df.iterrows():
    # We use 'without_docstrings' to keep the model lightweight for your M1 Max
    buggy_content = clean_code(row['before_merge_without_docstrings'])
    fixed_content = clean_code(row['after_merge_without_docstrings'])
    
    # Save as individual .py files for easy viewing/testing
    with open(f"{output_dir}/buggy/snippet_{i}.py", "w") as f:
        f.write(buggy_content)
    with open(f"{output_dir}/fixed/snippet_{i}.py", "w") as f:
        f.write(fixed_content)

print(f"Done! Files saved in {output_dir}")f