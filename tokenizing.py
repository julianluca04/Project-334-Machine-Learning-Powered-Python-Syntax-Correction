from transformers import AutoTokenizer
from datasets import load_dataset

# load tokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# load CSV file
dataset = load_dataset("csv", data_files="code_bug_fix_pairs.csv") # this creates a Dataset object which is optimized for memory

# define tokenization function
def tokenize_function(code):
    return tokenizer(
        code["buggy_code"], 
        code["fixed_code"], 
        truncation=True,      # cuts sequences that are too long
        padding="max_length", # pads shorter sequences
        max_length=80      # number of tokens to pad
    )

# map the function over the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# check result
print(tokenized_dataset["train"][0].keys())
# accesses the "train" portion (split) of your data, grabs only the first row (index 0), lists the labels/column names for that row

# select only the columns you want to keep
cols_to_keep = ["input_ids", "attention_mask", "token_type_ids"] # numerical representation of text, tells the model which tokens are real and which are padding, Distinguishes between the first column and the second column
final_df = tokenized_dataset["train"].select_columns(cols_to_keep)

# save to parquet in order to keep list data type
final_df.to_csv("code_bug_fix_pairs_token.csv")