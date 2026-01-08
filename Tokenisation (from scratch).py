import pandas as pd
import json

df = pd.read_csv("code_bug_fix_pairs.csv") # read data and turn into dataframe


def vocabulary(df): # this is a function to build character vocabulary

    full_text = "".join(df["buggy_code"].astype(str)) + "".join(df["fixed_code"].astype(str))
    # full text is a single string containing all characters from both columns
    # we joined them by empty spaces and converted to string type (+ adds them together)
    characters = sorted(list(set(full_text)))
    # we remove duplicates by converting string to set, then we convert it to list and sort it

    vocabulary_dictionary = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    # this is a vocabulary dictionary, we immediately add special tokens with their IDs
    for char in characters: # we go through our list of characters
        if char not in vocabulary_dictionary: # if charaters is not already in dictionary
            vocabulary_dictionary[char] = len(vocabulary_dictionary)
        # we add it to dictionary
        # the value (ID) is the current length of the dictionary (we started from 0)     
    return vocabulary_dictionary

our_vocabulary = vocabulary(df) # we define our vocabulary from dataframe

#Docs
with open("vocabulary_from_scratch.json", "w", encoding="utf-8") as f:
    json.dump(our_vocabulary, f, indent=4)

vocabulary_df = pd.DataFrame(list(our_vocabulary.items()), columns=['Character', 'ID'])
vocabulary_df.to_csv('vocabulary_from_scratch.csv', index=False)

#Final Print Statements
print(f"Vocabulary size: {len(our_vocabulary)}")
print("vocabulary_from_scratch.json and vocabulary_from_scratch.csv are saved")