import pandas as pd
import json

df = pd.read_csv("code_bug_fix_pairs.csv") # read data and turn it into dataframe


def vocabulary(df): 
    """ this is a function that creates vocabulary from dataframe
    full_text is a single string containing all characters from both columns
    we joined them by empty spaces and converted to string type (+ adds them together)
    we remove duplicates by converting string to set, then we convert it to list and sort it
    Unkown is added in case we encounter multiple characters that are not in our vocabulary
    Indent and dedent are represented by empty spaces (" ", "\n", "\r")
    """
    full_text = "".join(df["buggy_code"].astype(str)) + "".join(df["fixed_code"].astype(str))
    characters = sorted(list(set(full_text)))

    vocabulary_dictionary = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    # this is a vocabulary dictionary, we immediately add special tokens with their IDs
    # PAD = padding, UNK = unknown, SOS = start of sequence, EOS = end of sequence

    for char in characters: # we go through our list of characters
        if char not in vocabulary_dictionary: # if charaters is not already in dictionary
            vocabulary_dictionary[char] = len(vocabulary_dictionary)
        # we add it to the dictionary
        # the value (ID) is the current length of the dictionary (we started from 0)     
    return vocabulary_dictionary

our_vocabulary = vocabulary(df) # we define our vocabulary from dataframe

# Documents

# json file
# we open a document in which we will write our vocabulary in json format
# utf-8 helps with special characters
# .dump writes our vocabulary to the file with an indentation of 4 spaces to make it clearer
with open("vocabulary_from_scratch.json", "w", encoding="utf-8") as f:
    json.dump(our_vocabulary, f, indent=4)

# csv file
# use pd.DataFrame to turn dictionary into dataframe
# we turn ID and value into pairs using .items()
# we then create a list of these pairs with each pair being in its own row
# first element is the "character" and second is "ID"
vocabulary_df = pd.DataFrame(list(our_vocabulary.items()), columns=['Character', 'ID'])
vocabulary_df.to_csv('vocabulary_from_scratch.csv', index=False)

# Final Print Statements
print(f"Vocabulary size: {len(our_vocabulary)}")
print("vocabulary_from_scratch.json and vocabulary_from_scratch.csv are saved")