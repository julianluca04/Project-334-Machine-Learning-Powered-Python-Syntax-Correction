# Seq2Seq Deep Learning Model for Syntax Correction (Encoder-Decoder Transformer)

import pandas as pd
import tensorflow as tf
import numpy as np


# GPU regulation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")




################################################################
# PHASE 0: DATA PRE-PROCESSING
################################################################

# Load Data
df1 = pd.read_csv("code_bug_fix_pairs.csv")
df2 = pd.read_csv("bug_fix_pairs.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Shuffle it
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Combined Dataset Size: {len(df)} rows")

# Clean Data
import re
def clean_code_logic(text): # Function for cleaning datasets (remove # Sample ID)
    if not isinstance(text, str):
            return ""

    marker = "# Sample ID"
    index = text.find(marker)

    if index == -1:
        return text.strip()

    return text[:index].strip() 


print("Cleaning data and building custom vocabulary...")
df['buggy_clean'] = df['buggy_code'].apply(clean_code_logic)
df['fixed_clean'] = df['fixed_code'].apply(clean_code_logic)

# Gather cleaned tokens into a list
# Using .tolist() is much faster than iterrows()
texts = df['buggy_clean'].tolist() + df['fixed_clean'].tolist()
print(f"Collected {len(texts)} cleaned code snippets.")





#############################################################
# PHASE 1: CHARACTER-LEVEL TOKENIZATION
#############################################################

class CharTokenizer:
    def __init__(self, texts, indent_spaces=4):
        self.indent_spaces = indent_spaces

        # Special tokens (fixed IDs)
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<indent>"] # padding, beginning of sequence, end of sequence, indent
        self.stoi = {tok: i for i, tok in enumerate(self.special_tokens)} # map string to index
        self.itos = {i: tok for tok, i in self.stoi.items()} # map index to string

        # Collect characters
        chars = set() # set of unique characters
        for text in texts: 
            chars.update(text) # add all of the unique characters

        # Assign IDs
        offset = len(self.stoi) # offset by 4, ie numbers taken up by special tokens already
        for i, ch in enumerate(sorted(chars)): 
            self.stoi[ch] = i + offset # map the characters to index
            self.itos[i + offset] = ch # reverse map

        self.vocab_size = len(self.stoi)

    def encode(self, text, add_special_tokens=True):
        ids = []

        if add_special_tokens:
            ids.append(self.stoi["<bos>"])

        i = 0
        while i < len(text):
            # Handle indentation (only at line start)
            if text[i] == " ":
                count = 0
                while i < len(text) and text[i] == " ":
                    count += 1
                    i += 1
                # you kinda reverse engineer from the amount of spaces how many indents there are

                while count >= self.indent_spaces: # when count bigger than 4 it counts as an indent
                    ids.append(self.stoi["<indent>"]) # add token for indent
                    count -= self.indent_spaces # reduce count by 4

                # leftover spaces
                ids.extend([self.stoi[" "]] * count) # add remaining spaces as formatting spaces basically
            else:
                ids.append(self.stoi[text[i]])
                i += 1

        if add_special_tokens:
            ids.append(self.stoi["<eos>"])

        return ids

    def decode(self, ids):
        text = "" #initialize string
        for i in ids:
            token = self.itos.get(i, "") # get the token from ids (the index)
            if token == "<bos>" or token == "<eos>" or token == "<pad>":
                continue
            elif token == "<indent>":
                text += " " * self.indent_spaces #. add 4 spaces if there was an indent
            else:
                text += token #just add the token to the string
        return text



# Tokenization

tokenizer = CharTokenizer(texts)
print("Vocab size:", tokenizer.vocab_size)
print(list(tokenizer.stoi.items()))

sample = df.iloc[0]["buggy_clean"]
print("ORIGINAL:")
print(repr(sample))

encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)
print("ENCODED:")
print(encoded[:50])  # print first 50 tokens
print("DECODED:")
print(repr(decoded))


assert decoded == sample
print("Yippee reversiblityy")





#################################################################
## PHASE 2: FILTERING + PADDING
#################################################################

# Encode all entries 
df['buggy_entries'] = df['buggy_clean'].apply(tokenizer.encode)
df['fixed_entries'] = df['fixed_clean'].apply(tokenizer.encode)

# Define max length of an entry
max_entry_length = 512
initial_count = len(df) # initial number of entries


# take only those entries that are equal or shorter than max entry length
df = df[df['buggy_entries'].map(len) <= max_entry_length]
df = df[df['fixed_entries'].map(len) <= max_entry_length]
print(f"Removed {initial_count - len(df)} sequences longer than {max_entry_length}")
print(f"New dataset size: {len(df)} samples")


# Padding function
def pad_entry(entry, max_len, pad_token_id):
    """check entry length and add enough pads at the end of it
    to make it equally long as max entry length"""
    entry_copy = entry.copy() # safety step
    if len(entry_copy) < max_len:
        entry_copy.extend([pad_token_id] * (max_len - len(entry_copy)))
    return entry_copy


# Pad the entries
# stoi takes index of <pad> and we name that pad_token_id
# we apply function using lambda with lambda (x) = pad_entry(x,y,z)
pad_token_id = tokenizer.stoi["<pad>"]
df['buggy_padded'] = df['buggy_entries'].apply(lambda x: pad_entry(x, max_entry_length, pad_token_id))
df['fixed_padded'] = df['fixed_entries'].apply(lambda x: pad_entry(x, max_entry_length, pad_token_id))

print(f"Sample padded entry length: {len(df['buggy_padded'].iloc[0])}") 
df.to_csv("bug_fix_pairs_tokenized_padded_nomask.csv", index=False) # save padded entries






######################################################################
### PHASE 3: EMBEDDING LAYER + POSITIONAL ENCODING
######################################################################

import tensorflow as tf
from tensorflow import keras

# Part 1: EMBEDDING LAYER
embedding_dim = 128  # aribitrary choice of embedding dimension
embedding_layer = keras.layers.Embedding(input_dim=tokenizer.vocab_size,
                                         output_dim=embedding_dim,
                                         input_length=max_entry_length,
                                         )  # mask removed
# Overall there are 3 dimensions: 
# 1. Batch size (number of entries processed together) TBD
# 2. Sequence length = max entry length (due to padding)
# 3. Embedding dimension (number of elements in a token vector)

# test on a sample entry
# Take first entry from dataframe, make it a tensor, and run it through embedding layer
sample_padded_entry = df['buggy_padded'].iloc[0]
sample_padded_entry_tensor = tf.constant([sample_padded_entry])  # batch = 1
embedded_output = embedding_layer(sample_padded_entry_tensor)
print("Embedded output shape:", embedded_output.shape)
print("Embedded output (first 1 token):", embedded_output[0, 0, :]) 



# Part 2: POSITIONAL ENCODING
"""2D Matrix
every row of a matrix is a position in an entry (there is max_length positions)
Every position is defined as a vector of embedding dimension (columns of this matrix)
Values of the elements in a row are defined by sine and cosine functions (sin, cos, sin...)"""
import numpy as np
def positional_encoding(max_len, d_model):
    pos_enc = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    return pos_enc

pos_enc = positional_encoding(max_entry_length, embedding_dim)
print("Positional encoding shape:", pos_enc.shape)
print("Positional encoding (first position):", pos_enc[0])

# Combine embedding with positional encoding
def add_positional_encoding(embedded_inputs, pos_enc):
    """
    Imagine it like adding files in a drawer
    Each layer (or level) of "depth" in tensor recieves the same 2D file (positional encoding)
    Positional encoding is like a map to see positions

    embedded_inputs: 3D Tensor (batch_size, seq_len, embed_dim)
    pos_enc: 2D Matrix (seq_len, embed_dim)
    Tensor is like a set (a drawer) of n matrices
    Each of these matrices inside the tensor now gets subtracted by pos_enc matrix
    This makes every element in these matrices adjusted according to its position
    (Inside the matrix i.e. its position in the entry)

    use tf.cast to convert numpy array to tensor
    we also add a new axis to the matrix to make it 3D (1, seq_len, embed_dim)
    Finally, by adding them, each batch layer gets this same positional encoding
    We use "keras.layers.add" instead of "+" to preserve the mask
    """
    positional_encoding_tensor = tf.cast(pos_enc, dtype=tf.float32)
    positional_encoding_tensor = positional_encoding_tensor[tf.newaxis, ...]
    return keras.layers.add([embedded_inputs, positional_encoding_tensor])

# Combined output is a tensor of the same shape as embedded output
# However, now each value is adjusted according to its position in the sequence
combined_output = add_positional_encoding(embedded_output, pos_enc)
print("Combined output shape:", combined_output.shape)
print("Combined output (first token):", combined_output[0, 0, :])
print("Tokenization, embedding, and positional encoding complete.")






##########################################################################
#### PHASE 4: TRANSFORMER BLOCK - ENCODER
##########################################################################
class TransformerBlock(keras.layers.Layer):
    """
    We create a blueprint for a transfromer layer/block (containing multiple sub-layers) 
    that can be stacked up in a model of many transoformer blocks
    We combine multi-head attention which captures relationships between tokens
    and feed-forward networks which process each token individually
    We use Layer Normalization on inputs to stabilise and speed up training
    We use Dropout to prevent overfitting by randomly setting some outputs to zero.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
    Arguments:
    embed_dim: Dimension of embedding vectors
    num_heads: Number of attention heads
    ff_dim: Dimension of feed-forward network
    rate: Dropout rate (percentage of neurons to drop) to prevent overfitting

    Update: make sure model supports masking by setting self.supports_masking = True

    Returns:
    1. Multi-head attention layer checks relationships between tokens in sequence
    2. Feed-forward network processes each token individually, now with context from attention
    3. Layer normalization, normalizes inputs by adjusting and scaling activations
    3.1. Adjusting and scaling activations is done by subtracting meand and dividing by std
    3.2. Epsilon is a small constant to prevent division by zero
    4. Dropout randomly sets a fraction of input units to 0 at each update during training time 
    """
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=None): # no mask
        """
        Arguments:
        inputs: Combined tensor to the transformer block
        training: True/False (to train or not to train)
        
        Returns:
        Tensor of same shape as input, but each element is adjusted based on context

        Mechanism:
        1. Multi-head attention computes attention scores and weighted values,
        by comparing input to input itself (self-attention) while masking padding tokens
        2. Dropout randomly sets a fraction of input units to 0 during training
        3. First layer normalization normalizes the sum of inputs and attention output
        4. Feed-forward network processes each token individually
        5. Second dropout randomly sets a fraction of input units to 0 during training
        6. Second layer normalization normalizes the sum of the output 
        from first normalization and feed-forward output
        7. Result is a smarter tensor
        """
        attn_output = self.att(inputs, inputs) # no mask
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

print("Transformer block applied successfully.")





######################################################################
##### PHASE 5: TRANSFORMER BLOCK - DECODER
######################################################################
"""
Architecture very similar to encoder
There is an extra sub-layer becuase it has to look at the encoder output

1st layer: look at what you have generated so far
2nd layer: look at the encoder output (contextualized buggy code)
3rd layer: Feed Forward*

*In Encoder Feed Forward Network allows for analysed token to be changed
but based only on buggy code context
In Decoder feed forward takes into account what has happened so far in fixed code

Each sublayer is followed by droping out 10% of output to avoid overfitting
And Normalization of values by adding value input value to output
"""
class DecoderBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()

        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)

        self.cross_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)

        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"), 
            keras.layers.Dense(embed_dim),
            ])

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, enc_output, training=None):
        # Step 1: Masked self-attention (causal)
        attn1 = self.self_attn(x, x, use_causal_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Step 2: Cross-attention
        attn2 = self.cross_attn(out1, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Step 3:  Feed-forward
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.layernorm3(out2 + ffn_out)





################################################################
###### PHASE 6: TRAINING SET-UP (MAIN PHASE)
################################################################

from tensorflow import keras
from keras import layers
from keras import Input, Model
from sklearn.model_selection import train_test_split


# Define encoder-decoder transformer function
def encoder_decoder_transformer_model(
    vocab_size, max_len, embed_dim, num_heads,
    ff_dim, rate=0.1, num_layers=3):

    #######################
    # Encoder Section
    #######################

    encoder_inputs = Input(shape=(max_len,), name="encoder_inputs")
    enc_embed = layers.Embedding(vocab_size, embed_dim)(encoder_inputs)
    pos_enc = positional_encoding(max_len, embed_dim)
    enc_x = add_positional_encoding(enc_embed, pos_enc)

    for _ in range(num_layers):
        enc_x = TransformerBlock(embed_dim, num_heads, ff_dim, rate)(enc_x)

    encoder_outputs = enc_x


    #######################
    # Decoder Section
    #######################

    decoder_inputs = Input(shape=(max_len,), name="decoder_inputs")
    dec_embed = layers.Embedding(vocab_size, embed_dim)(decoder_inputs)
    dec_x = add_positional_encoding(dec_embed, pos_enc)

    for _ in range(num_layers):
        dec_x = DecoderBlock(embed_dim, num_heads, ff_dim, rate)(dec_x, encoder_outputs)

    dec_x = layers.Dropout(rate)(dec_x)
    outputs = layers.Dense(vocab_size)(dec_x)

    return Model(inputs=[encoder_inputs, decoder_inputs],outputs=outputs)




# Data modification for encoder-decoder set-up
def prepare_seq2seq_data(df, max_len):
    X_enc = np.array(df['buggy_padded'].tolist())
    Y_full = np.array(df['fixed_padded'].tolist())

    # Teacher forcing with same length
    bos_token_id = tokenizer.stoi["<bos>"]
    X_dec = np.zeros_like(Y_full)
    X_dec[:, 0] = bos_token_id  # starting sequence
    X_dec[:, 1:] = Y_full[:, :-1]

    # Decoder target
    Y = Y_full.copy()

    return X_enc, X_dec, Y



train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
X_enc_train, X_dec_train, y_train = prepare_seq2seq_data(train_df, max_entry_length)
X_enc_test, X_dec_test, y_test = prepare_seq2seq_data(test_df, max_entry_length)

cols = ["buggy_clean", "fixed_clean", "buggy_padded", "fixed_padded"]

train_df[cols].to_csv("train_nomask_data.csv", index=False)
test_df[cols].to_csv("test_nomask_data.csv", index=False)
np.save("train_nomask_indices.npy", train_df.index.values)
np.save("test_nomask_indices.npy", test_df.index.values)

# Define the model
model = encoder_decoder_transformer_model(
    tokenizer.vocab_size,
    max_entry_length,
    embedding_dim,
    num_heads=4,
    ff_dim=512
    )

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate = 0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
    )

results = model.fit([X_enc_train, X_dec_train], y_train,
                    validation_data=([X_enc_test, X_dec_test], y_test), 
                    epochs=20, batch_size=32)


# Save the model and tokenizer
import pickle

tokenizer_path = "char_tokenizer_nomask.pkl"
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Tokenizer saved to {tokenizer_path}")

model.save("se2seq_nomask_model.keras")
print("Model saved")




#################################################################
####### PHASE 7: ANALYSIS
#################################################################

# Evaluate the model
loss, accuracy = model.evaluate([X_enc_test, X_dec_test], y_test)

print(f"Final Accuracy: {accuracy * 100:.2f}%")


# some plots
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))


# Plotting the results
plt.subplot(1, 2, 1)
plt.plot(results.history['accuracy'], label='train accuracy')
plt.plot(results.history['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results.history['loss'], label='training loss')
plt.plot(results.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('seq2seq_acc_loss_no_mask.png')
print("Plot saved seq2seq_acc_loss.png")

plt.close('all')


#################################################
# PHASE 8: UNCERTAINTY ANALYSIS WITH MONTE CARLO DROPOUT
##################################################

def analyze_uncertainty(buggy_str, n_iterations=20):
    # 1. Pre-process
    encoded = tokenizer.encode(buggy_str)
    padded = pad_entry(encoded, max_entry_length, tokenizer.stoi["<pad>"])
    enc_input = tf.constant([padded], dtype=tf.int32)
    
    # --- FIX HERE: Initialize as a Tensor, not just a NumPy array ---
    dec_input_np = np.zeros((1, max_entry_length))
    dec_input_np[0, 0] = tokenizer.stoi["<bos>"]
    dec_input = tf.constant(dec_input_np, dtype=tf.float32) # Match the model's expected input type

    all_probs = []
    print(f"Running {n_iterations} stochastic passes...")
    
    for _ in range(n_iterations):
        # By passing both as Tensors, the ValueError disappears
        logits = model([enc_input, dec_input], training=True)
        probs = tf.nn.softmax(logits, axis=-1)
        all_probs.append(probs)

    # ... rest of your calculation ...
    all_probs = tf.stack(all_probs)
    mean_probs = tf.reduce_mean(all_probs, axis=0)
    variance = tf.math.reduce_variance(all_probs, axis=0)
    uncertainty_score = tf.reduce_mean(variance, axis=-1)[0]

    return mean_probs[0], uncertainty_score


def plot_uncertainty_vs_truth(buggy_str, fixed_str, n_iterations=20):
    plt.close('all')
    # 1. Get Mean Probs and Uncertainty Scores
    # (Using the analyze_uncertainty function from the previous step)
    mean_probs, scores = analyze_uncertainty(buggy_str, n_iterations)
    
    # 2. Tokenize fixed_str to align with the x-axis
    target_tokens = tokenizer.encode(fixed_str, add_special_tokens=False)
    # We only care about the length of the actual fixed code
    seq_len = len(target_tokens)
    
    # Slice the scores and tokens to match the actual code length
    display_scores = scores[:seq_len].numpy()
    display_chars = [tokenizer.itos[t] for t in target_tokens]

    # 3. Plotting
    plt.figure(figsize=(15, 5))
    bars = plt.bar(range(seq_len), display_scores, color='skyblue', edgecolor='navy')
    
    # Highlight high uncertainty in red
    threshold = np.mean(display_scores) + np.std(display_scores)
    for i, bar in enumerate(bars):
        if display_scores[i] > threshold:
            bar.set_color('salmon')

    plt.xticks(range(seq_len), display_chars, rotation=0, fontsize=10)
    plt.xlabel("Correct Characters (Ground Truth)")
    plt.ylabel("Uncertainty (Variance)")
    plt.title("Model Uncertainty Across Predicted Sequence")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("Natalia_plot_nomask.png")
    plt.close('all')


    print("\n" + "="*50)
print("RUNNING UNCERTAINTY ANALYSIS")
print("="*50)

# Pick a specific sample from the test set to analyze
# Example: Let's pick the first one from test_df
sample_row = test_df.iloc[0]
buggy_sample = sample_row['buggy_clean']
fixed_sample = sample_row['fixed_clean']

print(f"Analyzing Sample Bug:\n{repr(buggy_sample)}")

# Call the function you defined
plot_uncertainty_vs_truth(buggy_sample, fixed_sample, n_iterations=20)

print("\nUncertainty analysis complete. Check for 'Natalia_plot.png'.")

#########################################################  
# PHASE 9: LEVENSHTEIN DISTANCE
##########################################################

import Levenshtein

import Levenshtein
import numpy as np

def levenshtein(model, X_enc, X_dec, y_true):
    """
    Calculates the Levenshtein distance across the entire provided dataset.
    """
    all_distances = []
    
    print(f"Calculating Levenshtein distance for {len(X_enc)} samples...")

    # 1. Get model predictions for the entire set
    # Use training=False for standard inference
    predictions = model.predict([X_enc, X_dec], batch_size=32)
    pred_ids = np.argmax(predictions, axis=-1)

    # 2. Iterate through and compare strings
    for i in range(len(pred_ids)):
        # Convert IDs back to strings
        # Use y_true[i] for the ground truth and pred_ids[i] for the model's attempt
        predicted_str = tokenizer.decode(pred_ids[i])
        actual_str = tokenizer.decode(y_true[i])

        # Calculate Distance
        distance = Levenshtein.distance(predicted_str, actual_str)
        all_distances.append(distance)

    # 3. Aggregate Results
    avg_dist = np.mean(all_distances)
    max_dist = np.max(all_distances)
    min_dist = np.min(all_distances)
    total_perfect = all_distances.count(0)

    print("\n" + "="*40)
    print("FULL DATASET LEVENSHTEIN RESULTS")
    print("="*40)
    print(f"Average Edit Distance: {avg_dist:.2f} characters")
    print(f"Max Edit Distance:     {max_dist} characters")
    print(f"Min Edit Distance:     {min_dist} characters")
    print(f"Perfect Fixes (Dist 0): {total_perfect} / {len(all_distances)}")
    print(f"Global Accuracy:       {(total_perfect/len(all_distances))*100:.2f}%")
    print("="*40)

    return all_distances

# Run calculation on test data
test_distances = levenshtein(model, X_enc_test, X_dec_test, y_test)


###########################################################################
# EXTRA PHASE: MORE METRICS
############################################################################
import matplotlib.pyplot as plt
import numpy as np

def plot_full_error_distribution(distances):
    # 1. Clear state to ensure horizontal layout
    plt.close('all')
    
    # 2. Get the actual range of your data
    max_dist = int(np.max(distances))
    min_dist = int(np.min(distances))
    
    # 3. Force a very wide figure to accommodate the long X-axis
    plt.figure(figsize=(20, 7))
    
    # 4. Create bins for every single integer from 0 to max_dist + 1
    bins = np.arange(0, max_dist + 2)
    
    # 5. Plot the histogram
    # 'align=left' makes the bars center on the integer
    plt.hist(distances, bins=bins, color='teal', edgecolor='black', alpha=0.7, align='left')
    
    # 6. Formatting the X-axis
    # If the range is huge, we label every 10 or 20 to keep it clean
    if max_dist > 100:
        step = 20
    elif max_dist > 50:
        step = 10
    else:
        step = 1
        
    plt.xticks(np.arange(0, max_dist + 1, step))
    
    # 7. Titles and Labels
    plt.title(f"Complete Error Distribution (Range: 0 to {max_dist} characters)", fontsize=16)
    plt.xlabel("Exact Number of Character Errors (Edit Distance)", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)
    
    # 8. Add a specialized annotation for the 0-error successes
    perfect_fixes = np.sum(np.array(distances) == 0)
    plt.annotate(f'Perfect: {perfect_fixes}', 
                 xy=(0, perfect_fixes), 
                 xytext=(max_dist*0.1, perfect_fixes*0.8),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 9. Save to your cluster
    plt.savefig("nomask_error_distribution.png")
    plt.close('all')
    print(f"Success! Histogram saved. Max error found was: {max_dist}")

# --- THE CALL ---
plot_full_error_distribution(test_distances)





#########################################################
# PHASE 10: BLEU SCORE
###########################################################

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def calculate_corpus_bleu(model, X_enc, X_dec, y_test, tokenizer, batch_size=32):
    print(f"Generating predictions for {len(X_enc)} samples...")
    predictions = model.predict([X_enc, X_dec], batch_size=batch_size)
    pred_ids = np.argmax(predictions, axis=-1)
    
    all_references = [] 
    all_candidates = [] 
    
    for i in range(len(pred_ids)):
        predicted_str = tokenizer.decode(pred_ids[i])
        actual_str = tokenizer.decode(y_test[i])
        
        # Character-level formatting for NLTK
        all_references.append([list(actual_str)])
        all_candidates.append(list(predicted_str))

    smoothie = SmoothingFunction().method1
    weights = (0.5, 0.3, 0.15, 0.05)
    
    # We call the NLTK function here, NOT the name of our own function
    score = corpus_bleu(all_references, all_candidates, 
                        weights=weights, smoothing_function=smoothie)
    
    print("\n" + "="*50)
    print("CORPUS-LEVEL BLEU SCORE")
    print("="*50)
    print(f"Corpus BLEU Score: {score * 100:.2f}%")
    print("="*50)
    
    return score


final_bleu = calculate_corpus_bleu(model, X_enc_test, X_dec_test, y_test, tokenizer)

