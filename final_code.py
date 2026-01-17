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
df.to_csv("code_bug_fix_pairs_tokenized_padded.csv", index=False) # save padded entries






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
                                         mask_zero=True)  # mask_zero masks the padding tokens
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
        self.supports_masking = True
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=None, mask=None): # mask added to ignore padding tokens
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
        attn_output = self.att(inputs, inputs, attention_mask=mask)
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
        self.supports_masking = True

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
    enc_embed = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(encoder_inputs)
    pos_enc = positional_encoding(max_len, embed_dim)
    enc_x = add_positional_encoding(enc_embed, pos_enc)

    for _ in range(num_layers):
        enc_x = TransformerBlock(embed_dim, num_heads, ff_dim, rate)(enc_x)

    encoder_outputs = enc_x


    #######################
    # Decoder Section
    #######################

    decoder_inputs = Input(shape=(max_len,), name="decoder_inputs")
    dec_embed = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(decoder_inputs)
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


# Save the model
model.save("se2seq_model.keras")
print("Model saved")




#################################################################
####### PHASE 7: ANALYSIS
#################################################################

# Evaluate the model
loss, accuracy = model.evaluate([X_enc_test, X_dec_test], y_test)

print(f"Final Accuracy: {accuracy * 100:.2f}%")


# some plots
import matplotlib.pyplot as plt

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
plt.savefig('seq2seq_acc_loss.png')
print("Plot saved seq2seq_acc_loss.png")
