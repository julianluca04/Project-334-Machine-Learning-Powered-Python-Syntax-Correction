import tensorflow as tf
import pickle
import numpy as np
from tensorflow import keras

# PRE-REQUISITES

# Tokenizer Blueprint
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
    

# Transformer Encoder Blueprint

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
    

# Decoder Block
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
    

# Positional Encoding
def positional_encoding(max_len, d_model):
    pos_enc = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    return pos_enc


def add_positional_encoding(embedded_inputs, pos_enc):
    positional_encoding_tensor = tf.cast(pos_enc, dtype=tf.float32)
    positional_encoding_tensor = positional_encoding_tensor[tf.newaxis, ...]
    return keras.layers.add([embedded_inputs, positional_encoding_tensor])

# Load Files
with open("char_tokenizer_nomask.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

# Use custom_objects so Keras knows where to find your blocks
model = tf.keras.models.load_model("se2seq_nomask_model.keras", custom_objects={
    'TransformerBlock': TransformerBlock,
    'DecoderBlock': DecoderBlock
})




# LOOP
import tensorflow as tf
import pickle
import numpy as np

# Load Tokenizer (CharTokenizer class has to be pre-defined)
with open("char_tokenizer_nomask.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

# Load Model
model = tf.keras.models.load_model("se2seq_nomask_model.keras")

def predict_fix(buggy_code, max_len=512):
    # Encode input
    enc_input = tokenizer.encode(buggy_code)
    # Pad to match model's expected shape (1, 512)
    enc_input_padded = enc_input + [tokenizer.stoi["<pad>"]] * (max_len - len(enc_input))
    enc_input_tensor = tf.constant([enc_input_padded[:max_len]])

    # Start the decoder with <bos>
    dec_input = [tokenizer.stoi["<bos>"]]
    
    # Generate characters one by one (Inference Loop)
    for _ in range(max_len):
        # Pad decoder input to 512
        curr_dec_padded = dec_input + [tokenizer.stoi["<pad>"]] * (max_len - len(dec_input))
        curr_dec_tensor = tf.constant([curr_dec_padded[:max_len]])
        
        # Predict
        predictions = model([enc_input_tensor, curr_dec_tensor], training=False)
        
        # Get the character at the current position (the one we just added)
        predicted_id = tf.argmax(predictions[0, len(dec_input)-1, :]).numpy()
        
        if predicted_id == tokenizer.stoi["<eos>"]:
            break
            
        dec_input.append(predicted_id)
        
    return tokenizer.decode(dec_input)

# 3. THE INTERACTIVE LOOP
print("\n--- Syntax Fixer AI ---")
print("Enter your buggy code. Type 'quit' to exit.")

while True:
    user_input = input("\nBuggy Code: ")
    if user_input.lower() == 'quit':
        break
    
    try:
        fixed_code = predict_fix(user_input)
        print("\nPROPOSED FIX:")
        print(fixed_code)
    except Exception as e:
        print(f"Error: {e}")