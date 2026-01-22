import random
import pandas as pd

def generate_pro_dataset(num_samples=50000):
    # Expanded pools to prevent the model from just memorizing 'x' and 'y'
    vars = ['count', 'total', 'idx', 'buffer_size', 'is_valid', 'temp_str', 'item_val', 'result', 'user_id', 'status_code']
    vals = ['0', '1', '10', '100', 'True', 'False', 'None', '"Success"', '"Error"', '[]']
    ops = ['==', '!=', '>', '<', '>=', '<=']
    
    # Typing Error Engine: Mimics finger slips
    def add_noise(text, p=0.08):
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < p:
                noise_type = random.random()
                if noise_type < 0.4: # Swap with neighbor letter
                    chars[i] = random.choice('abdefghilmnoprstuvwy')
                elif noise_type < 0.7: # Accidental deletion
                    chars[i] = ""
        return "".join(chars)

    dataset = []

    for _ in range(num_samples):
        v1, v2, v3 = random.sample(vars, 3)
        val1, val2 = random.sample(vals, 2)
        op1 = random.choice(ops)
        
        # Determine scenario
        dice = random.random()
        
        # 1. THE NESTED BOSS: Multiple colons needed
        if dice < 0.35:
            fixed = f"if {v1} {op1} {val1}: if {v2} != {val2}: print({v3})"
            buggy = f"if {v1} {op1} {val1} if {v2} != {val2} print({v3})"
            # Occasionally add a spelling typo to the nested bug
            if random.random() < 0.5: buggy = buggy.replace("print", "pritn")

        # 2. THE INLINE ACTION: Colon in the middle of text
        elif dice < 0.65:
            fixed = f"while {v1} < {val1}: {v1} += 1; {v2} = {v1}"
            buggy = f"while {v1} < {val1} {v1} += 1 {v2} = {v1}"

        # 3. SPACING & SPELLING CRASH:
        elif dice < 0.85:
            # Randomly picks a keyword to break
            keywords = [("for", "fpr"), ("print", "peint"), ("while", "whil"), ("if", "iff")]
            k_fix, k_bug = random.choice(keywords)
            fixed = f"{k_fix} {v1} in range({val1}): print({v1})"
            buggy = f"{k_bug} {v1} in range({val1}) print({v1})"

        # 4. IDENTITY MAPPING: 15% Clean Code
        else:
            fixed = f"if {v1} == {val1}: {v2} = {v3}"
            buggy = fixed

        # Final pass: random character noise to 10% of buggy strings
        if random.random() < 0.1 and buggy != fixed:
            buggy = add_noise(buggy)

        dataset.append({"buggy_code": buggy, "fixed_code": fixed})

    return pd.DataFrame(dataset)

# Generate and Save
df = generate_pro_dataset(60000) # Aiming high for 3 layers!
df.to_csv("advanced_python_data.csv", index=False)
print(f"Generated {len(df)} samples of deep structural code.")