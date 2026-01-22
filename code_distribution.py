import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and Combine
df1 = pd.read_csv("code_bug_fix_pairs.csv")
df2 = pd.read_csv("bug_fix_pairs.csv")
df = pd.concat([df1, df2], ignore_index=True)

# 2. Calculate Lengths (using buggy_code as the primary constraint)
# We cast to string to handle any potential NaN values
df['lengths'] = df['buggy_code'].astype(str).apply(len)

# 3. Calculate Percentiles
percentile_at_512 = (df['lengths'] <= 512).mean() * 100
p95 = np.percentile(df['lengths'], 95)
p99 = np.percentile(df['lengths'], 99)
max_len = df['lengths'].max()

print(f"--- Dataset Length Analysis ---")
print(f"Total samples: {len(df)}")
print(f"Max character length found: {max_len}")
print(f"Percentile of data <= 512 chars: {percentile_at_512:.2f}%")
print(f"95% of data is shorter than: {p95:.1f} chars")
print(f"99% of data is shorter than: {p99:.1f} chars")

# 4. Plot Distribution
plt.figure(figsize=(12, 6))
plt.hist(df['lengths'], bins=100, color='teal', edgecolor='black', alpha=0.7)

# Add a vertical line for your current 512 threshold
plt.axvline(512, color='red', linestyle='--', linewidth=2, label=f'Current Limit (512)')


plt.title("Distribution of Code Sequence Lengths", fontsize=16)
plt.xlabel("Character Count", fontsize=14)
plt.ylabel("Number of Samples", fontsize=14)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.savefig("sequence_length_distribution.png")
print("\nSuccess! Plot saved as 'sequence_length_distribution.png'")