import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
print("Loading data...")
df = pd.read_csv("/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE45827_simulated_50x_cdf.csv")

print(f"Original data shape: {df.shape}")
print(f"First 4 columns (metadata): {df.columns[:4].tolist()}")

# Get numeric columns (all columns after the first 4)
numeric_cols = df.columns[4:].tolist()
print(f"Number of numeric columns to normalize: {len(numeric_cols)}")

# Extract numeric data
X = df[numeric_cols].values

# Check for any NaN or infinite values
if np.any(np.isnan(X)):
    print(f"Warning: Found {np.sum(np.isnan(X))} NaN values in the data")
if np.any(np.isinf(X)):
    print(f"Warning: Found {np.sum(np.isinf(X))} infinite values in the data")

# Normalize using StandardScaler (mean=0, variance=1)
print("\nNormalizing data...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Verify normalization
print("\nVerification:")
print(f"Mean of normalized data (should be ~0): {np.mean(X_normalized, axis=0)[:5]}...")
print(f"Std of normalized data (should be ~1): {np.std(X_normalized, axis=0)[:5]}...")
print(f"Overall mean: {np.mean(X_normalized):.10f}")
print(f"Overall std: {np.std(X_normalized):.6f}")

# Create new dataframe with normalized values
df_normalized = df.copy()
df_normalized[numeric_cols] = X_normalized

# Save the normalized data
output_path = "/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE45827_simulated_50x_cdf_normalized.csv"
print(f"\nSaving normalized data to {output_path}...")
df_normalized.to_csv(output_path, index=False)

print("Done!")
print(f"\nOutput file: {output_path}")
print(f"Shape: {df_normalized.shape}")
