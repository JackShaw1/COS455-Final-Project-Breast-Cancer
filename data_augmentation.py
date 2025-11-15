import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed, cpu_count
import warnings
import time
import scipy.stats as stats
from scipy.optimize import brentq

warnings.filterwarnings('ignore')

def gmm_cdf(x, gmm):
    """
    Calculates the Cumulative Distribution Function (CDF) of a 1D Gaussian Mixture Model.
    """
    x_input = np.atleast_1d(x)
    if x_input.ndim > 1:
         x_input = x_input.flatten()
    
    x_reshaped = x_input[:, np.newaxis] # Shape (n_samples, 1)
    
    weights = gmm.weights_ # Shape (n_components,)
    means = gmm.means_.flatten() # Shape (n_components,)
    std_devs = np.sqrt(gmm.covariances_.flatten()) # Shape (n_components,)

    # Calculate CDF for each component, results in shape (n_samples, n_components)
    component_cdfs = stats.norm.cdf(x_reshaped, loc=means, scale=std_devs)
    
    # Apply weights, results in shape (n_samples,)
    total_cdf = np.dot(component_cdfs, weights)
    
    return total_cdf[0] if np.isscalar(x) else total_cdf

def gmm_ppf(p, gmm, bracket_min, bracket_max):
    """
    Calculates the Percent Point Function (Inverse CDF) of a 1D GMM using a root finder.
    Finds x such that gmm_cdf(x) = p.
    """
    def func_to_zero(x):
        return gmm_cdf(x, gmm) - p
    
    # Use brentq to find the root (fast & numerically stable)
    return brentq(func_to_zero, bracket_min, bracket_max, xtol=1e-5, rtol=1e-5)

# Main script
print("Loading data...")
df_original = pd.read_csv('Breast_GSE45827.csv')
df_subtyping = pd.read_csv('subtyping_by_clustering.csv') # Jack Shaw's subtypes

# Map sample IDs
sample_ids = df_original['samples'].values
sample_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}
subtyping_sample_ids = df_subtyping['sample_id'].values
indices_in_original = np.array([sample_to_idx[sid] for sid in subtyping_sample_ids if sid in sample_to_idx])
print(f"Found {len(indices_in_original)} samples from subtyping in original data")

# Numeric columns
numeric_cols = df_original.columns[2:].tolist()
print(f"Processing {len(numeric_cols)} columns...")

n_samples_per_row = 10  # 10x data

def simulate_column(col_idx, col):
    """
    Fits a GMM to a column and generates new samples based on the CDF perturbation logic.
    """
    start_time_col = time.time()
    
    # Get column data
    col_data = df_original[col].values
    col_data_reshape = col_data.reshape(-1, 1)

    # Fit 1- and 2-component GMM
    try:
        gmm1 = GaussianMixture(n_components=1, random_state=42, n_init=10).fit(col_data_reshape)
        bic1 = gmm1.bic(col_data_reshape)

        gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=10).fit(col_data_reshape)
        bic2 = gmm2.bic(col_data_reshape)

        gmm = gmm1 if bic1 < bic2 else gmm2
    except Exception:
        # Fallback if GMM fitting fails
        gmm = GaussianMixture(n_components=1, random_state=42, n_init=10).fit(col_data_reshape)

    # --- New Simulation Logic ---
    
    # Determine a wide bracket for the inverse CDF (PPF) search
    # Use data min/max and extend by 5 std devs to be safe
    col_std = np.std(col_data)
    if col_std == 0: # Handle columns with no variance
        col_std = 1e-3 # Add a tiny std dev
    bracket_min = col_data.min() - 5 * col_std
    bracket_max = col_data.max() + 5 * col_std
    
    # Ensure min and max are distinct
    if bracket_min >= bracket_max:
        bracket_min -= 1e-3
        bracket_max += 1e-3

    # Calculate the actual CDF values at the bracket endpoints
    try:
        cdf_at_min = gmm_cdf(bracket_min, gmm)
        cdf_at_max = gmm_cdf(bracket_max, gmm)
    except Exception:
        # Fallback if CDF fails even here
        cdf_at_min = 0.0
        cdf_at_max = 1.0

    # Ensure they are distinct for clipping
    if cdf_at_min >= cdf_at_max:
        cdf_at_min = 0.0
        cdf_at_max = 1.0
    
    # Small epsilon to ensure p is *strictly* inside the bracket [cdf_at_min, cdf_at_max]
    clip_epsilon = 1e-7 

    col_simulated = []
    for idx in indices_in_original:
        # 1. Get original value
        x_orig = col_data[idx]
        
        # 2. Calculate its CDF
        try:
            cdf_orig = gmm_cdf(x_orig, gmm)
        except Exception:
            cdf_orig = 0.5 # Fallback if CDF calculation fails

        # 3. Generate n_samples_per_row new samples
        for _ in range(n_samples_per_row):
            # 4. Get target CDF: cdf_orig +/- 2.5%
            perturbation = np.random.uniform(-0.025, 0.025)
            target_cdf = cdf_orig + perturbation
            
            # 5. Clip to valid CDF range [0, 1]
            # Clip to the *actual* dynamic CDF range of the bracket, not a static [0, 1]
            # This guarantees f(a) and f(b) will have different signs for brentq
            target_cdf = np.clip(target_cdf, cdf_at_min + clip_epsilon, cdf_at_max - clip_epsilon)

            # 6. Find x_new using inverse CDF (PPF)
            try:
                x_new = gmm_ppf(target_cdf, gmm, bracket_min, bracket_max)
            except Exception:
                # Fallback if PPF fails (e.g., root not in bracket, GMM is weird)
                # Revert to adding simple multiplicative noise to the original value
                x_new = x_orig * (1 + np.random.uniform(-0.025, 0.025))
            
            col_simulated.append(x_new)
    # --- End New Logic ---

    if (col_idx + 1) % 5000 == 0:
        elapsed = time.time() - start_time_col
        print(f"Processed column {col_idx + 1}/{len(numeric_cols)} in {elapsed:.2f}s")

    return np.array(col_simulated)

# Run job on all CPU cores in parallel
n_jobs = int(cpu_count()) 
print(f"Starting parallel processing on {n_jobs} cores")

start_time_total = time.time()
results = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(simulate_column)(col_idx, col) for col_idx, col in enumerate(numeric_cols)
)
total_elapsed = time.time() - start_time_total
print(f"All columns processed in {total_elapsed/60:.2f} minutes")

# Combine results
simulated_array = np.column_stack(results)
print(f"Generated {simulated_array.shape[0]} simulated samples with {simulated_array.shape[1]} features")

# Build DataFrame
df_simulated = pd.DataFrame(simulated_array, columns=numeric_cols)

# Repeat subtyping info
sample_info_repeated = []
for idx, row in df_subtyping.iterrows():
    if row['sample_id'] in sample_to_idx:
        for i in range(n_samples_per_row):
            sample_info_repeated.append({
                'original_sample_id': row['sample_id'],
                'subtype': row['subtype'],
                'cluster': row['cluster'],
                'simulated_idx': i
            })

df_sample_info = pd.DataFrame(sample_info_repeated)

# Combine
df_simulated_final = pd.concat([df_sample_info.reset_index(drop=True),
                                df_simulated.reset_index(drop=True)], axis=1)

# Save
output_file = 'Breast_GSE45827_simulated_10x_cdf.csv'
df_simulated_final.to_csv(output_file, index=False)
print(f"\nSimulated data saved to {output_file}")
print(f"Shape: {df_simulated_final.shape}")
