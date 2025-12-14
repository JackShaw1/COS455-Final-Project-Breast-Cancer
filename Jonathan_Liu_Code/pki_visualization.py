#!/usr/bin/env python3
"""
PKI Visualization Script
Visualizes Gaussian Mixture Models from saved PKI models
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import sys

# Load the GMM models from the pickle file
gmm_file = '/n/fs/vision-mix/jl0796/qcb/qcb455_project/augmented_data/gmm_models/Breast_GSE7904_gmm_models.pkl'
print(f"Loading GMM models from: {gmm_file}")

with open(gmm_file, 'rb') as f:
    gmm_models = pickle.load(f)

print(f"Loaded {len(gmm_models)} GMM models")
print(f"Column names (first 10): {list(gmm_models.keys())[:10]}")

# Select the first column with a 2-component GMM for visualization
selected_column = None
for col_name, gmm_info in gmm_models.items():
    if gmm_info['n_components'] == 2:
        selected_column = col_name
        selected_gmm_info = gmm_info
        break

if selected_column is None:
    # If no 2-component model, just use the first one
    selected_column = list(gmm_models.keys())[0]
    selected_gmm_info = gmm_models[selected_column]

# Extract the actual GMM model
selected_gmm = selected_gmm_info['model']
n_components = selected_gmm_info['n_components']

print(f"\nSelected column: {selected_column}")
print(f"Number of components: {n_components}")

# Extract GMM parameters
means = selected_gmm.means_.flatten()
stds = np.sqrt(selected_gmm.covariances_.flatten())
weights = selected_gmm.weights_

print(f"Means: {means}")
print(f"Standard deviations: {stds}")
print(f"Weights: {weights}")

# Set random seed for reproducibility
np.random.seed(42)

# Sample from the GMM with experimental error (5% of CDF)
n_samples = 1000
samples, _ = selected_gmm.sample(n_samples)
samples = samples.flatten()

# Add experimental error (randomize within 5% of CDF)
samples_with_error = samples.copy()
for i in range(len(samples_with_error)):
    # Calculate CDF at this point
    cdf_value = 0
    for j in range(n_components):
        cdf_value += weights[j] * norm.cdf(samples[i], means[j], stds[j])

    # Add random error within ±5% of CDF position
    error_factor = np.random.uniform(-0.05, 0.05)
    noisy_cdf = np.clip(cdf_value + error_factor, 0.01, 0.99)

    # Approximate inverse CDF by adding proportional noise
    # Simple approach: add noise proportional to the mixture std
    mixture_std = np.sqrt(np.sum(weights * (stds**2 + means**2)) - (np.sum(weights * means))**2)
    samples_with_error[i] += error_factor * mixture_std

# Select a random original data point to highlight
highlight_idx = np.random.randint(0, len(samples))
original_point = samples[highlight_idx]

# Calculate CDF at the original point
original_cdf = 0
for j in range(n_components):
    original_cdf += weights[j] * norm.cdf(original_point, means[j], stds[j])

print(f"\nHighlighted point:")
print(f"Original value: {original_point:.3f}")
print(f"CDF at original: {original_cdf:.3f}")

# Calculate the range of possible perturbed values (±5% CDF)
# We need to find x values corresponding to CDF ± 5%
def gmm_cdf_func(x):
    """Calculate GMM CDF at point x"""
    cdf = 0
    for j in range(n_components):
        cdf += weights[j] * norm.cdf(x, means[j], stds[j])
    return cdf

# Find x values for CDF bounds using numerical search
from scipy.optimize import brentq

lower_cdf = max(0.01, original_cdf - 0.05)
upper_cdf = min(0.99, original_cdf + 0.05)

# Search range
search_min = original_point - 3 * mixture_std
search_max = original_point + 3 * mixture_std

try:
    lower_x = brentq(lambda x: gmm_cdf_func(x) - lower_cdf, search_min, original_point)
    upper_x = brentq(lambda x: gmm_cdf_func(x) - upper_cdf, original_point, search_max)
except:
    # Fallback if root finding fails
    lower_x = original_point - 0.5
    upper_x = original_point + 0.5

print(f"Error range: [{lower_x:.3f}, {upper_x:.3f}]")
print(f"CDF range: [{lower_cdf:.3f}, {upper_cdf:.3f}]")

# Create the visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Plot histogram of samples
counts, bins, patches = ax.hist(samples_with_error, bins=50, density=True, alpha=0.6,
                                color='skyblue', edgecolor='black', label='Sampled Data (with 5% error)')

# Create x range for plotting the Gaussian components
x_min = samples_with_error.min() - 0.5 * samples_with_error.std()
x_max = samples_with_error.max() + 0.5 * samples_with_error.std()
x_range = np.linspace(x_min, x_max, 1000)

# Plot individual Gaussian components
colors = ['red', 'green', 'purple', 'orange']
gmm_total = np.zeros_like(x_range)

for i in range(n_components):
    gaussian_component = weights[i] * norm.pdf(x_range, means[i], stds[i])
    gmm_total += gaussian_component
    ax.plot(x_range, gaussian_component, color=colors[i % len(colors)],
            linewidth=2.5, label=f'Gaussian {i+1} (μ={means[i]:.3f}, σ={stds[i]:.3f}, w={weights[i]:.3f})')

# Plot combined GMM
ax.plot(x_range, gmm_total, 'b--', linewidth=2.5, label='Combined GMM')

# Highlight the original point and its error range
# Fill the error range
y_max = ax.get_ylim()[1]
ax.axvspan(lower_x, upper_x, alpha=0.2, color='yellow',
           label=f'±5% CDF Error Range')

# Mark the original point
ax.axvline(original_point, color='orange', linewidth=3, linestyle='-',
           label=f'Original Point ({original_point:.3f})')

# Add markers for the error bounds
ax.axvline(lower_x, color='orange', linewidth=2, linestyle=':', alpha=0.7)
ax.axvline(upper_x, color='orange', linewidth=2, linestyle=':', alpha=0.7)

# Formatting
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title(f'{n_components}-Component Gaussian Mixture Model\n'
             f'Column: {selected_column} (Breast_GSE7904)\nwith 5% Experimental Error',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Add text box with sampling information
textstr = (f'Samples: {n_samples}\n'
           f'Components: {n_components}\n'
           f'Error: ±5% CDF\n\n'
           f'Highlighted Point:\n'
           f'Original: {original_point:.3f}\n'
           f'Range: [{lower_x:.3f}, {upper_x:.3f}]\n'
           f'Span: {upper_x - lower_x:.3f}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()
output_file = '/n/fs/vision-mix/jl0796/qcb/qcb455_project/pki_gmm_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")
plt.show()

# Print statistics
print(f"\nGMM Statistics for column: {selected_column}")
for i in range(n_components):
    print(f"Component {i+1}: μ={means[i]:.3f}, σ={stds[i]:.3f}, weight={weights[i]:.3f}")
print(f"\nSample statistics:")
print(f"Sample mean: {np.mean(samples_with_error):.3f}")
print(f"Sample std: {np.std(samples_with_error):.3f}")
theoretical_mean = np.sum(weights * means)
print(f"Theoretical mean: {theoretical_mean:.3f}")
