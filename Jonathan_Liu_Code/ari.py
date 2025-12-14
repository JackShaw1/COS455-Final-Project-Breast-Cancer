# Calculation of Adjusted Rand Index (ARI)
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import pandas as pd

# Load the data
csv = pd.read_csv("subtyping_by_clustering.csv")
print(len(csv))
# Calculate ARI
labels_true = csv['subtype']
labels_cluster = csv['cluster']
ARI = adjusted_rand_score(labels_true, labels_cluster)
print(f"The Adjusted Rand Index of the new clusters is: {ARI}")




