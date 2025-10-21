import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('C:/Users/Adi Awasthi/crop_yield.csv')
df = df.dropna()

# Define feature columns
feature_cols = [
    'Ph', 'K', 'P', 'N', 'Zn', 'S',
    'QV2M-W', 'QV2M-Sp', 'QV2M-Su', 'QV2M-Au',
    'T2M_MAX-W', 'T2M_MAX-Sp', 'T2M_MAX-Su', 'T2M_MAX-Au',
    'T2M_MIN-W', 'T2M_MIN-Sp', 'T2M_MIN-Su', 'T2M_MIN-Au',
    'PRECTOTCORR-W', 'PRECTOTCORR-Sp', 'PRECTOTCORR-Su', 'PRECTOTCORR-Au',
    'WD10M', 'GWETTOP', 'CLOUD_AMT', 'WS2M_RANGE', 'PS'
]

# Generate weights matching the number of features
weights = np.random.uniform(0.5, 2.0, size=len(feature_cols))

# Simulate crop yield based on weighted sum of features + noise
np.random.seed(42)
df['crop_yield_tph'] = df[feature_cols].dot(weights) + np.random.normal(0, 100, size=len(df))

# Save updated dataset
df.to_csv('crop_yield.csv', index=False)
print("✅ Saved crop_yield.csv with feature-based simulated yield.")
