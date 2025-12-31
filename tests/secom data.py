import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load Data (Available from UCI or Kaggle)
# URL: https://archive.ics.uci.edu/ml/datasets/SECOM
# The data is separated by spaces, not commas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
df = pd.read_csv(url, sep=" ", header=None)

# 2. Preprocessing (Crucial for SPNs)
# A. Drop constant columns (sensors that never change)
df = df.loc[:, (df != df.iloc[0]).any()]

# B. Impute Missing Values (SECOM has many NaNs)
imputer = SimpleImputer(strategy="mean")
X_full = imputer.fit_transform(df.values)

# C. Select Top Features (Optional but recommended for speed)
# Select 20-50 features with highest variance to make Causal Discovery visible
# (Feeding 590 features to an SPN might be too slow for now)
variances = np.var(X_full, axis=0)
top_d_indices = np.argsort(variances)[-20:]  # Top 20 sensors
X_global = X_full[:, top_d_indices]

# D. Normalize (SPNs love 0-mean, 1-std)
scaler = StandardScaler()
X_global = scaler.fit_transform(X_global)

print(f"SECOM Benchmark Ready: {X_global.shape}")
# Now feed X_global into your TestFedCDH pipeline!
