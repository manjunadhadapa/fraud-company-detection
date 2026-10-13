import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 business records
n_samples = 1000
fraud_ratio = 0.2 # 20% of the dataset will be fraudulent clones
n_fraud = int(n_samples * fraud_ratio)
n_legit = n_samples - n_fraud

# Generate data for Legitimate Companies (Class 0)
legit_data = {
    'ip_distance_metric': np.random.normal(loc=15.0, scale=5.0, size=n_legit), # Normal IP distribution
    'shared_director_count': np.random.poisson(lam=0.5, size=n_legit), # Very few shared directors
    'transaction_frequency': np.random.normal(loc=50.0, scale=15.0, size=n_legit), # Stable transactions
    'days_since_registration': np.random.uniform(365, 3650, size=n_legit), # Older companies
    'industry_code_overlap': np.random.uniform(0.0, 0.4, size=n_legit), # Low overlap with banned entities
    'is_fraud_clone': np.zeros(n_legit, dtype=int)
}

# Generate data for Fraudulent Clones (Class 1)
fraud_data = {
    'ip_distance_metric': np.random.normal(loc=2.0, scale=1.5, size=n_fraud), # IP very close to banned entities
    'shared_director_count': np.random.poisson(lam=3.5, size=n_fraud), # High overlap in board members
    'transaction_frequency': np.random.exponential(scale=100.0, size=n_fraud), # Erratic, high-volume transactions
    'days_since_registration': np.random.uniform(1, 180, size=n_fraud), # Newly registered
    'industry_code_overlap': np.random.uniform(0.6, 1.0, size=n_fraud), # High industry overlap
    'is_fraud_clone': np.ones(n_fraud, dtype=int)
}

# Combine and shuffle the dataset
df_legit = pd.DataFrame(legit_data)
df_fraud = pd.DataFrame(fraud_data)
df = pd.concat([df_legit, df_fraud]).sample(frac=1).reset_index(drop=True)

# Save to CSV for the rest of the project
df.to_csv("shell_company_data.csv", index=False)
print("Dataset successfully created and saved as 'shell_company_data.csv'\n")

# --- EXPLORATORY DATA ANALYSIS (EDA) ---
print("--- Dataset Overview ---")
print(df.head(), "\n")

print("--- Class Distribution ---")
print(df['is_fraud_clone'].value_counts(normalize=True) * 100, "\n")

# Visualization 1: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Visualization 2: Shared Directors vs IP Distance (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='ip_distance_metric', y='shared_director_count', hue='is_fraud_clone', palette='Set1', alpha=0.7)
plt.title("IP Distance vs Shared Directors by Fraud Status")
plt.tight_layout()
plt.show()