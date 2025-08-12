import pandas as pd
import numpy as np

# Set the number of samples (patients) to generate
num_samples = 3

# Define the feature names
features = [
    'Age_at_diagnosis', 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 
    'MUC16', 'PIK3CA', 'NF1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 
    'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA'
]

# Generate random data for each feature
data = {
    'Age_at_diagnosis': np.random.randint(20, 80, size=num_samples),  # Random ages between 20 and 80
    'IDH1': np.random.choice([0, 1], size=num_samples),  # Random binary values (0 or 1)
    'TP53': np.random.choice([0, 1], size=num_samples),
    'ATRX': np.random.choice([0, 1], size=num_samples),
    'PTEN': np.random.choice([0, 1], size=num_samples),
    'EGFR': np.random.choice([0, 1], size=num_samples),
    'CIC': np.random.choice([0, 1], size=num_samples),
    'MUC16': np.random.choice([0, 1], size=num_samples),
    'PIK3CA': np.random.choice([0, 1], size=num_samples),
    'NF1': np.random.choice([0, 1], size=num_samples),
    'FUBP1': np.random.choice([0, 1], size=num_samples),
    'RB1': np.random.choice([0, 1], size=num_samples),
    'NOTCH1': np.random.choice([0, 1], size=num_samples),
    'BCOR': np.random.choice([0, 1], size=num_samples),
    'CSMD3': np.random.choice([0, 1], size=num_samples),
    'SMARCA4': np.random.choice([0, 1], size=num_samples),
    'GRIN2A': np.random.choice([0, 1], size=num_samples),
    'IDH2': np.random.choice([0, 1], size=num_samples),
    'FAT4': np.random.choice([0, 1], size=num_samples),
    'PDGFRA': np.random.choice([0, 1], size=num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the data to a CSV file
df.to_csv('sample_patient_data.csv', index=False)

print("Sample patient data has been generated and saved as 'synthetic_patient_data.csv'.")
