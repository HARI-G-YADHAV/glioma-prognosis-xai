import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap

# Load the datasets
info_path = "dataset/TCGA_InfoWithGrade.csv"
mutations_path = "dataset/TCGA_GBM_LGG_Mutations_all.csv"
df_info = pd.read_csv(info_path)
df_mutations = pd.read_csv(mutations_path)

# Preprocessing: Normalize Gender and Grade encoding

# Mapping Gender
gender_mapping = {0: 0, 1: 1, 2: 1}
df_info["Gender"] = df_info["Gender"].map(gender_mapping)
df_mutations["Gender"] = df_mutations["Gender"].map(gender_mapping)

# Handle 'Grade' conversion
grade_mapping = {"LGG": 1, "HGG": 2}
df_mutations["Grade"] = df_mutations["Grade"].replace(grade_mapping)
df_mutations["Grade"] = pd.to_numeric(df_mutations["Grade"], errors="coerce").fillna(0).astype(int)

# Impute missing values (Example for 'Age_at_diagnosis')
imputer = SimpleImputer(strategy='median')
df_info['Age_at_diagnosis'] = imputer.fit_transform(df_info[['Age_at_diagnosis']])

# Standardization
scaler = StandardScaler()
df_info["Age_at_diagnosis"] = scaler.fit_transform(df_info[["Age_at_diagnosis"]])

# One-Hot Encoding for categorical columns
df_info = pd.get_dummies(df_info, columns=['Gender', 'Grade'], drop_first=True)

# Outlier detection using IQR
Q1 = df_info['Age_at_diagnosis'].quantile(0.25)
Q3 = df_info['Age_at_diagnosis'].quantile(0.75)
IQR = Q3 - Q1
df_info = df_info[(df_info['Age_at_diagnosis'] >= (Q1 - 1.5 * IQR)) & (df_info['Age_at_diagnosis'] <= (Q3 + 1.5 * IQR))]

import re
from sklearn.impute import SimpleImputer

# Load the dataset
df_mutations = pd.read_csv("dataset/TCGA_GBM_LGG_Mutations_all.csv")

# Function to extract numeric age from string (e.g., '51 years 108 days')
def extract_numeric_age(age_str):
    # Extract the first numeric value from the string
    match = re.match(r"(\d+)", str(age_str))
    if match:
        return int(match.group(1))  # Return the first numeric part
    return np.nan  # Return NaN if no numeric value is found

# Apply the function to clean the 'Age_at_diagnosis' column
df_mutations['Age_at_diagnosis'] = df_mutations['Age_at_diagnosis'].apply(extract_numeric_age)

# Impute missing values for numerical columns (e.g., 'Age_at_diagnosis')
numerical_columns = ['Age_at_diagnosis']  # List of numerical columns to impute
numerical_imputer = SimpleImputer(strategy='median')
df_mutations[numerical_columns] = numerical_imputer.fit_transform(df_mutations[numerical_columns])

# Handle categorical missing values (e.g., 'Gender', 'Primary_Diagnosis', 'Race')
categorical_columns = ['Gender', 'Primary_Diagnosis', 'Race']
categorical_imputer = SimpleImputer(strategy='most_frequent')
df_mutations[categorical_columns] = categorical_imputer.fit_transform(df_mutations[categorical_columns])

# One-Hot Encoding for categorical columns ('Gender', 'Primary_Diagnosis', 'Race')
df_mutations = pd.get_dummies(df_mutations, columns=['Gender', 'Primary_Diagnosis', 'Race'], drop_first=True)

# List of mutation columns
mutation_columns = ['IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 
                    'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 
                    'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']

# Mapping 'mutated' to 1 and 'non-mutated' to 0 for the mutation columns
for col in mutation_columns:
    # Check if the value is 'mutated', if so, set it to 1, otherwise 0
    df_mutations[col] = df_mutations[col].apply(lambda x: 1 if str(x).lower() == 'mutated' else 0)

# Outlier Detection using IQR for numerical columns
for column in numerical_columns:
    Q1 = df_mutations[column].quantile(0.25)
    Q3 = df_mutations[column].quantile(0.75)
    IQR = Q3 - Q1
    df_mutations = df_mutations[(df_mutations[column] >= (Q1 - 1.5 * IQR)) & 
                                 (df_mutations[column] <= (Q3 + 1.5 * IQR))]

# Reset index after preprocessing
df_mutations = df_mutations.reset_index(drop=True)


import seaborn as sns
import matplotlib.pyplot as plt

# List of major features to be included in the correlation matrix
major_features = [
    'Grade', 'Age_at_diagnosis','Gender_Female', 'Gender_Male',
    'Primary_Diagnosis_Astrocytoma, NOS', 'Primary_Diagnosis_Astrocytoma, anaplastic', 
    'Primary_Diagnosis_Glioblastoma', 'Primary_Diagnosis_Mixed glioma', 
    'Primary_Diagnosis_Oligodendroglioma, NOS', 'Primary_Diagnosis_Oligodendroglioma, anaplastic', 
    'Race_american indian or alaska native', 'Race_asian', 'Race_black or african american', 
    'Race_not reported', 'Race_white'
]

# Select only the columns corresponding to major features
major_features_df = df_mutations[major_features]

# Ensure the columns selected are numeric (if not, handle conversion as needed)
major_features_df = major_features_df.apply(pd.to_numeric, errors='coerce')

# Compute the correlation matrix for major features
correlation_matrix = major_features_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Major Features")
plt.show()

# List of major features to be included in the correlation matrix
major_features = [
    'Grade', 'Age_at_diagnosis', 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC',
    'MUC16', 'PIK3CA', 'Gender_Female', 'Gender_Male',
]

# Select only the columns corresponding to major features
major_features_df = df_mutations[major_features]

# Ensure the columns selected are numeric (if not, handle conversion as needed)
major_features_df = major_features_df.apply(pd.to_numeric, errors='coerce')

# Compute the correlation matrix for major features
correlation_matrix = major_features_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Major Features")
plt.show()

# Select major numerical features
major_features = ['Age_at_diagnosis']  # You can add more features as needed

# Create a boxplot for major numerical features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_mutations[major_features], palette='Set2')
plt.title('Box Plot for Major Features')
plt.show()

# 3. Histogram for 'Age_at_diagnosis'
plt.figure(figsize=(10, 6))
sns.histplot(df_mutations['Age_at_diagnosis'], kde=True, bins=30, color='skyblue')
plt.title('Histogram of Age_at_diagnosis')
plt.xlabel('Age_at_diagnosis')
plt.ylabel('Frequency')
plt.show()

# Combine Gender_Female and Gender_Male into a single column
df_mutations['Gender'] = df_mutations[['Gender_Female', 'Gender_Male']].idxmax(axis=1).str.replace('Gender_', '')

# Now create the count plot using the combined 'Gender' column
plt.figure(figsize=(10, 6))
sns.countplot(data=df_mutations, x='Gender', palette='Set1')
plt.title('Count Plot for Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming these are the encoded columns for 'Race'
race_columns = ['Race_american indian or alaska native', 
                'Race_asian', 
                'Race_black or african american', 
                'Race_not reported', 
                'Race_white']

# Sum the boolean values (True = 1, False = 0) for each race category
df_race_counts = df_mutations[race_columns].sum()

# Convert the series into a DataFrame for easier plotting
df_race_counts = df_race_counts.reset_index()
df_race_counts.columns = ['Race', 'Count']

# Create a count plot (barplot in this case)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_race_counts, x='Race', y='Count', palette='Set2')
plt.title('Count Plot of Race Categories')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()

