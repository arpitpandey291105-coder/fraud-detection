# Step 1 - Libraries Import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2 - Dataset Load (r string use karo Windows path ke liye)
df = pd.read_csv(r'C:\Users\ARPIT PANDEY\Desktop\fraud-detection\data\creditcard.csv')

# Step 3 - Basic Info of dataset
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nColumn Names:")
print(df.columns.tolist())

print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print(df['Class'].value_counts())

fraud_percent = df['Class'].value_counts()[1] / len(df) * 100
print(f"Fraud transactions: {fraud_percent:.4f}%")
print(df['Amount'].describe())

# Graph 1: Fraud vs Legit Count
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Legit Transactions')
plt.xticks([0,1], ['Legit', 'Fraud'])
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.savefig('notebooks/fraud_vs_legit.png')
plt.show()
print("Graph 1 saved!")

# Graph 2: Transaction Amount Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Count')
plt.savefig('notebooks/amount_distribution.png')
plt.show()
print("Graph 2 saved!")

# Graph 3: Fraud vs Legit Amount Comparison
plt.figure(figsize=(8,4))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Amount: Fraud vs Legit')
plt.xticks([0,1], ['Legit', 'Fraud'])
plt.savefig('notebooks/amount_boxplot.png')
plt.show()
print("Graph 3 saved!")