# Libraries Import
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Data Load
df = pd.read_csv(r'C:\Users\ARPIT PANDEY\Desktop\fraud-detection\data\creditcard.csv')
print("Data Loaded Successfully!")
print("Shape:", df.shape)

print("\nBefore SMOTE:")
print(df['Class'].value_counts())

X = df.drop('Class', axis=1)  # all columns except Class
y = df['Class']           

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(pd.Series(y_resampled).value_counts())
print("\nNew Shape:", X_resampled.shape)

# Feature Scaling
scaler = StandardScaler()

X_resampled['Amount'] = scaler.fit_transform(X_resampled[['Amount']])
X_resampled['Time'] = scaler.fit_transform(X_resampled[['Time']])

print("\nFeature Scaling Done!")
print("Amount column sample:", X_resampled['Amount'].head().values)

# Train Test 
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,        # 20% testing, 80% training
    random_state=42         
)

print("\nTrain Test Split Done!")
print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)
print("Training Fraud Count:", y_train.sum())
print("Testing Fraud Count:", y_test.sum())