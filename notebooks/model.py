# Libraries Import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load Data
df = pd.read_csv(r'C:\Users\ARPIT PANDEY\Desktop\fraud-detection\data\creditcard.csv')

# Preprocessing
X = df.drop('Class', axis=1)
y = df['Class']

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scaling
scaler = StandardScaler()
X_resampled['Amount'] = scaler.fit_transform(X_resampled[['Amount']])
X_resampled['Time'] = scaler.fit_transform(X_resampled[['Time']])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)

print("Data Ready!")
print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)

#Train Random Forest Model
print("\nTraining Random Forest Model...")
print("Please wait - this may take 2-3 minutes...")

rf_model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees 
    random_state=42,    
    n_jobs=-1          
)

rf_model.fit(X_train, y_train)

print("\nModel Training Complete!")
print("Total Trees in Forest:", rf_model.n_estimators)

# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predictions 
y_pred = rf_model.predict(X_test)

# Results print 
print("\n--- Model Evaluation ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Save Model
import pickle

pickle.dump(rf_model, open('model/fraud_model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))

print("\nModel Saved Successfully!")
print("Location: model/fraud_model.pkl")