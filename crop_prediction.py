import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

print("version pandas:", pd.__version__)
print("version numpy:", np.__version__)
print("version matplotlib:", matplotlib.__version__)
print("version seaborn:", sns.__version__)
print("version sklearn:", sklearn.__version__)



df = pd.read_csv(r"D:\Madhudada_brainwareuniversity\python\basicAiTools\Crop_recommendation.csv")
print(df.head())
print(df.info())
print(df.describe())
# Check for missing values
print(df.isnull().sum())
# label encoding for categorical variable 'label'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
# Define features and target variable
X = df.drop('label', axis=1)
y = df['label']
print(X.head())
print(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
# Predict on the test set
y_pred = rf_classifier.predict(X_test_scaled)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
#save the model
joblib.dump(rf_classifier, 'crop_recommendation_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print(df['label'].value_counts())




