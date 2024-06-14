import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Data
df = pd.read_csv('Crop_recommendation (1).csv')

# EDA
df.info()
df['label'].unique()
df['label'].value_counts()
plt.figure(figsize=(10,10))
df['label'].value_counts().plot(kind='pie', autopct="%.1f%%")
plt.show()

# Histograms
for col, color in zip(df.columns[:-1], ['blue', 'green', 'orange', 'purple', 'yellow', 'red', 'cyan']):
    sns.histplot(df[col], color=color)
    plt.title(f'Histogram of {col}')
    plt.show()

# KDE Plots
plt.figure(figsize=(12,12))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.kdeplot(df[col])
plt.show()

# QQ Plots
import scipy.stats as sm
plt.figure(figsize=(12,12))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sm.probplot(df[col], dist='norm', plot=plt)
plt.show()

# Outliers Detection
plt.figure(figsize=(12,12))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    df[[col]].boxplot()
plt.show()

# Skewness
df.iloc[:,:-1].skew()

# Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split Data
x = df.drop('label', axis=1)
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)

# Build Initial Model
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print()
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter Tuning
rf = RandomForestClassifier()
param_grid = {
    'n_estimators': np.arange(50, 200),
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(2, 25),
    'min_samples_split': np.arange(2, 25),
    'min_samples_leaf': np.arange(2, 25)
}

rscv_model = RandomizedSearchCV(rf, param_grid, cv=5)
rscv_model.fit(x_train, y_train)
best_rf_model = rscv_model.best_estimator_

# Evaluation of Tuned Model
y_pred = best_rf_model.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print()
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(best_rf_model, 'random_forest_model.pkl')

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')
