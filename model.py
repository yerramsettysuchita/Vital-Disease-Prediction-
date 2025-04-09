import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import joblib
import numpy as np

# Load data from CSV
df = pd.read_csv("C:/Users/DELL/Downloads/Vital Disease Prediction/vital_disease_prediction_dataset.csv")

# Print data info for debugging
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Convert Disease_Prediction to string type first to handle potential float values
df["Disease_Prediction"] = df["Disease_Prediction"].astype(str)

# Replace 'nan' strings with 'None' 
df["Disease_Prediction"] = df["Disease_Prediction"].replace('nan', 'None')

# Prepare features and labels
X = df.drop(columns=["Name", "Gender", "Disease_Prediction"])
y_raw = df["Disease_Prediction"].apply(lambda x: x.split(", ") if x != "None" else [])

# Scale numerical features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Convert to multi-label format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_raw)
print(f"Number of disease classes: {len(mlb.classes_)}")
print(f"Disease classes: {mlb.classes_}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y if y.shape[1] == 1 else None)

# Define best parameters for Random Forest (you can uncomment grid search if needed)
'''
param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
multi_rf = MultiOutputClassifier(rf)
grid_search = GridSearchCV(multi_rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
'''

# Train model with optimized parameters
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced', 
    random_state=42
)
multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_train, y_train)

# Evaluate model
y_pred = multi_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
h_loss = hamming_loss(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Hamming Loss: {h_loss:.4f}")

# Save model and all preprocessing components
model_path = "C:/Users/DELL/Downloads/Vital Disease Prediction/vital_disease_predictor.pkl"
mlb_path = "C:/Users/DELL/Downloads/Vital Disease Prediction/label_binarizer.pkl"
scaler_path = "C:/Users/DELL/Downloads/Vital Disease Prediction/scaler.pkl"

joblib.dump(multi_rf, model_path)
joblib.dump(mlb, mlb_path)
joblib.dump(scaler, scaler_path)

# Feature importance analysis
importances = np.mean([
    estimator.feature_importances_ for estimator in multi_rf.estimators_
], axis=0)
feature_imp = pd.DataFrame(
    sorted(zip(importances, X.columns), reverse=True), 
    columns=['Importance', 'Feature']
)
print("\nTop 10 most important features:")
print(feature_imp.head(10))

print(f"\nModel saved to: {model_path}")
print(f"Label binarizer saved to: {mlb_path}")
print(f"Scaler saved to: {scaler_path}")








