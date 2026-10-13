import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")
df = pd.read_csv("shell_company_data.csv")

# 1. Prepare Data
X = df.drop('is_fraud_clone', axis=1)
y = df['is_fraud_clone']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Data Preprocessing (Scaling)
# We must scale numerical data so models treat all features equally
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize the 3 Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 4. Train and Compare Models
print("\n--- Model Comparison ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc * 100:.2f}%")

# 5. Hyperparameter Tuning (Improving the Model)
print("\n--- Tuning Random Forest ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")
final_pred = best_model.predict(X_test_scaled)
print(f"Optimized Model Accuracy: {accuracy_score(y_test, final_pred) * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, final_pred))

# 6. Model Persistence (Save the best model and scaler)
with open('best_fraud_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Success! 'best_fraud_model.pkl' and 'scaler.pkl' saved to your folder.")