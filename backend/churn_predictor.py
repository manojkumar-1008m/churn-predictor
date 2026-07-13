

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('teleco dataset.csv')

# Clean
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Encode text columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds) * 100, 2), "%")
print(classification_report(y_test, preds))

# Top factors driving churn
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 5 churn factors:")
print(importances.nlargest(5))

import matplotlib.pyplot as plt

# Feature importance chart
top_features = importances.nlargest(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features.index, top_features.values, color='steelblue')
plt.xlabel('Importance Score')
plt.title('Top 10 Factors That Predict Customer Churn')
plt.tight_layout()
plt.savefig('churn_factors.png')
plt.show()

# Churn distribution
plt.figure(figsize=(6, 4))
df['Churn'].value_counts().plot(kind='bar', color=['steelblue', 'coral'])
plt.xticks([0, 1], ['Stays', 'Leaves'], rotation=0)
plt.title('Customer Churn Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model2 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
model2.fit(X_train, y_train)
preds2 = model2.predict(X_test)
print("XGBoost Accuracy:", round(accuracy_score(y_test, preds2) * 100, 2), "%")
# Predict a single at-risk customer
sample = X_test.iloc[0]
prediction = model.predict([sample])[0]
probability = model.predict_proba([sample])[0]

print("Customer will churn:", "YES" if prediction == 1 else "NO")
print(f"Confidence: {round(max(probability) * 100, 1)}%")

# Try a few customers to find a good demo example
for i in range(10):
    sample = X_test.iloc[i]
    prediction = model.predict([sample])[0]
    probability = model.predict_proba([sample])[0]
    print(f"Customer {i+1} — Churn: {'YES' if prediction == 1 else 'NO'} | Confidence: {round(max(probability) * 100, 1)}%")

# Clean demo prediction - Customer 3
sample = X_test.iloc[2].to_frame().T  # fixes the warning
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("=== LIVE CHURN PREDICTION ===")
print(f"Customer will churn: {'YES' if prediction == 1 else 'NO'}")
print(f"Confidence: {round(max(probability) * 100, 1)}%")
print(f"Action: {'Send retention offer immediately!' if prediction == 1 else 'Customer is safe.'}")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier # Import XGBoost Classifier
from collections import Counter # Required for scale_pos_weight

# ── 1. LOAD ──────────────────────────────────────────
df = pd.read_csv('teleco dataset.csv')

# Store a copy of the original dataframe for categorical mapping later
df_original_for_mapping = df.copy()

# ── 2. CLEAN ─────────────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Encode text columns and store LabelEncoders
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le # Store the fitted encoder

# ── 3. SPLIT ─────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight for handling imbalanced classes
# This is typically (count of negative samples) / (count of positive samples)
counter = Counter(y_train)
scale_pos_weight_value = counter[0] / counter[1]
print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# ── 4. TRAIN RandomForest Classifier ──────────────────
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# ── 5. EVALUATE RandomForest Classifier ──────────────
preds_rf = model_rf.predict(X_test)
print("\n" + "=" * 40)
print(f"  Accuracy (RandomForest): {round(accuracy_score(y_test, preds_rf) * 100, 2)}%")
print("=" * 40)
print(classification_report(y_test, preds_rf, target_names=['Stays', 'Churns']))

# ── 4.1. TRAIN XGBoost Classifier (with scale_pos_weight) ────────────────────
# Using parameters from a previous attempt, adding scale_pos_weight
model_xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, scale_pos_weight=scale_pos_weight_value)
model_xgb.fit(X_train, y_train)

# ── 5.1. EVALUATE XGBoost Classifier ─────────────────
preds_xgb = model_xgb.predict(X_test)
print("\n" + "=" * 40)
print(f"  Accuracy (XGBoost with scale_pos_weight): {round(accuracy_score(y_test, preds_xgb) * 100, 2)}%")
print("=" * 40)
print(classification_report(y_test, preds_xgb, target_names=['Stays', 'Churns']))

# ── 6. FEATURE IMPORTANCE CHART (RandomForest) ───────
importances_rf = pd.Series(model_rf.feature_importances_, index=X.columns)
top_features_rf = importances_rf.nlargest(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features_rf.index, top_features_rf.values, color='steelblue')
plt.xlabel('Importance Score')
plt.title('Top 10 Factors That Predict Customer Churn (RandomForest)')
plt.tight_layout()
plt.savefig('churn_factors_rf.png') # Changed filename to distinguish
plt.show()

# ── 7. CHURN DISTRIBUTION CHART ──────────────────────
plt.figure(figsize=(6, 4))
df['Churn'].value_counts().plot(kind='bar', color=['steelblue', 'coral'])
plt.xticks([0, 1], ['Stays', 'Leaves'], rotation=0)
plt.title('Customer Churn Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()

# ── 8. LIVE DEMO PREDICTION (XGBoost - using the potentially better model) ───
# Use the XGBoost model for demo prediction
sample = X_test.iloc[2].to_frame().T
prediction_xgb = model_xgb.predict(sample)[0]
probability_xgb = model_xgb.predict_proba(sample)[0]

print("\n=== LIVE CHURN PREDICTION (XGBoost) ===")
print(f"Customer will churn: {'YES' if prediction_xgb == 1 else 'NO'}")
print(f"Confidence: {round(max(probability_xgb) * 100, 1)}%")
print(f"Action: {'Send retention offer immediately!' if prediction_xgb == 1 else 'Customer is safe.'}")

# ── 9. TOP CHURN FACTORS (XGBoost) ──────────────
importances_xgb = pd.Series(model_xgb.feature_importances_, index=X.columns)
print("\n=== TOP 5 CHURN FACTORS (XGBoost) ===")
print(importances_xgb.nlargest(5).to_string())

print("\n=== INTERACTIVE CHURN PREDICTION ===")
print("Please enter the details for the new customer:")

# Prepare an empty dictionary to store user input
user_input = {}

# Get feature names from the trained model's input
feature_columns = X.columns.tolist()

for col in feature_columns:
    # Check if a LabelEncoder exists for the column (meaning it was an object type that was encoded)
    if col in label_encoders:
        le = label_encoders[col]
        options = le.classes_.tolist()
        while True:
            value = input(f"Enter {col} (Options: {', '.join(options)}): ")
            if value in options:
                user_input[col] = le.transform([value])[0]
                break
            else:
                print("Invalid input. Please choose from the provided options.")
    # For numerical features or features not handled by LabelEncoder
    else:
        while True:
            try:
                value = float(input(f"Enter {col}: "))
                user_input[col] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value.")

# Convert user input to a DataFrame, ensuring column order matches X_train
new_customer_df = pd.DataFrame([user_input], columns=X_train.columns)

# Make prediction using the best XGBoost model
prediction_new_customer = model_xgb.predict(new_customer_df)[0]
probability_new_customer = model_xgb.predict_proba(new_customer_df)[0]

print("\n--- PREDICTION RESULT ---")
print(f"Customer will churn: {'YES' if prediction_new_customer == 1 else 'NO'}")
print(f"Confidence: {round(max(probability_new_customer) * 100, 1)}%")
print(f"Action: {'Send retention offer immediately!' if prediction_new_customer == 1 else 'Customer is safe.'}")

print("Features used for model training:")
print(X.columns.tolist())

# 1. Sort the dataframe by 'tenure'
df_sorted = df.sort_values(by='tenure').reset_index(drop=True)

# 2. Calculate the split index (80% for training)
split_idx = int(len(df_sorted) * 0.8)

# 3. Create train and test DataFrames by slicing
train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

# 4. Separate features and target
# Assuming 'Churn' is the target. 'customerID' was already dropped in earlier cells.
X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

# 5. Print shapes to verify
print(f"Total rows: {len(df_sorted)}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Actual Split: {round(len(train_df)/len(df_sorted)*100, 2)}% Train / {round(len(test_df)/len(df_sorted)*100, 2)}% Test")

"""**Reasoning**:
I will engineer the new features 'Monthly_Total_Interaction' and 'Average_Monthly_Cost' for both training and testing sets as instructed, ensuring that I handle potential division by zero for the average cost calculation.


"""

# 1. Create Monthly_Total_Interaction: MonthlyCharges * tenure
X_train['Monthly_Total_Interaction'] = X_train['MonthlyCharges'] * X_train['tenure']
X_test['Monthly_Total_Interaction'] = X_test['MonthlyCharges'] * X_test['tenure']

# 2. Create Average_Monthly_Cost: TotalCharges / tenure
# Using a safe approach to avoid division by zero
X_train['Average_Monthly_Cost'] = X_train['TotalCharges'] / X_train['tenure']
X_test['Average_Monthly_Cost'] = X_test['TotalCharges'] / X_test['tenure']

# 3. Handle potential infinity or NaN from division by zero (tenure = 0)
X_train['Average_Monthly_Cost'] = X_train['Average_Monthly_Cost'].fillna(0).replace([float('inf'), float('-inf')], 0)
X_test['Average_Monthly_Cost'] = X_test['Average_Monthly_Cost'].fillna(0).replace([float('inf'), float('-inf')], 0)

# 4. Verify new features
print("Updated Features in X_train:")
print(X_train[['MonthlyCharges', 'tenure', 'TotalCharges', 'Monthly_Total_Interaction', 'Average_Monthly_Cost']].head())
print("\nAll columns:", X_train.columns.tolist())

from sklearn.preprocessing import StandardScaler

# 1. Define numerical columns to be scaled
num_cols = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Monthly_Total_Interaction', 'Average_Monthly_Cost']

# 2. Instantiate the StandardScaler
scaler = StandardScaler()

# 3. Fit on training data and transform both sets
# We fit only on X_train to prevent data leakage from the test set
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 4. Verify the scaling (means should be ~0 and std should be ~1 for train set)
print("Scaling complete.")
print("\nMean of scaled features in X_train:")
print(X_train[num_cols].mean().round(4))
print("\nStandard Deviation of scaled features in X_train:")
print(X_train[num_cols].std().round(4))

# Show sample of scaled data
X_train[num_cols].head()

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter

# 1. Re-calculate scale_pos_weight based on the current y_train distribution
# Current split is time-based, so the ratio might differ from the previous random split
counter = Counter(y_train)
scale_pos_weight_value = counter[0] / counter[1]
print(f"Re-calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# 2. Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma': [0, 0.1, 0.2]
}

# 3. Initialize the XGBClassifier with scale_pos_weight
xgb_base = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value, use_label_encoder=False, eval_metric='logloss')

# 4. Instantiate GridSearchCV
# Using 'f1' as the scoring metric to better handle the imbalanced churn class
grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)

# 5. Fit the search object to X_train and y_train
print("Starting Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

# 6. Identify and store the best model and parameters
best_xgb_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\nBest Hyperparameters found:")
print(best_params)
print(f"Best F1-Score: {grid_search.best_score_:.4f}")

from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# 1. Generate predictions using the best XGBoost model
preds_final = best_xgb_model.predict(X_test)

# 2. Print accuracy and detailed classification report
print("=" * 40)
print(f"  FINAL OPTIMIZED XGBOOST ACCURACY: {round(accuracy_score(y_test, preds_final) * 100, 2)}%")
print("=" * 40)
print(classification_report(y_test, preds_final, target_names=['Stays', 'Churns']))

# 3. Extract feature importances
importances_final = pd.Series(best_xgb_model.feature_importances_, index=X_train.columns)
top_10_features = importances_final.nlargest(10)

# 4. Visualize top 10 features
plt.figure(figsize=(10, 6))
top_10_features.sort_values().plot(kind='barh', color='teal')
plt.xlabel('Importance Score')
plt.title('Top 10 Features: Optimized XGBoost (Time-Based Split)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Comparison Insight
print("\n=== Feature Importance Comparison ===")
print(f"Baseline Top Factor: {importances_rf.idxmax()}")
print(f"Optimized Top Factor: {importances_final.idxmax()}")

new_features = ['Monthly_Total_Interaction', 'Average_Monthly_Cost']
for feat in new_features:
    rank = importances_final.sort_values(ascending=False).index.get_loc(feat) + 1
    print(f"New Feature '{feat}' Rank: {rank} out of {len(importances_final)}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stays', 'Churns'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from collections import Counter

# 1. Load and initial cleaning
df = pd.read_csv('teleco dataset.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# 2. Encode categorical variables and sort by tenure
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df_sorted = df.sort_values(by='tenure').reset_index(drop=True)

# 3. Manual 80/20 time-based split
split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx].copy()
test_df = df_sorted.iloc[split_idx:].copy()

X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

# 4. Feature Engineering
for dataset in [X_train, X_test]:
    dataset['Monthly_Total_Interaction'] = dataset['MonthlyCharges'] * dataset['tenure']
    dataset['Average_Monthly_Cost'] = (dataset['TotalCharges'] / dataset['tenure']).fillna(0).replace([float('inf'), float('-inf')], 0)

# 5. Scaling
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Monthly_Total_Interaction', 'Average_Monthly_Cost']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Calculate scale_pos_weight
counter = Counter(y_train)
scale_weight = counter[0] / counter[1]

# 7. Train Optimized XGBClassifier
model_final = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.01,
    gamma=0,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric='logloss'
)
model_final.fit(X_train, y_train)

# 8. Evaluation and Visualization
preds = model_final.predict(X_test)
print("=== FINAL REFINED MODEL REPORT ===")
print(classification_report(y_test, preds, target_names=['Stays', 'Churns']))

importances = pd.Series(model_final.feature_importances_, index=X_train.columns)
top_10 = importances.nlargest(10).sort_values()

plt.figure(figsize=(10, 6))
top_10.plot(kind='barh', color='darkorange')
plt.title('Top 10 Feature Importances (Optimized XGBoost)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

def preprocess_customer_input(raw_data):
    """
    Preprocesses raw customer input dictionary into a format ready for the XGBoost model.
    """
    # 1. Convert dictionary to DataFrame
    input_df = pd.DataFrame([raw_data])

    # 2. Feature Engineering
    input_df['Monthly_Total_Interaction'] = input_df['MonthlyCharges'] * input_df['tenure']
    input_df['Average_Monthly_Cost'] = (input_df['TotalCharges'] / input_df['tenure']).fillna(0).replace([float('inf'), float('-inf')], 0)

    # 3. Scaling numerical features
    # Using the 'scaler' object fitted during the consolidation step
    num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Monthly_Total_Interaction', 'Average_Monthly_Cost']
    input_df[num_cols_to_scale] = scaler.transform(input_df[num_cols_to_scale])

    # 4. Ensure column order matches X_train exactly
    # X_train.columns was defined in the consolidated pipeline cell
    input_df = input_df[X_train.columns]

    return input_df

# --- Test the helper function with a sample ---
sample_raw_input = {
    'gender': 1, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0, 'tenure': 12,
    'PhoneService': 1, 'MultipleLines': 0, 'InternetService': 1, 'OnlineSecurity': 0,
    'OnlineBackup': 0, 'DeviceProtection': 0, 'TechSupport': 0, 'StreamingTV': 0,
    'StreamingMovies': 0, 'Contract': 0, 'PaperlessBilling': 1, 'PaymentMethod': 2,
    'MonthlyCharges': 70.0, 'TotalCharges': 840.0
}

processed_sample = preprocess_customer_input(sample_raw_input)
print("Processed Sample (First 5 columns):")
print(processed_sample.iloc[:, :5])
print("\nTotal Features:", len(processed_sample.columns))



"""**Reasoning**:
I will create an interactive script using input() to collect customer data for all 19 original features, handling categorical inputs via stored LabelEncoders and validating numerical inputs with try-except blocks.


"""

print("=== INTERACTIVE CUSTOMER DATA COLLECTION ===")
print("Please provide the customer details below.\n")

# 1. Initialize empty dictionary for raw features
raw_customer_data = {}

# 2. Define the 19 original features (excluding the 2 engineered ones)
# Based on X.columns.tolist() from previous cells
original_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# 3. Iterate through features and collect input
for col in original_features:
    # Handle Categorical Columns using stored LabelEncoders
    if col in label_encoders:
        le = label_encoders[col]
        options = le.classes_.tolist()
        while True:
            val = input(f"Enter {col} ({', '.join(options)}): ").strip()
            if val in options:
                # Store the encoded integer value
                raw_customer_data[col] = le.transform([val])[0]
                break
            else:
                print(f"Invalid selection. Please choose from: {options}")

    # Handle Numerical Columns
    else:
        while True:
            try:
                val = input(f"Enter {col}: ").strip()
                raw_customer_data[col] = float(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value (e.g., 10.5 or 24).")

print("\n--- Data Collection Complete ---")
print("Raw features collected:", len(raw_customer_data))
print(raw_customer_data)
import numpy as np

# 1. Preprocess the raw input using the helper function
processed_input = preprocess_customer_input(raw_customer_data)

# 2. Generate prediction and probabilities
prediction = model_final.predict(processed_input)[0]
probabilities = model_final.predict_proba(processed_input)[0]

# 3. Calculate confidence
confidence = round(max(probabilities) * 100, 1)

# 4. Display results
print("\n" + "="*30)
print("   PREDICTION RESULTS")
print("="*30)
print(f"Customer will churn: {'YES' if prediction == 1 else 'NO'}")
print(f"Confidence Level:    {confidence}%")
print("-"*30)

# 5. Recommended Action
if prediction == 1:
    print("ACTION: High Risk! Send retention offer or discount immediately.")
else:
    print("ACTION: Low Risk. Continue standard engagement.")
print("="*30)

import numpy as np

# 1. Preprocess the raw input using the helper function
processed_input = preprocess_customer_input(raw_customer_data)

# 2. Generate prediction and probabilities
prediction = model_final.predict(processed_input)[0]
probabilities = model_final.predict_proba(processed_input)[0]

# 3. Calculate confidence
confidence = round(max(probabilities) * 100, 1)

# 4. Display results
print("\n" + "="*30)
print("   PREDICTION RESULTS")
print("="*30)
print(f"Customer will churn: {'YES' if prediction == 1 else 'NO'}")
print(f"Confidence Level:    {confidence}%")
print("-"*30)

# 5. Recommended Action
if prediction == 1:
    print("ACTION: High Risk! Send retention offer or discount immediately.")
else:
    print("ACTION: Low Risk. Continue standard engagement.")
print("="*30)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Reload and initial cleaning
df_audit = pd.read_csv('teleco dataset.csv')
df_audit['TotalCharges'] = pd.to_numeric(df_audit['TotalCharges'], errors='coerce')
df_audit.dropna(inplace=True)
df_audit.drop('customerID', axis=1, inplace=True)

# 2. Sort by tenure to simulate temporal sequence
df_audit = df_audit.sort_values(by='tenure').reset_index(drop=True)

# 3. Time-based split (80% Train, 20% Test)
split_idx = int(len(df_audit) * 0.8)
train_data = df_audit.iloc[:split_idx].copy()
test_data = df_audit.iloc[split_idx:].copy()

X_train_audit = train_data.drop('Churn', axis=1)
y_train_audit = train_data['Churn']
X_test_audit = test_data.drop('Churn', axis=1)
y_test_audit = test_data['Churn']

# 4. Initialize Encoders and Scalers
label_encoders_audit = {}
scaler_audit = StandardScaler()

# 5. Isolated Label Encoding
cat_cols = X_train_audit.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    # Fit only on training data
    le.fit(X_train_audit[col])
    X_train_audit[col] = le.transform(X_train_audit[col])

    # Transform test data, handling potential unseen labels by mapping to a safe value if they exist
    # In a real scenario, one might use a 'Unknown' category or skip transform
    X_test_audit[col] = X_test_audit[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    label_encoders_audit[col] = le

# 6. Isolated Numerical Scaling
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
# Fit ONLY on training set to avoid leakage
scaler_audit.fit(X_train_audit[num_cols])

X_train_audit[num_cols] = scaler_audit.transform(X_train_audit[num_cols])
X_test_audit[num_cols] = scaler_audit.transform(X_test_audit[num_cols])

print("Data Splitting and Preprocessing Audit Complete.")
print(f"Training samples: {len(X_train_audit)}")
print(f"Testing samples: {len(X_test_audit)}")
print("\nScale of 'tenure' in training (Mean/Std):", round(X_train_audit['tenure'].mean(), 2), "/", round(X_train_audit['tenure'].std(), 2))
print("Verification: Preprocessing isolation successful.")

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Calculate Average_Monthly_Cost and handle division by zero
X_train_audit['Average_Monthly_Cost'] = (X_train_audit['TotalCharges'] / X_train_audit['tenure']).fillna(0).replace([float('inf'), float('-inf')], 0)
X_test_audit['Average_Monthly_Cost'] = (X_test_audit['TotalCharges'] / X_test_audit['tenure']).fillna(0).replace([float('inf'), float('-inf')], 0)

# 2. Create Charges_Trend: MonthlyCharges - Average_Monthly_Cost
X_train_audit['Charges_Trend'] = X_train_audit['MonthlyCharges'] - X_train_audit['Average_Monthly_Cost']
X_test_audit['Charges_Trend'] = X_test_audit['MonthlyCharges'] - X_test_audit['Average_Monthly_Cost']

# 3. Create High_Risk_Fiber_No_Security binary feature
# Identify encoded values from existing encoders
fiber_val = label_encoders_audit['InternetService'].transform(['Fiber optic'])[0]
no_security_val = label_encoders_audit['OnlineSecurity'].transform(['No'])[0]

X_train_audit['High_Risk_Fiber_No_Security'] = ((X_train_audit['InternetService'] == fiber_val) & (X_train_audit['OnlineSecurity'] == no_security_val)).astype(int)
X_test_audit['High_Risk_Fiber_No_Security'] = ((X_test_audit['InternetService'] == fiber_val) & (X_test_audit['OnlineSecurity'] == no_security_val)).astype(int)

# 4. Create Service_Tier categorical interaction
# Combine as strings to create unique categories
X_train_audit['Service_Tier'] = X_train_audit['InternetService'].astype(str) + "_" + X_train_audit['Contract'].astype(str)
X_test_audit['Service_Tier'] = X_test_audit['InternetService'].astype(str) + "_" + X_test_audit['Contract'].astype(str)

# 5. Encode Service_Tier using a fresh LabelEncoder fitted only on training data
le_tier = LabelEncoder()
X_train_audit['Service_Tier'] = le_tier.fit_transform(X_train_audit['Service_Tier'])
# Transform test set, handling unseen tiers by mapping to -1 or a safe default if necessary
X_test_audit['Service_Tier'] = X_test_audit['Service_Tier'].map(lambda s: le_tier.transform([s])[0] if s in le_tier.classes_ else -1)

print("Advanced Feature Engineering Complete.")
print("New Features Created:", ['Average_Monthly_Cost', 'Charges_Trend', 'High_Risk_Fiber_No_Security', 'Service_Tier'])
print("X_train_audit Shape:", X_train_audit.shape)
display(X_train_audit[['Average_Monthly_Cost', 'Charges_Trend', 'High_Risk_Fiber_No_Security', 'Service_Tier']].head())

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# 1. Instantiate TimeSeriesSplit with 5 folds
tscv = TimeSeriesSplit(n_splits=5)

# 2. Encode categorical 'Churn' labels in y_train_audit to integers
le_target = LabelEncoder()
y_train_audit_encoded = le_target.fit_transform(y_train_audit)

print(f"Target classes identified: {le_target.classes_}")
print(f"Encoded Churn sample: {y_train_audit_encoded[:5]}\n")

# 3. Verify temporal validation setup by printing fold shapes
print("=== TimeSeriesSplit Fold Verification ===")
for i, (train_index, val_index) in enumerate(tscv.split(X_train_audit)):
    X_tr_fold, X_val_fold = X_train_audit.iloc[train_index], X_train_audit.iloc[val_index]
    print(f"Fold {i+1}: Training set shape: {X_tr_fold.shape}, Validation set shape: {X_val_fold.shape}")
    print(f"      Min Tenure index: {train_index[0]}, Max Tenure index: {val_index[-1]}")

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import numpy as np

# 1. Calculate initial scale_pos_weight
counts = np.bincount(y_train_audit_encoded)
initial_ratio = counts[0] / counts[1]
print(f"Initial scale_pos_weight ratio (No/Yes): {initial_ratio:.2f}")

# 2. Define parameter grid focusing on Recall and scale_pos_weight
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [initial_ratio, initial_ratio * 1.5, initial_ratio * 2.0]
}

# 3. Initialize XGBClassifier
xgb_recall_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# 4. Instantiate GridSearchCV with Recall scoring and TimeSeriesSplit
grid_search_recall = GridSearchCV(
    estimator=xgb_recall_model,
    param_grid=param_grid,
    scoring='recall',
    cv=tscv,
    n_jobs=-1,
    verbose=1
)

# 5. Fit GridSearchCV to the training data
print("Starting Hyperparameter Tuning for Recall...")
grid_search_recall.fit(X_train_audit, y_train_audit_encoded)

# 6. Extract results
best_recall_params = grid_search_recall.best_params_
best_recall_score = grid_search_recall.best_score_

print("\n=== GRID SEARCH RESULTS (RECALL FOCUS) ===")
print(f"Best Parameters: {best_recall_params}")
print(f"Best Validation Recall: {round(best_recall_score * 100, 2)}%")

"""# Task
Generate an optimized XGBoost model for the "teleco dataset.csv" focusing on recall. Use `GridSearchCV` with `TimeSeriesSplit` (5 folds) and a parameter grid including `n_estimators`, `max_depth`, `learning_rate`, and `scale_pos_weight` to address class imbalance. Evaluate the final model's performance on the held-out test set by printing a classification report and plotting a confusion matrix. Finally, visualize the top 10 feature importances to identify the key drivers of churn in this temporal framework.

## Configure and Run Hyperparameter Tuning

### Subtask:
Perform hyperparameter optimization for an XGBoost classifier using GridSearchCV and TimeSeriesSplit to maximize recall.
"""

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import numpy as np

# 1. Initialize XGBClassifier
xgb_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# 2. Define parameter grid relative to the calculated initial_ratio
# initial_ratio (No/Yes) was calculated in a previous step
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [initial_ratio, initial_ratio * 1.5, initial_ratio * 2.0]
}

# 3. Configure GridSearchCV with recall scoring and tscv (TimeSeriesSplit)
grid_search_recall = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='recall',
    cv=tscv,
    n_jobs=-1,
    verbose=1
)

# 4. Fit to the training data
print("Starting Hyperparameter Tuning (Recall Focus)...")
grid_search_recall.fit(X_train_audit, y_train_audit_encoded)

# 5. Extract and store the best model and parameters
best_recall_model = grid_search_recall.best_estimator_
best_recall_params = grid_search_recall.best_params_

# 6. Print results
print("\n=== GRID SEARCH RESULTS (RECALL FOCUS) ===")
print(f"Best Parameters: {best_recall_params}")
print(f"Best Validation Recall: {round(grid_search_recall.best_score_ * 100, 2)}%")

"""**Reasoning**:
I will evaluate the performance of the best recall model on the held-out test set by encoding the string labels in y_test_audit and generating a detailed classification report.


"""

from sklearn.metrics import classification_report, accuracy_score

# 1. Encode y_test_audit labels using the target encoder from the previous step
y_test_audit_encoded = le_target.transform(y_test_audit)

# 2. Generate predictions using the best recall model
test_preds = best_recall_model.predict(X_test_audit)

# 3. Print evaluation metrics
print("=" * 60)
print("  TEST SET EVALUATION: OPTIMIZED RECALL XGBOOST")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test_audit_encoded, test_preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_audit_encoded, test_preds, target_names=le_target.classes_))

# 4. Specifically highlight the Recall for Churners
# When output_dict=True, keys are class labels as strings if they are numeric
report_dict = classification_report(y_test_audit_encoded, test_preds, output_dict=True)
churn_recall = report_dict['1']['recall']
print(f"\nTargeted Metric (Recall for 'Yes'): {churn_recall:.2%}")

"""# Task
Evaluate the optimized XGBoost model (`best_recall_model`) on the held-out test set `X_test_audit` (from "teleco dataset.csv") by printing a comprehensive classification report and plotting a confusion matrix to verify the effectiveness of the recall-focused tuning. Additionally, visualize the top 10 feature importances to determine the impact of new features like `Average_Monthly_Cost` and `Charges_Trend` on predicting churn within this temporal context. Finally, conclude by summarizing the performance and recommending whether further data balancing techniques or feature engineering are necessary.

## Evaluate Best Recall Model on Test Set

### Subtask:
Assess the performance of the optimized XGBoost model on the held-out temporal test set to verify the effectiveness of the recall-focused tuning.

## Visualize Confusion Matrix

### Subtask:
Generate a confusion matrix plot for the optimized recall-focused model to visualize classification performance on the test set.

**Reasoning**:
I will generate a confusion matrix plot using sklearn's metrics to visualize the performance of the recall-optimized XGBoost model on the test set.
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Compute the confusion matrix
# y_test_audit_encoded and test_preds are available from the previous evaluation step
cm_recall = confusion_matrix(y_test_audit_encoded, test_preds)

# 2. Create the display object
disp = ConfusionMatrixDisplay(confusion_matrix=cm_recall, display_labels=le_target.classes_)

# 3. Plot the matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='YlGnBu', values_format='d')
plt.title('Confusion Matrix: Optimized Recall XGBoost (Test Set)')
plt.grid(False)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 1. Access feature importances from the best recall model
importances_recall = pd.Series(best_recall_model.feature_importances_, index=X_train_audit.columns)

# 2. Sort and select top 10
top_10_recall = importances_recall.nlargest(10).sort_values(ascending=True)

# 3. Generate horizontal bar plot
plt.figure(figsize=(10, 6))
top_10_recall.plot(kind='barh', color='mediumseagreen')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Top 10 Feature Importances (Recall-Optimized Model)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Analyze engineered features specifically
engineered_feats = ['Average_Monthly_Cost', 'Charges_Trend']
sorted_all = importances_recall.sort_values(ascending=False)

print("=== Engineered Feature Rankings ===")
for feat in engineered_feats:
    if feat in sorted_all.index:
        rank = sorted_all.index.get_loc(feat) + 1
        score = sorted_all[feat]
        print(f"{feat}: Rank {rank} out of {len(sorted_all)} (Score: {score:.4f})")
    else:
        print(f"{feat} not found in feature list.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Load Data
df = pd.read_csv('teleco dataset.csv')

# 2. Clean Data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# 3. Feature Engineering
# Handle tenure=0 to avoid division by zero
df['Average_Monthly_Cost'] = (df['TotalCharges'] / df['tenure']).replace([np.inf, -np.inf], 0).fillna(0)
df['Charges_Trend'] = df['MonthlyCharges'] - df['Average_Monthly_Cost']

# 4. Time-based Split (80/20 based on tenure)
df_sorted = df.sort_values(by='tenure').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)

train_df = df_sorted.iloc[:split_idx].copy()
test_df = df_sorted.iloc[split_idx:].copy()

X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn'].map({'Yes': 1, 'No': 0})
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn'].map({'Yes': 1, 'No': 0})

# 5. One-Hot Encoding with Strict Isolation
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit only on training data
encoded_train = encoder.fit_transform(X_train[cat_cols])
encoded_test = encoder.transform(X_test[cat_cols])

# Get feature names and create DataFrames
encoded_cols = encoder.get_feature_names_out(cat_cols)
X_train_encoded = pd.concat([X_train.drop(columns=cat_cols).reset_index(drop=True),
                             pd.DataFrame(encoded_train, columns=encoded_cols)], axis=1)
X_test_encoded = pd.concat([X_test.drop(columns=cat_cols).reset_index(drop=True),
                            pd.DataFrame(encoded_test, columns=encoded_cols)], axis=1)

# 6. Scaling with Strict Isolation
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Average_Monthly_Cost', 'Charges_Trend']
scaler = StandardScaler()

# Fit only on encoded training data
X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])

print("Preprocessing Complete.")
print(f"X_train shape: {X_train_encoded.shape}, X_test shape: {X_test_encoded.shape}")
print(f"New features sample:\n{X_train_encoded[['Average_Monthly_Cost', 'Charges_Trend']].head()}")

from imblearn.over_sampling import SMOTE

# 1. Initialize SMOTE with random_state for reproducibility
smote = SMOTE(random_state=42)

# 2. Print class distribution before SMOTE
print("Class distribution before SMOTE:")
print(y_train.value_counts())

# 3. Apply SMOTE to the preprocessed training data
X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train)

# 4. Print class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(y_train_smote.value_counts())

print(f"\nNew shape of training features: {X_train_smote.shape}")

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
import numpy as np

# 1. Use original time-ordered training data (X_train_encoded)
# TimeSeriesSplit requires the temporal order to be preserved
tscv_final = TimeSeriesSplit(n_splits=5)

# 2. Calculate scale_pos_weight to handle imbalance without SMOTE
# (Number of negative samples / Number of positive samples)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 3. Define the parameter grid
param_grid_final = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [pos_weight, pos_weight * 1.5]
}

# 4. Initialize the XGBClassifier
xgb_final = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

# 5. Create the GridSearchCV object with ROC-AUC scoring
grid_search_final = GridSearchCV(
    estimator=xgb_final,
    param_grid=param_grid_final,
    scoring='roc_auc',
    cv=tscv_final,
    n_jobs=-1,
    verbose=1
)

# 6. Fit to the time-ordered encoded training data
print('Starting GridSearchCV (Temporal Validation)...')
grid_search_final.fit(X_train_encoded, y_train)

# 7. Store the best model and print results
best_model_final = grid_search_final.best_estimator_
print('\n=== Hyperparameter Tuning Results (ROC-AUC) ===')
print(f'Best Parameters: {grid_search_final.best_params_}')
print(f'Best Cross-Validation ROC-AUC: {grid_search_final.best_score_:.4f}')

"""**Reasoning**:
I will perform the final evaluation of the optimized XGBoost model by generating predictions on the test set, printing a classification report, and calculating the ROC-AUC score as per the instructions.


"""

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# 1. Generate predictions and probability scores
test_preds = best_model_final.predict(X_test_encoded)
test_probs = best_model_final.predict_proba(X_test_encoded)[:, 1]

# 2. Print Classification Report
print("=== FINAL MODEL CLASSIFICATION REPORT ===")
print(classification_report(y_test, test_preds, target_names=['Stays', 'Churns']))

# 3. Calculate and print ROC-AUC score
roc_auc = roc_auc_score(y_test, test_probs)
print(f"Test Set ROC-AUC Score: {roc_auc:.4f}")

# 4. Plot Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stays', 'Churns'])
disp.plot(cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix')

# 5. Plot ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, test_probs)
roc_auc_val = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Receiver Operating Characteristic (ROC)')
ax2.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 6. Feature Importance Visualization
importances = pd.Series(best_model_final.feature_importances_, index=X_train_encoded.columns)
top_10_features = importances.nlargest(10).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh', color='teal')
plt.title('Top 10 Influential Features (Optimized XGBoost)')
plt.xlabel('Importance Score')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Note impact of engineered features
eng_feats = ['Average_Monthly_Cost', 'Charges_Trend']
print("\n=== Engineered Feature Importance ===")
for feat in eng_feats:
    if feat in importances.index:
        rank = importances.sort_values(ascending=False).index.get_loc(feat) + 1
        print(f"{feat}: Importance Score = {importances[feat]:.4f} (Ranked {rank} out of {len(importances)})")