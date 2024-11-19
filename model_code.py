# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the dataset
file_path = '/Users/apple/Desktop/phython/Disease_symptom_and_patient_profile_dataset .csv'
df = pd.read_csv(file_path)

# Encode binary categorical variables
binary_columns = [
    'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 
    'Sore Throat', 'Rash', 'Headache', 'Nausea', 
    'Vomiting', 'Diarrhea', 'Muscle Pain'
]

# One-hot encode other categorical variables
df = pd.get_dummies(df, columns=['Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable'], drop_first=True)

# Replace 'Yes' and 'No' with 1 and 0 respectively
df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0}).astype(int)

# Focus on the top 10 most common diseases
top_diseases = df['Disease'].value_counts().nlargest(10).index
df = df[df['Disease'].isin(top_diseases)]

# Split the dataset into features (X) and labels (y)
X = df.drop(['Disease'], axis=1)  # Features
y = df['Disease']  # Labels

# Handle imbalanced data using SMOTE
min_class_count = y.value_counts().min()
k_neighbors = min(min_class_count - 1, 5)  # Ensure k_neighbors is less than the smallest class count

if k_neighbors > 0:
    smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
else:
    X_res, y_res = X, y

# Standardize the features
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Initialize the Random Forest and Gradient Boosting classifiers
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)

# Expanded parameter grid for thorough tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Use 3-fold cross-validation for faster testing
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_gb = GridSearchCV(estimator=gb_classifier, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)

# Fit the models using the entire dataset
grid_search_rf.fit(X_res, y_res)
grid_search_gb.fit(X_res, y_res)

# Best parameters found by GridSearchCV
best_params_rf = grid_search_rf.best_params_
best_params_gb = grid_search_gb.best_params_

# Train the classifiers with the best parameters on the entire dataset
best_rf_classifier = grid_search_rf.best_estimator_
best_gb_classifier = grid_search_gb.best_estimator_

# Split the resampled data into training and testing sets for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train the classifiers with the best parameters on the training set
best_rf_classifier.fit(X_train, y_train)
best_gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = best_rf_classifier.predict(X_test)
y_pred_gb = best_gb_classifier.predict(X_test)

# Evaluate the models
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

classification_report_rf = classification_report(y_test, y_pred_rf)
classification_report_gb = classification_report(y_test, y_pred_gb)

confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
confusion_matrix_gb = confusion_matrix(y_test, y_pred_gb)

print(f"Random Forest Accuracy: {accuracy_rf}")
print("Random Forest Classification Report:\n", classification_report_rf)
print("Random Forest Confusion Matrix:\n", confusion_matrix_rf)

print(f"Gradient Boosting Accuracy: {accuracy_gb}")
print("Gradient Boosting Classification Report:\n", classification_report_gb)
print("Gradient Boosting Confusion Matrix:\n", confusion_matrix_gb)

# Perform cross-validation on the entire dataset for a more reliable performance estimate
# Random Forest Cross-Validation
rf_cv_classifier = RandomForestClassifier(random_state=42, **best_params_rf)
rf_cv_scores = cross_val_score(rf_cv_classifier, X_res, y_res, cv=5, scoring='accuracy')

# Gradient Boosting Cross-Validation
gb_cv_classifier = GradientBoostingClassifier(random_state=42, **best_params_gb)
gb_cv_scores = cross_val_score(gb_cv_classifier, X_res, y_res, cv=5, scoring='accuracy')

print(f"Random Forest Cross-Validation Accuracy: {rf_cv_scores.mean()} ± {rf_cv_scores.std()}")
print(f"Gradient Boosting Cross-Validation Accuracy: {gb_cv_scores.mean()} ± {gb_cv_scores.std()}")
