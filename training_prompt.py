import xarray as xr
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Install with: pip install imbalanced-learn
from xgboost import XGBClassifier  # Install with: pip install xgboost

# Function to extract radiance variables from a NetCDF file
def extract_radiance_variables(file_path):
    dataset = xr.open_dataset(file_path)
    radiance_vars = ['lightning_area_net_radiance', 'lightning_flash_radiance', 
                     'lightning_group_radiance', 'lightning_event_radiance', 
                     'lightning_event_bg_radiance']
    radiance_data = {}
    for var in radiance_vars:
        if var in dataset.variables:
            data = dataset[var].values
            radiance_data[var] = data.mean() if data.size > 1 else data.item()
        else:
            radiance_data[var] = None
    return radiance_data

# Function to process all NetCDF files in a directory and create a DataFrame
def process_nc_files(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".nc"):
            file_path = os.path.join(folder_path, filename)
            radiance_data = extract_radiance_variables(file_path)
            all_data.append(radiance_data)

    df = pd.DataFrame(all_data)
    df['event_occurs'] = [1 if i % 2 == 0 else 0 for i in range(len(df))]  # Dummy target variable
    return df

# Function to prepare data for training
def prepare_data(df):
    X = df.drop(columns=['event_occurs']).fillna(0)  # Handle missing data
    y = df['event_occurs']
    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Function to perform hyperparameter tuning with GridSearchCV
def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2'],  # Removed 'auto' as it's deprecated
        'bootstrap': [True, False]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to evaluate multiple models
def evaluate_models(X_train, X_test, y_train, y_test):
    # Models to evaluate
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='rbf', gamma='scale', C=1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(classification_report(y_test, y_pred))

# Paths to the train and test folders (updated paths)
train_folder = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\Lightning_Project\pyltg\pyltg\examples\train_files'
test_folder = r'C:\Users\jbull\OneDrive - Fayetteville State University\CSC490 SENIOR PROJECT\Lightning_Project\pyltg\pyltg\examples\test_files'

# Process NetCDF files for training and testing
train_df = process_nc_files(train_folder)
test_df = process_nc_files(test_folder)

# Prepare data for training and testing
X_train, _, y_train, _ = prepare_data(train_df)
_, X_test, _, y_test = prepare_data(test_df)

# Perform cross-validation with RandomForestClassifier
print("\nCross-validation with Random Forest:")
rf = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean() * 100:.2f}%")

# Perform hyperparameter tuning
print("\nTuning RandomForest with GridSearchCV:")
best_rf_model = tune_model(X_train, y_train)
y_pred = best_rf_model.predict(X_test)
print(f"Best RandomForest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Ensemble method with VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1))
], voting='soft')

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
print(f"\nVoting Classifier Accuracy: {accuracy_score(y_test, y_pred_voting) * 100:.2f}%")
print(classification_report(y_test, y_pred_voting))

# Evaluate multiple models
print("\nEvaluating multiple models on test data:")
evaluate_models(X_train, X_test, y_train, y_test)


