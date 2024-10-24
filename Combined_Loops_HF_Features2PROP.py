import logging
from pathlib import Path
from datetime import datetime
import pickle
import sys
import subprocess
import pkg_resources
import time
import joblib
from itertools import product

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

# Function to install a package if it's not already installed
def install_package(package):
    try:
        pkg_resources.get_distribution(package)
    except pkg_resources.DistributionNotFound:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package("openpyxl")

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths and file names
data_dir = Path(r"C:\Users\16036\OneDrive\Desktop\Investor Sight\Vulnerability\Model")
stata_dir = data_dir
output_dir = data_dir
data_file = "final_dataset3.dta"
#data_file = "Merged_dataset-2024-04-04.dta"

target_file = data_dir / "test_data2_2022prop.xlsx"

selected_features_summary = pd.read_excel(output_dir / "all_selected_features_summary929prop.xlsx")

# Constants
TARGET_VAR = "F1_factset_activism"
INDUSTRY_VAR = "sic_1_digit"
CLUSTER_ON = ["ln_mkv"]
CUTOFF_YEAR = 2014
MODEL_LABEL = "xgb"

'''NEIGHBORS_MAPPING = {
    9: 1, # Public Administration
    8: 29, # Services
    2: 28, # Construction
    4: 34, # Transportation
    6: 23, # Retail Trade
    7: 34, # Finance
    3: 32, # Manufacturing
    1: 29, # Mining
    5: 21, # Wholesale Trade
    0: 1 # Agriculture
}'''

NEIGHBORS_MAPPING = {
    9: 7, # Public Administration
    8: 9, # Services
    2: 17, # Construction
    4: 11, # Transportation
    6: 14, # Retail Trade
    7: 10, # Finance
    3: 7, # Manufacturing
    1: 15, # Mining
    5: 3, # Wholesale Trade
    0: 4 # Agriculture
}

NEIGHBORS_ALL = 1

df = pd.read_stata(stata_dir/data_file)
DATA_FIELDS = df.columns.tolist()
ID_FIELDS = DATA_FIELDS[:5]

'''param_grid = {
    'colsample_bytree': [0.8, 0.9],              # 2 values
    'gamma': [0.1, 0.2],                         # 2 values
    'learning_rate': [0.1, 0.15, 0.2],           # 3 values
    'max_depth': [3, 4],                         # 2 values
    'min_child_weight': [1, 2],                  # 2 values
    'n_estimators': [100, 150],                  # 2 values
    'reg_alpha': [0.05, 0.1],                    # 2 values
    'reg_lambda': [0, 0.5],                      # 2 values
    'subsample': [0.7, 0.8, 0.9]                 # 3 values
}'''

param_grid = {
    'colsample_bytree': 0.8,              # 2 values
    'gamma': 0.1,                         # 2 values
    'learning_rate': 0.1,           # 3 values
    'max_depth': 3,                         # 2 values
    'min_child_weight': 1,                  # 2 values
    'n_estimators': 100,                  # 2 values
    'reg_alpha': 0.05,                    # 2 values
    'reg_lambda': 0,                      # 2 values
    'subsample': 0.8                 # 3 values
}


'''
param_grid = {
    'colsample_bytree': [0.6, 0.8, 0.9],
    'gamma': [0.1, 0.2],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 200],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [0, 1],
    'subsample': [0.7, 0.8]
}

# Hyperparameters for XGBoost
param_grid = {
    'colsample_bytree': 0.9,
    'gamma': 0.2,
    'learning_rate': 0.2,
    'max_depth': 3,
    'min_child_weight': 1,
    'n_estimators': 100,
    'reg_alpha': 0.1,
    'reg_lambda': 0,
    'subsample': 0.8
}'''


# SIC 1 Digit mapping
sic_mapping = {
    0: "Agriculture",
    1: "Mining",
    2: "Construction",
    3: "Manufacturing",
    4: "Transportation, Communications, Electric, Gas, and Sanitary Services",
    5: "Wholesale Trade",
    6: "Retail Trade",
    7: "Finance, Insurance, and Real Estate",
    8: "Services",
    9: "Public Administration",
    "ALL": "ALL"
}

class DataProcessor:
    def __init__(self, data_file, data_fields, id_fields, cutoff_year=CUTOFF_YEAR):
        self.data_file = data_file
        self.data_fields = data_fields
        self.id_fields = id_fields
        self.cutoff_year = cutoff_year
        logging.info(f"DataProcessor initialized with {self.data_file}")

    def prepare_data(self, data):
        # Log the initial data state
        start_rows = len(data)
        logging.info("Removing rows with missing 'ln_mkv'")

        # Drop rows where 'ln_mkv' is NaN
        data = data.dropna(subset=['ln_mkv'])
        
        # Log how many rows were dropped
        end_rows = len(data)
        logging.info(f"Dropped {start_rows - end_rows} rows due to missing 'ln_mkv'")

        # Calculate the number of times a company has been targeted in the previous 3 years
        #data['activism_past_3_years'] = data.apply(lambda row: self.count_activism_past_years(data, row['gvkey'], row['year']), axis=1)
        data['hf_activism_past_3_years'] = data.apply(lambda row: self.count_hf_activism_past_years(data, row['gvkey'], row['year']), axis=1)
        data['hf_activism_last_year'] = data.apply(lambda row: self.count_hf_activism_last_year(data, row['gvkey'], row['year']), axis=1)

        # Add activism_past_3_years to DATA_FIELDS after it has been created
        #if 'activism_past_3_years' not in self.data_fields:
        #    self.data_fields.append('activism_past_3_years')
        if 'hf_activism_past_3_years' not in self.data_fields:
            self.data_fields.append('hf_activism_past_3_years')

        if 'hf_activism_last_year' not in self.data_fields:
            self.data_fields.append('hf_activism_last_year')

        # Generate sic1digit dummy variables
        data, dummy_columns = self.create_sic1digit_dummies(data)
        
        return data, dummy_columns
    
    def create_sic1digit_dummies(self, data):
        # Generate dummy variables for sic_1_digit
        sic_dummies = pd.get_dummies(data['sic_1_digit'], prefix='SIC')

        # Safely convert the dummy column names to integers before using them in the sic_mapping
        def safe_convert(value):
            try:
                return sic_mapping.get(int(float(value.split('_')[-1])), value)
            except ValueError:
                return value

        # Rename dummy columns using the mapping, safely converting to int where appropriate
        sic_dummies.columns = [safe_convert(col) for col in sic_dummies.columns]

        # Append dummy variables to the original dataframe
        data = pd.concat([data, sic_dummies], axis=1)

        logging.info(f"Created {len(sic_dummies.columns)} dummy variables for 'sic_1_digit'")

        return data, sic_dummies.columns
    
    def load_data(self):
        logging.info(f"Reading {self.data_file} into a DataFrame")
        df = pd.read_stata(self.data_file)
        logging.info(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        df.to_csv(output_dir / "raw_data2prop.csv", index=False)
        df = df[self.data_fields]
        logging.info(f"Data filtered for {len(self.data_fields)} fields")

        logging.info(f"Data filtered for year >= {self.cutoff_year}")
        df = df[df["year"] >= self.cutoff_year]

        logging.info("Preparation and filtering complete, proceeding without removing missing values.")
        df, dummy_columns = self.prepare_data(df)

        logging.info(f"New dummy variables created: {dummy_columns}")

        return df, dummy_columns
    
    @staticmethod
    def count_hf_activism_past_years(df, gvkey, year, years=3):
        start_year = year - years
        subset = df[(df['gvkey'] == gvkey) & (df['year'] >= start_year) & (df['year'] < year)]
        return subset['F1_factset_diligent_HF'].sum()
    
    @staticmethod
    def count_hf_activism_last_year(df, gvkey, year, years=1):
        start_year = year - years
        subset = df[(df['gvkey'] == gvkey) & (df['year'] >= start_year) & (df['year'] < year)]
        return subset['F1_factset_diligent_HF'].sum()
    
    @staticmethod
    def count_activism_past_years(df, gvkey, year, years=3):
        start_year = year - years
        subset = df[(df['gvkey'] == gvkey) & (df['year'] >= start_year) & (df['year'] < year)]
        return subset['F1_factset_diligent_HF'].sum()


class DataMatcher(BaseEstimator, TransformerMixin):
    def __init__(self, target_var, id_fields, group_fields=[INDUSTRY_VAR, 'year'], cluster_on=['ln_mkv'], neighbors=1):
        self.target_var = target_var
        self.id_fields = id_fields
        self.group_fields = group_fields
        self.cluster_on = cluster_on
        self.nearest_neighbors = neighbors

    def create_groups(self, df):
        groups = df.groupby(self.group_fields, group_keys=False, observed=False)
        return groups

    def match_simple(self, group):
        '''logging.info(f"Matching in group with size: {len(group)}")
        minority = group[group[self.target_var] == 1]
        majority = group[group[self.target_var] == 0]

        if minority.empty or majority.empty:
            logging.info("Insufficient data for matching in this group.")
            return pd.DataFrame()

        nn = NearestNeighbors(n_neighbors=min(len(majority), self.nearest_neighbors))
        nn.fit(majority[self.cluster_on])
        distances, indices = nn.kneighbors(minority[self.cluster_on])
        matched_majority_indices = indices.flatten()

        matched_majority = majority.iloc[matched_majority_indices]
        matched_samples = pd.concat([minority, matched_majority]).drop_duplicates()

        logging.info(f"Matched samples size: {len(matched_samples)}")'''
        logging.info(f"Matching in group with size: {len(group)}")

        minority = group[group[self.target_var] == 1]
        majority = group[group[self.target_var] == 0]

        if minority.empty or majority.empty:
            logging.info("Insufficient data for matching in this group.")
            return pd.DataFrame()

        # No filtering, take all majority samples
        matched_samples = pd.concat([minority, majority]).drop_duplicates()

        logging.info(f"Matched samples size: {len(matched_samples)}")
        return matched_samples

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        groups = self.create_groups(X)
        matched_samples = pd.DataFrame()
        for name, group in groups:
            matched_group = self.match_simple(group)
            matched_samples = pd.concat([matched_samples, matched_group], ignore_index=True)
        logging.info(f"Total matched data size after processing all groups: {len(matched_samples)}")
        return matched_samples


def market_cap_category(ln_mkv):
    mkv = np.exp(ln_mkv)
    if mkv >= 10**10:
        return "Large Cap (>$10B)"
    elif 2 * 10**9 <= mkv < 10**10:
        return "Mid Cap ($2B-$10B)"
    elif 250 * 10**6 <= mkv < 2 * 10**9:
        return "Small Cap ($250M-$2B)"
    elif  0 <= mkv < 250 * 10**6:
        return "Micro Cap (<$250M)"
    else:
        return "Unknown"

def train_and_save_model(train_features, train_labels, pipeline, industry, neighbors, selected_features, selected_features_all):
    # Check if the number of selected features is less than 5
    if len(selected_features) < 5:
        logging.warning(f"Less than 5 features selected for industry {industry}. Skipping model training for this industry and using the ALL model.")
        return False  # Indicate that no model was trained

    print(selected_features)

    # Select only the relevant features for this industry
    train_features = train_features[selected_features]

    # Train the model
    pipeline.fit(train_features, train_labels)

    # Save the model
    model_filename = output_dir / f"model2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
    joblib.dump(pipeline, model_filename)
    logging.info(f"Model saved for {industry} industry with {neighbors} neighbors")

    # Save the feature names
    feature_names_filename = output_dir / f"feature_names2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
    with open(feature_names_filename, 'wb') as f:
        pickle.dump(train_features.columns.tolist(), f)
    logging.info(f"Feature names saved for {industry} industry with {neighbors} neighbors")
    
    return True  # Indicate that a model was trained



def train_and_save_model_all(train_features, train_labels, pipeline, neighbors, selected_features):
    print(selected_features)
    train_features = train_features[selected_features]

    # Train the model
    pipeline.fit(train_features, train_labels)

    # Save the model
    model_filename = output_dir / f"model2_sic_ALL_neighbors_{neighbors}prop.pkl"
    joblib.dump(pipeline, model_filename)
    logging.info(f"Model saved for ALL with {neighbors} neighbors")

    # Save the feature names
    feature_names_filename = output_dir / f"feature_names2_sic_ALL_neighbors_{neighbors}prop.pkl"
    with open(feature_names_filename, 'wb') as f:
        pickle.dump(train_features.columns.tolist(), f)
    logging.info(f"Feature names saved for ALL with {neighbors} neighbors")


def predict_confidence_scores(test_data, sic_mapping):
    results = []

    for idx, row in test_data.iterrows():
        industry = row[INDUSTRY_VAR]
        neighbors = NEIGHBORS_MAPPING.get(industry, 1)
        model_filename = output_dir / f"model2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
        feature_names_filename = output_dir / f"feature_names2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"

        if not model_filename.exists() or not feature_names_filename.exists():
            logging.warning(f"No model or feature names found for {sic_mapping[industry]} industry with {neighbors} neighbors")
            continue

        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(feature_names_filename, 'rb') as f:
            feature_names = pickle.load(f)

        features = row.reindex(feature_names, axis=0)

        # Predict the confidence score
        confidence_score = model.predict_proba(features.values.reshape(1, -1))[0, 1]  # Probability of class 1
        results.append((row['gvkey'], industry, confidence_score))

    return results

'''def generate_sample_weights(df, target_var=TARGET_VAR):
    # Calculate class distribution
    class_counts = df[target_var].value_counts()

    # Calculate the minority and majority classes
    minority_class = class_counts.idxmin()  # Class with fewer samples
    majority_class = class_counts.idxmax()  # Class with more samples

    # Calculate the class weights
    class_weight_ratio = class_counts[majority_class] / class_counts[minority_class]

    # Initialize weights to 1 for all samples
    sample_weights = pd.Series(1, index=df.index)

    # Assign higher weight to minority class samples
    sample_weights[df[target_var] == minority_class] = class_weight_ratio

    return sample_weights'''



def generate_sample_weights(df, target_var=TARGET_VAR, weight_by_cap=True):
    # Calculate class distribution for the target variable
    class_counts = df[target_var].value_counts()

    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    sample_weights = pd.Series(1, index=df.index)

    class_weight_ratio = class_counts[majority_class] / class_counts[minority_class]
    sample_weights[df[target_var] == minority_class] = class_weight_ratio

    if weight_by_cap:
        for idx, row in df.iterrows():
            cap_category = market_cap_category(row['ln_mkv'])
            if cap_category == "Small Cap ($250M-$2B)" or cap_category == "Mid Cap ($2B-$10B)" or cap_category == "Micro Cap (<$250M)":
                sample_weights[idx] *= 2  # Increase weight for small/mid/micro caps (adjust the multiplier as needed)
    
    return sample_weights


# Function to calculate weights based on class imbalance for training
def generate_sample_weights_by_industry(train_data, target_col, industry_col, imbalance_weight_factor=1):
    industry_imbalance_ratios = train_data.groupby(industry_col)[target_col].mean()  # Mean gives class 1 ratio
    print("Industry-wise Imbalance Ratios:", industry_imbalance_ratios)
    
    sample_weights = pd.Series(1, index=train_data.index)  # Initialize with 1
    
    # Adjust sample weights for the minority class based on the imbalance ratio
    for industry, ratio in industry_imbalance_ratios.items():
        if ratio > 0:  # Avoid division by zero
            weight = 1 / ratio * imbalance_weight_factor  # Adjust weight based on the ratio
            sample_weights[train_data[industry_col] == industry] = train_data[train_data[industry_col] == industry][target_col].apply(lambda x: weight if x == 1 else 1)

    return sample_weights

def train_and_save_model_with_weights(train_features, train_labels, sample_weights, pipeline, industry, neighbors, selected_features):
    # Check if selected_features is empty
    if not selected_features:
        logging.error(f"Selected features for industry {industry} are empty. Skipping training.")
        return False

    logging.info(f"Training model for industry {industry} with neighbors {neighbors}.")
    logging.info(f"Selected Features: {selected_features}")

    # Ensure that the selected features exist in the training data
    missing_features = [feature for feature in selected_features if feature not in train_features.columns]
    if missing_features:
        logging.error(f"Missing features from train_features: {missing_features}")
        return False

    # Filter train_features by selected_features
    train_features = train_features[selected_features]

    # Debugging: Log input shapes
    logging.info(f"train_features shape: {train_features.shape}")
    logging.info(f"train_labels shape: {train_labels.shape}")
    logging.info(f"sample_weights shape: {sample_weights.shape}")

    # Ensure input shapes match
    if len(train_features) != len(train_labels) or len(train_labels) != len(sample_weights):
        logging.error("Mismatch in input shapes.")
        return False

    try:
        # Train the model with sample weights
        pipeline.named_steps[MODEL_LABEL].fit(train_features, train_labels, sample_weight=sample_weights)
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        return False

    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the trained model
        model_filename = output_dir / f"model2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
        logging.info(f"Saving model to {model_filename}")
        joblib.dump(pipeline, model_filename)

        # Save the feature names
        feature_names_filename = output_dir / f"feature_names2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
        logging.info(f"Saving feature names to {feature_names_filename}")
        with open(feature_names_filename, 'wb') as f:
            pickle.dump(train_features.columns.tolist(), f)

        logging.info(f"Model and feature names saved for {industry} industry with {neighbors} neighbors")
        return True  # Indicate success

    except Exception as e:
        logging.error(f"Error saving model or feature names: {str(e)}")
        return False



def train_and_save_model_all_with_weights(train_features, train_labels, sample_weights, pipeline, neighbors, selected_features, hyperparams):
    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prefix hyperparameters for the XGB model inside the pipeline
        hyperparams_prefixed = {f'xgb__{key}': value for key, value in hyperparams.items()}
        pipeline.set_params(**hyperparams_prefixed)

        # Select the relevant features
        train_features = train_features[selected_features]

        # Train the model with sample weights
        logging.info(f"Training model with {len(train_features)} features and {len(train_labels)} labels.")
        pipeline.named_steps[MODEL_LABEL].fit(train_features, train_labels, sample_weight=sample_weights)

        # Save the trained model
        model_filename = output_dir / f"model2_sic_ALL_neighbors_{NEIGHBORS_ALL}prop.pkl"
        logging.info(f"Saving model to {model_filename}")
        joblib.dump(pipeline, model_filename)
        logging.info(f"Model saved for ALL with {neighbors} neighbors")

        # Save the feature names
        feature_names_filename = output_dir / f"feature_names2_sic_ALL_neighbors_{NEIGHBORS_ALL}prop.pkl"
        logging.info(f"Saving feature names to {feature_names_filename}")
        with open(feature_names_filename, 'wb') as f:
            pickle.dump(train_features.columns.tolist(), f)
        logging.info(f"Feature names saved for ALL with {neighbors} neighbors")

        return True  # Return True if everything succeeded

    except Exception as e:
        logging.error(f"Failed to train or save the model: {e}")
        return False  # Return False if any error occurs



# Function to calculate feature influence
def feature_influence(feature_value, feature_name):
    # Example logic for determining influence
    return feature_value

# Function to calculate confusion matrix and metrics for a given threshold and mask
def calculate_confusion_matrix_and_metrics(threshold, mask, true_labels, predicted_probabilities):
    if len(predicted_probabilities) == 0:  # If the array is empty, return an empty confusion matrix
        return np.array([]), {}
    filtered_true_labels = np.array(true_labels)[mask]
    filtered_predicted_probabilities = np.array(predicted_probabilities)[mask]
    thresholded_predictions = [1 if prob >= threshold else 0 for prob in filtered_predicted_probabilities]
    conf_matrix = confusion_matrix(filtered_true_labels, thresholded_predictions)
    if conf_matrix.shape != (2, 2):
        return np.array([]), {}  # Return empty array and dict if the shape is not (2, 2)
    metrics = {
        'accuracy': accuracy_score(filtered_true_labels, thresholded_predictions),
        'precision': precision_score(filtered_true_labels, thresholded_predictions, average=None),
        'recall': recall_score(filtered_true_labels, thresholded_predictions, average=None),
        'f1_score': f1_score(filtered_true_labels, thresholded_predictions, average=None),
        #'auc_roc': roc_auc_score(filtered_true_labels, filtered_predicted_probabilities)  # Add AUC ROC here
    }
    return conf_matrix, metrics

# Function to calculate AUC ROC and best F1 score
# Function to calculate AUC ROC and best F1 score
def calculate_auc_and_best_f1(mask, true_labels, predicted_probabilities, thresholds):
    if len(predicted_probabilities) == 0:  # If the array is empty, return default values
        return 0.0, 0.0, 0.0, 0, 0, 0, 0

    filtered_true_labels = np.array(true_labels)[mask]
    filtered_predicted_probabilities = np.array(predicted_probabilities)[mask]

    # Check if there are both classes in the true labels
    if len(np.unique(filtered_true_labels)) == 1:
        logging.warning("Only one class present in y_true. ROC AUC score is not defined in that case.")
        auc = 0.0
    else:
        auc = roc_auc_score(filtered_true_labels, filtered_predicted_probabilities)

    best_f1 = 0.0
    best_threshold = 0.0
    best_recall = 0.0
    best_accuracy = 0.0
    best_precision = 0.0
    for threshold in thresholds:
        _, metrics = calculate_confusion_matrix_and_metrics(threshold, mask, true_labels, predicted_probabilities)
        f1_class_1 = metrics['f1_score'][1] if 'f1_score' in metrics and len(metrics['f1_score']) > 1 else 0.0
        if f1_class_1 > best_f1:
            best_f1 = f1_class_1
            best_accuracy = metrics['accuracy']
            best_recall = metrics['recall'][1] if 'recall' in metrics and len(metrics['recall']) > 1 else 0.0
            best_precision = metrics['precision'][1] if 'precision' in metrics and len(metrics['precision']) > 1 else 0.0
            best_threshold = threshold
    
    number_points = np.sum(mask)

    return auc, best_f1, best_threshold, number_points, best_recall, best_accuracy, best_precision


# Function to calculate top N hit rates
def calculate_top_n_hit_rate(df, n):
    sorted_df = df.sort_values(by='Confidence Score', ascending=False)  # Sort the entire DataFrame
    top_n = sorted_df.head(n)  # Select the top N rows
    hit_rate = (top_n[TARGET_VAR] == 1).mean()  # Convert to percentage
    return hit_rate

def calculate_top_n_hit_rate_all(df, n):
    sorted_df = df.sort_values(by='Confidence Score All', ascending=False)  # Sort the entire DataFrame
    top_n = sorted_df.head(n)  # Select the top N rows
    hit_rate = (top_n[TARGET_VAR] == 1).mean()  # Convert to percentage
    return hit_rate

def calculate_top_n_hit_rate2(df, n, target_var):
    """
    Calculate the hit rate for the top N rows, sorted by predicted probabilities.
    """
    sorted_df = df.sort_values(by='predicted_probabilities', ascending=False)
    top_n = sorted_df.head(n)  # Select the top N rows
    hit_rate = (top_n[target_var] == 1).mean()  # Hit rate (percentage of target = 1)
    return hit_rate

def main():
    stata_file_path = stata_dir / data_file
    logging.info(f"Loading data from {stata_file_path}")

    df, dummy_columns = DataProcessor(stata_file_path, DATA_FIELDS, ID_FIELDS, cutoff_year=CUTOFF_YEAR).load_data()
    print(dummy_columns)

    pd.Series(df.columns).to_csv('columnsprop.csv', index=False)

    selected_features_summary = pd.read_excel(output_dir / "all_selected_features_summary929prop.xlsx")

    #df = df[df['year'] != 2023.0]
    # Separate the data from 2022 as the testing set and save it as CSV
    df = df[df['russell3000'] == 1]
    df = df[df['year'] != 2023.0]

    test_data = df[df['year'] == 2022.0]
    test_data.to_csv('test_data2022.prop.csv')

    #companies_to_remove = df[(df[TARGET_VAR] == 1) & (df['year'] == 2021.0)]['gvkey'].tolist()
    #print(len(companies_to_remove))
    #test_data = test_data[~test_data['gvkey'].isin(companies_to_remove)]

    #test_data = test_data[test_data['russell3000'] == 1]

    train_data = df[df['year'] != 2022.0]
    test_data.to_excel(output_dir / "test_data2_2022prop.xlsx", index=False)

    # 
    test_data_all = df[df['year'] == 2022.0]
    train_data_all = df[df['year'] != 2022.0]
    matched_train_data = DataMatcher(target_var=TARGET_VAR, id_fields=ID_FIELDS, cluster_on=CLUSTER_ON, neighbors=NEIGHBORS_ALL).fit_transform(train_data_all)
    train_labels = matched_train_data[TARGET_VAR]
    train_features = matched_train_data.drop([TARGET_VAR, INDUSTRY_VAR], axis=1)

    selected_features_all = selected_features_summary["ALL"].dropna().tolist()
    print(selected_features_all)
    #train_features = train_features.drop(columns=["ticker", "sic_2_digit", "sic_3_digit", "exchg", "state", "conm", "F1_num_ownership_activism", "F1_num_factset_activism", "F1_num_factset_HFactivism"])
    train_features = train_features.drop(columns=["ticker", "sic_2_digit", "sic_3_digit", "state", "exchg", "conm"])
    train_features = matched_train_data[selected_features_all]
    #sample_weights_all = generate_sample_weights(train_data_all, TARGET_VAR, INDUSTRY_VAR, imbalance_weight_factor=1)
    sample_weights_all = generate_sample_weights(matched_train_data, TARGET_VAR)
    # sample_weights_all = generate_sample_weights(matched_train_data, TARGET_VAR, weight_by_cap=True)

    pipeline = Pipeline([(MODEL_LABEL, xgb.XGBClassifier(**param_grid))])
    train_and_save_model_all_with_weights(train_features, train_labels, sample_weights_all, pipeline, NEIGHBORS_ALL, selected_features_all, hyperparams=param_grid)

    '''keys, values = zip (*param_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    results = []  # To store the results for each combination

    for hyperparams in hyperparameter_combinations:
        logging.info(f"Testing hyperparameters: {hyperparams}")

        # Train and evaluate the model with the current hyperparameters
        pipeline = Pipeline([(MODEL_LABEL, xgb.XGBClassifier(**param_grid))])

        model_trained = train_and_save_model_all_with_weights(train_features, train_labels, sample_weights_all, pipeline, NEIGHBORS_ALL, selected_features_all, hyperparams)
        
        if not model_trained:
            logging.warning(f"Model training failed with hyperparameters: {hyperparams}")
            continue
        
        true_labels = test_data[TARGET_VAR]  # Extract true labels from the test data
        thresholds = np.linspace(0, 1, 100)

        # Get predictions and calculate metrics on the test set
        predicted_probabilities = pipeline.named_steps[MODEL_LABEL].predict_proba(test_data[selected_features_all])[:, 1]
        auc, best_f1, best_threshold, number_points, best_recall, best_accuracy, best_precision = calculate_auc_and_best_f1(np.ones(len(true_labels), dtype=bool), true_labels, predicted_probabilities, thresholds)

        test_data['predicted_probabilities'] = predicted_probabilities
        test_data['true_labels'] = true_labels

        # Calculate top N hit rates
        top_10_hit_rate = calculate_top_n_hit_rate2(test_data, 10, 'true_labels')
        top_20_hit_rate = calculate_top_n_hit_rate2(test_data, 20, 'true_labels')
        top_30_hit_rate = calculate_top_n_hit_rate2(test_data, 30, 'true_labels')
        top_50_hit_rate = calculate_top_n_hit_rate2(test_data, 50, 'true_labels')
        top_100_hit_rate = calculate_top_n_hit_rate2(test_data, 100, 'true_labels')

        # Store the metrics along with the hyperparameters
        results.append({
            'Hyperparameters': hyperparams,
            'AUC': auc,
            'Best F1': best_f1,
            'Best Threshold': best_threshold,
            'Best Recall': best_recall,
            'Best Accuracy': best_accuracy,
            'Best Precision': best_precision,
            'Number of Data Points': number_points,
            'Top 10 Hit Rate': top_10_hit_rate,
            'Top 20 Hit Rate': top_20_hit_rate,
            'Top 30 Hit Rate': top_30_hit_rate,
            'Top 50 Hit Rate': top_50_hit_rate,
            'Top 100 Hit Rate': top_100_hit_rate
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    # Save the DataFrame to an Excel file
    results_df.to_excel(output_dir / "hyperparameter_tuning_resultsprop.xlsx", index=False)
    logging.info(f"Saved hyperparameter tuning results to {output_dir / 'hyperparameter_tuning_resultsprop.xlsx'}")
    '''


    # Process each industry within the training data
    industries = train_data[INDUSTRY_VAR].unique()
    for industry in sorted(industries):
        logging.info(f"Running model on {sic_mapping[industry]} industry")
        df_industry = train_data[train_data[INDUSTRY_VAR] == industry]

        neighbors = NEIGHBORS_MAPPING.get(industry, 1)  # Default to 1 neighbor if not specified
        matched_train_data = DataMatcher(target_var=TARGET_VAR, id_fields=ID_FIELDS, cluster_on=CLUSTER_ON, neighbors=neighbors).fit_transform(df_industry)

        if matched_train_data.empty:
            logging.warning(f"No matched data for {sic_mapping[industry]} industry with {neighbors} neighbors")
            continue

        train_labels = matched_train_data[TARGET_VAR]
        selected_features = selected_features_summary[sic_mapping[industry]].dropna().tolist()
        print(selected_features)
        train_features = matched_train_data[selected_features]

        #train_to_auto = matched_train_data
        #train_to_auto.to_csv(output_dir / f"matched_train2_data_{sic_mapping[industry]}.csv", index=False)
        #train_features = matched_train_data.drop([TARGET_VAR, INDUSTRY_VAR], axis=1)
        #train_features = train_features.drop(columns=["ticker", "sic_2_digit", "sic_3_digit", "exchg", "state", "conm", "F1_num_ownership_activism", "F1_num_factset_activism", "F1_num_factset_HFactivism"])
        #train_features = train_features.drop(columns=["ticker", "sic_2_digit", "sic_3_digit", "state", "exchg", "conm", "F1_num_factset_HFactivism", "F1_num_factset_activism"])

        # Define the important feature and weight factor
        #weight_feature = 'stock_ret'
        #base_weight = 1
        #weight_factor = 2|
        sample_weights = generate_sample_weights(matched_train_data, TARGET_VAR, weight_by_cap=True)


        pipeline = Pipeline([(MODEL_LABEL, xgb.XGBClassifier(**param_grid))])
        #train_and_save_model_with_weights(train_features, train_labels, pipeline, industry, neighbors, important_feature, base_weight, weight_factor)
        model_trained = train_and_save_model_with_weights(train_features, train_labels, sample_weights, pipeline, industry, neighbors, selected_features)


    if not model_trained:
        logging.info(f"Using the ALL model for industry {sic_mapping[industry]} due to insufficient features.")
        # Handle prediction using ALL model later during prediction step
    
    # Load the confidence scores
    target_df = pd.read_excel(target_file)
    target_scores = target_df["F1_factset_activism"]

    test_data = pd.read_excel(output_dir / "test_data2_2022prop.xlsx")

    # Take a few rows from the test set for prediction
    # test_sample = test_data.sample(n=10, random_state=42)

    test_sample = test_data
    test_sample.to_excel(output_dir / "test_sample2prop.xlsx", index=False)

    # Predict confidence scores
    confidence_scores = []
    true_labels = []
    predicted_labels = []
    predicted_probabilities = []
    predicted_labels_all = []
    predicted_probabilities_all = []
    industries = []

    for _, row in test_sample.iterrows():
        
        industry = row[INDUSTRY_VAR]
        industries.append(industry)
        neighbors = NEIGHBORS_MAPPING.get(industry, 1)

        model_filename_all = output_dir / f"model2_sic_ALL_neighbors_{NEIGHBORS_ALL}prop.pkl"
        feature_names_filename_all = output_dir / f"feature_names2_sic_ALL_neighbors_{NEIGHBORS_ALL}prop.pkl"
        with open(model_filename_all, 'rb') as model_file:
            model_all = pickle.load(model_file)
        with open(feature_names_filename_all, 'rb') as f:
            feature_names_all = pickle.load(f)

        features = row.drop([TARGET_VAR])
        features = features.reindex(feature_names_all, axis=1)

        confidence_score_all = model_all.predict_proba(features.values.reshape(1, -1))[0][1]  # Probability of class 1
        predicted_label_all = model_all.predict(features.values.reshape(1, -1))[0]  # Predicted label      

        # Check if an industry-specific model exists and was trained
        industry_model_filename = output_dir / f"model2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
        industry_feature_names_filename = output_dir / f"feature_names2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
        print(industry_model_filename)
        print(industry_feature_names_filename)
        if industry_model_filename.exists() and industry_feature_names_filename.exists():
            model_filename = industry_model_filename
            feature_names_filename = industry_feature_names_filename
        else:
            logging.warning(f"No specific model found for {sic_mapping[industry]} industry, using ALL model instead.")

        # Load the model and feature names
        try:
            with open(model_filename, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(feature_names_filename, 'rb') as f:
                feature_names = pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load model or features: {e}")
            continue  # Skip to the next row if loading fails

        features = row.reindex(feature_names, axis=1)
        
        confidence_score = model.predict_proba(features.values.reshape(1, -1))[0][1]  # Probability of class 1
        predicted_label = model.predict(features.values.reshape(1, -1))[0]  # Predicted label

        # Also predict using the ALL model for comparison
        with open(output_dir / f"model2_sic_ALL_neighbors_{NEIGHBORS_ALL}prop.pkl", 'rb') as model_file_all:
            model_all = pickle.load(model_file_all)
        with open(output_dir / f"feature_names2_sic_ALL_neighbors_{NEIGHBORS_ALL}prop.pkl", 'rb') as f_all:
            feature_names_all = pickle.load(f_all)

        features_all = row.reindex(feature_names_all, axis=1)
        confidence_score_all = model_all.predict_proba(features_all.values.reshape(1, -1))[0][1]
        predicted_label_all = model_all.predict(features_all.values.reshape(1, -1))[0]

        # Create dictionary for current row
        row_dict = {
            "Ticker": row['ticker'],
            "Gvkey": row['gvkey'],
            "Confidence Score": confidence_score,
            "Confidence Score All": confidence_score_all,
            "SIC 1 digit name": sic_mapping[industry],
            "SIC 1 digit": row[INDUSTRY_VAR],
            "SIC 2 digit": row['sic_2_digit'],
            "SIC 3 digit": row['sic_3_digit'],
            "Market Cap Category": market_cap_category(row['ln_mkv']),
            "Vulnerability": "High" if confidence_score >= 0.5 else "Mid" if confidence_score >= 0.2 else "Low"
        }

        # Add feature influences
        for feature in feature_names:
            if feature in row.index:
                row_dict[f"{feature}"] = feature_influence(row[feature], feature)
            else:
                logging.warning(f"Feature '{feature}' not found in the current data row.")
                row_dict[f"{feature}"] = None

        confidence_scores.append(row_dict)

        true_labels.append(row[TARGET_VAR])
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(confidence_score)

        predicted_probabilities_all.append(confidence_score_all)
        predicted_labels_all.append(predicted_label_all)

    # Convert confidence scores to DataFrame
    confidence_scores_df = pd.DataFrame(confidence_scores)
    max = len(confidence_scores_df)
    # Calculate top N hit rates
    top_n_values = [10, 20, 30, 50, 100, max]
    hit_rates = {f'Top {n} Hit Rate': [] for n in top_n_values}
    print(industries)
    confidence_scores_df.insert(2, "F1_factset_activism", target_scores)

    for industry in set(industries):
        print("O")
        print(industry)
        industry_df = confidence_scores_df[confidence_scores_df['SIC 1 digit'] == industry]
        for n in top_n_values:
            hit_rate = calculate_top_n_hit_rate(industry_df, n)
            hit_rates[f'Top {n} Hit Rate'].append(hit_rate)
    
    # Calculate combined hit rates
    for n in top_n_values:
        hit_rate = calculate_top_n_hit_rate(confidence_scores_df, n)
        hit_rates[f'Top {n} Hit Rate'].append(hit_rate)

    for n in top_n_values:
        hit_rate = calculate_top_n_hit_rate_all(confidence_scores_df, n)
        hit_rates[f'Top {n} Hit Rate'].append(hit_rate)

    # Save hit rates to DataFrame
    hit_rates_df = pd.DataFrame(hit_rates)
    hit_rates_df['Industry'] = list(set(industries)) + ['Combined'] + ['All']

    # Save results to Excel
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    auc_roc_scores = []
    best_f1_scores = []

    with pd.ExcelWriter(output_dir / "confidence_scoresnew4baseline2prop.xlsx") as writer:
        confidence_scores_df.to_excel(writer, sheet_name="Confidence Scores", index=False)

        # Define combined_mask
        combined_mask = [True] * len(true_labels)

        # Calculate and save AUC ROC scores and best F1 scores to the second sheet
        auc, best_f1, best_threshold, number_points, best_recall, best_accuracy, best_precision = calculate_auc_and_best_f1(combined_mask, true_labels, predicted_probabilities, thresholds)
        all_hit_rates = {f'Top {n} Hit Rate': hit_rates[f'Top {n} Hit Rate'][-1] for n in top_n_values}
        auc_roc_scores.append({"Industry": "All", "AUC ROC": auc, "Best F1 1 Score": best_f1, "Best Precision 1": best_precision, "Best Recall 1": best_recall, "Best Accuracy": best_accuracy, "Number of Data Points": number_points, "Best Threshold": best_threshold, **all_hit_rates})

        # Calculate and save AUC ROC scores and best F1 scores to the second sheet
        auc, best_f1, best_threshold, number_points, best_recall, best_accuracy, best_precision = calculate_auc_and_best_f1(combined_mask, true_labels, predicted_probabilities_all, thresholds)
        combined_hit_rates = {f'Top {n} Hit Rate': hit_rates[f'Top {n} Hit Rate'][-2] for n in top_n_values}
        auc_roc_scores.append({"Industry": "Combined", "AUC ROC": auc, "Best F1 1 Score": best_f1, "Best Precision 1": best_precision, "Best Recall 1": best_recall, "Best Accuracy": best_accuracy, "Number of Data Points": number_points, "Best Threshold": best_threshold, **combined_hit_rates})

        for industry in set(industries):
            print("ok")
            print(industry)
            print("ok")
            industry_mask = [i == industry for i in industries]
            auc, best_f1, best_threshold, number_points, best_recall, best_accuracy, best_precision = calculate_auc_and_best_f1(industry_mask, true_labels, predicted_probabilities, thresholds)
            industry_hit_rates = {f'Top {n} Hit Rate': hit_rates[f'Top {n} Hit Rate'][list(set(industries)).index(industry)] for n in top_n_values}
            auc_roc_scores.append({"Industry": sic_mapping[industry], "AUC ROC": auc, "Best F1 1 Score": best_f1, "Best Precision 1": best_precision, "Best Recall 1": best_recall, "Best Accuracy": best_accuracy, "Number of Data Points": number_points, "Best Threshold": best_threshold, **industry_hit_rates})

        auc_roc_df = pd.DataFrame(auc_roc_scores)
        
        # Calculate weighted averages (excluding "Combined")
        total_points_excluding_combined = auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"].sum()
        weighted_avg = {
            "Industry": "Weighted Average",
            "AUC ROC": (auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "AUC ROC"] * auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"]).sum() / total_points_excluding_combined,
            "Best F1 1 Score": (auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Best F1 1 Score"] * auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"]).sum() / total_points_excluding_combined,
            "Best Precision 1": (auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Best Precision 1"] * auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"]).sum() / total_points_excluding_combined,
            "Best Recall 1": (auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Best Recall 1"] * auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"]).sum() / total_points_excluding_combined,
            "Best Accuracy": (auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Best Accuracy"] * auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"]).sum() / total_points_excluding_combined,
            "Number of Data Points": total_points_excluding_combined,
            "Best Threshold": (auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Best Threshold"] * auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"]).sum() / total_points_excluding_combined,
        }
        
        for n in top_n_values:
            weighted_avg[f'Top {n} Hit Rate'] = (auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), f'Top {n} Hit Rate'] * auc_roc_df.loc[(auc_roc_df["Industry"] != "Combined") & (auc_roc_df["Industry"] != "All"), "Number of Data Points"]).sum() / total_points_excluding_combined

        # Append weighted averages row to DataFrame
        auc_roc_df = pd.concat([auc_roc_df, pd.DataFrame([weighted_avg])], ignore_index=True)

        # Format columns to percentage with 2 decimal places
        percentage_columns = ["AUC ROC", "Best F1 1 Score", "Best Precision 1", "Best Recall 1", "Best Accuracy"] + [f'Top {n} Hit Rate' for n in top_n_values]
        auc_roc_df[percentage_columns] = auc_roc_df[percentage_columns].applymap(lambda x: f"{x:.2f}")


        auc_roc_df.to_excel(writer, sheet_name="AUC ROC", index=False, startrow=0, startcol=0)

        # Calculate and save combined results (without splitting by industry)
        for threshold in thresholds:
            conf_matrix, metrics = calculate_confusion_matrix_and_metrics(threshold, combined_mask, true_labels, predicted_probabilities)
            if conf_matrix.shape == (2, 2):   # If confusion matrix is not empty
                sheet_name = f"Combined {int(threshold*100)}%"
                conf_matrix_df = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
                conf_matrix_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)

                metrics_df = pd.DataFrame(metrics, index=["Class 0", "Class 1"])
                metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=conf_matrix_df.shape[0] + 3, startcol=0)

        # Iterate over each industry and save their confusion matrices and metrics
        for industry in set(industries):
            industry_mask = [i == industry for i in industries]
            for threshold in thresholds:
                conf_matrix, metrics = calculate_confusion_matrix_and_metrics(threshold, industry_mask, true_labels, predicted_probabilities)
                if conf_matrix.size != 0:  # If confusion matrix is not empty
                    sheet_name = f"{sic_mapping[industry]} {int(threshold*100)}%"
                    conf_matrix_df = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
                    conf_matrix_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)

                    metrics_df = pd.DataFrame(metrics, index=["Class 0", "Class 1"])
                    metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=conf_matrix_df.shape[0] + 3, startcol=0)


if __name__ == "__main__":
    main()

