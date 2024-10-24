import logging
from pathlib import Path
from datetime import datetime
import pickle
import sys
import subprocess
import pkg_resources
import time
import joblib

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
import logging
import pandas as pd
import numpy as np
import shap
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, make_scorer
import xgboost as xgb
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import make_scorer, precision_score
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths and file names
data_dir = Path(r"C:\Users\16036\OneDrive\Desktop\Investor Sight\Vulnerability\Model")
stata_dir = data_dir
output_dir = data_dir
data_file = "final_dataset3.dta"

# Constants
TARGET_VAR = "F1_factset_activism"
INDUSTRY_VAR = "sic_1_digit"
CLUSTER_ON = ["ln_mkv"]
CUTOFF_YEAR = 2014
MODEL_LABEL = "xgb"
NEIGHBORS_MAPPING = {
    9: 1, # Public Administration
    8: 1, # Services
    2: 1, # Construction
    4: 1, # Transportation
    6: 1, # Retail Trade
    7: 1, # Finance
    3: 1, # Manufacturing
    1: 1, # Mining
    5: 1, # Wholesale Trade
    0: 1 # Agriculture
}

NEIGHBORS_ALL = 1


df6 = pd.read_stata(stata_dir / data_file)
DATA_FIELDS = df6.columns.tolist()
print(DATA_FIELDS)


columns_to_drop = ["yq", "index", "t_wc", "liwc_WC", "year1", "companyid", "F1_num_diligent_HF", "F1_num_factset_diligent_HF", "F1_num_shareholder_proposal", "comname", "isin", "ticker", "cusip", "cusip6", "cusip8", "cik", "state", "conm", "exchg", "loc", "sic_2_digit", "sic_3_digit", "ff17", "ff18", "ff48", "F1_partial_HF_own", "F1_primary_HF_own", "F1_primary_partial_HF_own", "F1_HF", "F1_factset_HFactivism", "F1_num_factset_activism", "F1_num_factset_HFactivism", "F1_factset_diligent_HF", "F1_diligent_HF", "F1_HF_own"]

DATA_FIELDS = [col for col in DATA_FIELDS if col not in columns_to_drop]
DATA_FIELDS = list(dict.fromkeys(DATA_FIELDS))
ID_FIELDS = DATA_FIELDS[:4]

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
}

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
    100: "ALL"
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
        #data['hf_activism_past_3_years'] = data.apply(lambda row: self.count_hf_activism_past_years(data, row['gvkey'], row['year']), axis=1)

        # Add activism_past_3_years to DATA_FIELDS after it has been created
        #if 'activism_past_3_years' not in self.data_fields:
            #self.data_fields.append('activism_past_3_years')
        #if 'hf_activism_past_3_years' not in self.data_fields:
        #    self.data_fields.append('hf_activism_past_3_years')

        return data

    def load_data(self):
        logging.info(f"Reading {self.data_file} into a DataFrame")
        df = pd.read_stata(self.data_file)
        logging.info(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        df.to_csv(output_dir / "raw_data929prop.csv", index=False)
        df = df[self.data_fields]
        logging.info(f"Data filtered for {len(self.data_fields)} fields")

        logging.info(f"Data filtered for year >= {self.cutoff_year}")
        df = df[df["year"] >= self.cutoff_year]

        logging.info("Preparation and filtering complete, proceeding without removing missing values.")
        df = self.prepare_data(df)
        return df
    
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
        return subset['F1_factset_activism'].sum()

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
    
def custom_time_series_split(X, y, split_years):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for train_years, val_years in split_years:
        train_idx = X.index[X['year'].isin(train_years)].to_numpy()
        val_idx = X.index[X['year'].isin(val_years)].to_numpy()
        
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Ensure validation fold has both positive and negative samples
        if len(np.unique(y_val_fold)) == 2:
            yield train_idx, val_idx

def perform_rfecv_and_save_results(train_features, train_labels, industry, neighbors):
    global results_df
    try:
        logging.info("Initializing the XGBClassifier")
        xgb_classifier = xgb.XGBClassifier(**param_grid)

        logging.info("Starting RFECV setup")
        rfecv = setup_rfecv(xgb_classifier, train_features, train_labels)  # Pass the train_features and train_labels here
        
        logging.info("Fitting RFECV")
        rfecv.fit(train_features, train_labels)
        logging.info("RFECV fitting process completed")

        selected_features = train_features.columns[rfecv.support_].tolist()
        
        # Calculate max number of features based on the number of rows in the training set
        max_features = max(len(train_labels) // 50, 1)  # Ensure at least 10 features are allowed
        logging.info(f"Max features allowed: {max_features}")
            
        if len(selected_features) == 0:
            logging.warning("RFECV selected zero features. Skipping this configuration.")
            return [], 0

        # If the number of selected features exceeds the limit, reduce them based on SHAP values
        if len(selected_features) > max_features:
            logging.info(f"Selected features exceed max limit. Reducing from {len(selected_features)} to {max_features}.")
            xgb_classifier.fit(train_features[selected_features], train_labels)

            shap_values = compute_and_save_shap_values(xgb_classifier, train_features[selected_features], train_labels, selected_features, industry, neighbors)

            # Calculate mean absolute SHAP values and rank features by importance
            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_feature_importance = pd.Series(shap_importance, index=selected_features)
            shap_feature_importance = shap_feature_importance.sort_values(ascending=False)

            # Keep only the top features based on the max_features limit
            selected_features = shap_feature_importance.head(max_features).index.tolist()

            xgb_classifier.fit(train_features[selected_features], train_labels)

        logging.info("Saving RFECV results")
        save_rfecv_results(selected_features, industry, neighbors)

        logging.info("Computing SHAP values")
        shap_values = compute_and_save_shap_values(xgb_classifier, train_features[selected_features], train_labels, selected_features, industry, neighbors)

        # Train the model with the final set of selected features
        pipeline = Pipeline([
            (MODEL_LABEL, xgb.XGBClassifier(**param_grid))
        ])
        model_filename, feature_names_filename = train_and_save_model(train_features[selected_features], train_labels, pipeline, industry, neighbors)

        predictions = xgb_classifier.predict(train_features[selected_features])        

        # Calculate metrics for all classes if needed and specifically for class 1 (minority class)
        accuracy = accuracy_score(train_labels, predictions)

        # Calculate metrics specifically for the minority class (assuming it is labeled as '1')
        precision_1 = precision_score(train_labels, predictions, labels=[1], average='binary')
        recall_1 = recall_score(train_labels, predictions, labels=[1], average='binary')
        f1_1 = f1_score(train_labels, predictions, labels=[1], average='binary')
        
        # Count of total and minority samples in training
        total_train = len(train_labels)
        minority = (train_labels == 1).sum()

        # Append new row to the DataFrame
        summary_data = {
            'Industry': industry,
            'Neighbors': neighbors,
            'Selected Features': len(selected_features),
            'Total Train': total_train,
            'Minority': minority,
            'F1 Score 1': f1_1,
            'Precision 1': precision_1,
            'Recall 1': recall_1,
            'Accuracy': accuracy
        }
        results_df = pd.concat([results_df, pd.DataFrame([summary_data])], ignore_index=True)
        return selected_features, f1_1, model_filename, feature_names_filename

    except Exception as e:
        logging.error(f"An error occurred during RFECV processing: {e}")
        return [], 0, None, None  

def setup_rfecv(estimator, train_features, train_labels):
    f1_scorer = make_scorer(f1_score, pos_label=1, average='binary')

    # Define the custom split years
    split_years = [
        (range(2014, 2017), [2018]),
        (range(2014, 2018), [2019]),
        (range(2014, 2019), [2020]),
        (range(2014, 2020), [2021]),
    ]

    # Create the custom time series split object
    custom_split = list(custom_time_series_split(X=train_features, y=train_labels, split_years=split_years))

    return RFECV(
        estimator=estimator,
        step=1,
        cv=custom_split,  # Use the custom split here
        scoring=f1_scorer,
        n_jobs=-1,
        min_features_to_select=10
    )

output_dir.mkdir(parents=True, exist_ok=True)

def save_rfecv_results(selected_features, industry, neighbors):
    try:
        results_filename = output_dir / f"rfecv_results_sic_{int(industry)}_neighbors_{neighbors}.txt"
        with open(results_filename, 'w') as f:
            f.write("Selected features:\n")
            for feature in selected_features:
                f.write(f"{feature}\n")
        logging.info(f"RFECV results saved for {industry} industry with {neighbors} neighbors")

        results_table_filename = output_dir / f"all_rfecv_results_sic_{int(industry)}_neighbors_{neighbors}929prop.csv"
        results_df = pd.DataFrame({"Selected Features": selected_features})
        results_df.to_csv(results_table_filename, index=False)
        logging.info(f"RFECV results table saved for {industry} industry with {neighbors} neighbors")

    except Exception as e:
        logging.error(f"Error saving RFECV results: {e}")

def compute_and_save_shap_values(estimator, features, labels, selected_features, industry, neighbors):
    try:
        logging.info("Training the model with selected features")
        estimator.fit(features[selected_features], labels)

        logging.info("Initializing SHAP explainer")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(features[selected_features])

        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_raw_values = shap_values.mean(axis=0)

        shap_df = pd.DataFrame({
            'Feature': selected_features,
            'Mean Absolute SHAP Value': shap_importance,
            'Raw SHAP Values': shap_raw_values
        })

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the SHAP values to a CSV file
        shap_results_filename = output_dir / f"all_shap_values_sic_{int(industry)}_neighbors_{neighbors}929prop.csv"
        shap_df.to_csv(shap_results_filename, index=False)

        logging.info(f"SHAP values saved for {industry} industry with {neighbors} neighbors")

        return shap_values

    except Exception as e:
        logging.error(f"Error computing or saving SHAP values: {e}")
        return None


def train_and_save_model(train_features, train_labels, pipeline, industry, neighbors):

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

    return model_filename, feature_names_filename

def predict_confidence_scores(test_data, sic_mapping):
    results = []

    for idx, row in test_data.iterrows():
        industry = row[INDUSTRY_VAR]
        neighbors = NEIGHBORS_MAPPING.get(industry, 1)
        model_filename = output_dir / f"model2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"

        if not model_filename.exists():
            logging.warning(f"No model found for {industry} industry with {neighbors} neighbors")
            continue

        # Load the model
        model = joblib.load(model_filename)

        # Prepare the data for prediction
        row_features = row.drop([TARGET_VAR, INDUSTRY_VAR]).values.reshape(1, -1)

        # Predict the confidence score
        confidence_score = model.predict_proba(row_features)[0, 1]  # Assuming binary classification
        results.append((row['gvkey'], industry, confidence_score))

    return results

def train_and_save_model_with_weights(train_features, train_labels, pipeline, industry, neighbors, weight_feature, weight_factor):
    # Generate sample weights based on the value of the important feature
    sample_weights = train_features[weight_feature].apply(lambda x: x * weight_factor)
    
    # Train the model with sample weights
    pipeline.named_steps[MODEL_LABEL].fit(train_features, train_labels, sample_weight=sample_weights)
    
    # Save the model
    model_filename = output_dir / f"model2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
    joblib.dump(pipeline, model_filename)
    logging.info(f"Model saved for {industry} industry with {neighbors} neighbors")

    # Save the feature names
    feature_names_filename = output_dir / f"feature_names2_sic_{int(industry)}_neighbors_{neighbors}prop.pkl"
    with open(feature_names_filename, 'wb') as f:
        pickle.dump(train_features.columns.tolist(), f)
    logging.info(f"Feature names saved for {industry} industry with {neighbors} neighbors")

hit_rates = pd.DataFrame(columns=['Industry', 'Neighbors', 'Hit Rate 10', 'Hit Rate 20', 'Hit Rate 30', 'Hit Rate 50', 'Hit Rate 100', 'Points'])

def update_hit_rates(industry, neighbors, new_data, points):
    global hit_rates  # Reference the global DataFrame

    # Append 'Industry' and 'Neighbors' to the new_data dictionary
    new_data.update({'Industry': industry, 'Neighbors': neighbors, 'Points': points})

    # Create a new DataFrame row from new_data
    new_row = pd.DataFrame([new_data])  # Make sure it's a list of dictionary for a single row
    
    # Check if the row already exists
    existing_rows = (hit_rates['Industry'] == industry) & (hit_rates['Neighbors'] == neighbors)
    if existing_rows.any():
        # Update existing row
        for key, value in new_data.items():
            hit_rates.loc[existing_rows, key] = value
    else:
        # Append the new row to the DataFrame
        hit_rates = pd.concat([hit_rates, new_row], ignore_index=True)


results_df = pd.DataFrame(columns=['Industry', 'Neighbors', 'Selected Features', 'Total Train', 'Minority', 'F1 Score 1', 'Precision 1', 'Recall 1', 'Accuracy'])

def main():
    stata_file_path = stata_dir / data_file
    logging.info(f"Loading data from {stata_file_path}")
    neighbors = 1
    df = DataProcessor(stata_file_path, DATA_FIELDS, ID_FIELDS, cutoff_year=CUTOFF_YEAR).load_data()
    df = df[df['russell3000'] == 1]
    df = df[df['year'] != 2023.0]
    test_data = df[df['year'] == 2022.0]

    # Dictionary to hold selected features and best neighbors for each industry
    industry_selected_features = {}
    industry_best_neighbors = {}

    # Perform initial analysis on all industries combined
    logging.info("Running model on all industries combined")
    train_data_all = df[df['year'] != 2022.0]

    best_hit_rate_top_30 = 0
    best_neighbors_all = NEIGHBORS_ALL
    best_selected_features_all = []

    matched_train_data = DataMatcher(target_var=TARGET_VAR, id_fields=ID_FIELDS, cluster_on=CLUSTER_ON, neighbors=NEIGHBORS_ALL).fit_transform(train_data_all)

    train_labels = matched_train_data[TARGET_VAR]
    train_features = matched_train_data.drop([TARGET_VAR, INDUSTRY_VAR], axis=1)
    train_features['year'] = matched_train_data['year']
    # Check for non-numeric columns in the training features
    non_numeric_columns = train_features.select_dtypes(exclude=[np.number]).columns

    # If there are any non-numeric columns, print them
    if len(non_numeric_columns) > 0:
        print("Non-numeric columns found:")
        print(non_numeric_columns)
    else:
        print("All columns are numeric.")
    selected_features, f1_1, model_filename, feature_names_filename = perform_rfecv_and_save_results(train_features, train_labels, 100, neighbors)

    #test the model on test data
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(feature_names_filename, 'rb') as f:
        feature_names = pickle.load(f)

    test_features = test_data.reindex(columns=feature_names)
    test_data['confidence_score'] = model.predict_proba(test_features)[:, 1]
    test_data_sorted = test_data.sort_values(by='confidence_score', ascending=False)

    hit_rate_values = {f'Hit Rate {n}': (test_data_sorted.head(n)[TARGET_VAR] == 1).mean() for n in [10, 20, 30, 50, 100]}
    points = test_data_sorted.shape[0]
    update_hit_rates(sic_mapping[100], neighbors, hit_rate_values, points)

    current_hit_rate = (test_data_sorted.head(30)[TARGET_VAR] == 1).mean()
    if current_hit_rate > best_hit_rate_top_30:
        best_hit_rate_top_30 = current_hit_rate            
        best_neighbors_all = neighbors
        best_selected_features_all = selected_features
        
    logging.info(f"Best neighbors for all industries combined: {best_neighbors_all} with best hit rate for top 30: {best_hit_rate_top_30}")
    industry_selected_features[sic_mapping[100]] = best_selected_features_all
    industry_best_neighbors[sic_mapping[100]] = best_neighbors_all

    # Process each industry within the training data
    train_data = df[df['year'] != 2022.0]
    industries = train_data[INDUSTRY_VAR].unique()

    for industry in sorted(industries):
        logging.info(f"Running model on {sic_mapping[industry]} industry")

        df_industry = train_data[train_data[INDUSTRY_VAR] == industry]
        best_hit_rate_top_30 = 0
        best_neighbors = NEIGHBORS_MAPPING.get(industry, 1)  # Start with default or predefined neighbors
        best_selected_features = []

        matched_train_data = DataMatcher(target_var=TARGET_VAR, id_fields=ID_FIELDS, cluster_on=CLUSTER_ON, neighbors=neighbors).fit_transform(df_industry)

        if matched_train_data.empty:
            logging.warning(f"No matched data for {sic_mapping[industry]} industry with {neighbors} neighbors")
            continue

        train_labels = matched_train_data[TARGET_VAR]
        train_features = matched_train_data.drop([TARGET_VAR, INDUSTRY_VAR], axis=1)
        train_features['year'] = matched_train_data['year']

        selected_features, f1_1, model_filename, feature_names_filename = perform_rfecv_and_save_results(train_features, train_labels, industry, neighbors)

        if not selected_features or model_filename is None:
            logging.info(f"Skipping model testing for {industry} with {neighbors} neighbors due to RFECV failure")
            continue  # Skip further processing for this neighbor

        #test the model on test data
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(feature_names_filename, 'rb') as f:
            feature_names = pickle.load(f)
        test_data_industry = test_data[test_data[INDUSTRY_VAR] == industry]
        test_features = test_data_industry.reindex(columns=feature_names)
        test_data_industry['confidence_score'] = model.predict_proba(test_features)[:, 1]
        test_data_sorted = test_data_industry.sort_values(by='confidence_score', ascending=False)


        hit_rate_values = {f'Hit Rate {n}': (test_data_sorted.head(n)[TARGET_VAR] == 1).mean() for n in [10, 20, 30, 50, 100]}
        points = test_data_sorted.shape[0]
        update_hit_rates(sic_mapping[industry], neighbors, hit_rate_values, points)

        current_hit_rate = (test_data_sorted.head(30)[TARGET_VAR] == 1).mean()
        if current_hit_rate > best_hit_rate_top_30:
            best_hit_rate_top_30 = current_hit_rate
            best_neighbors = neighbors
            best_selected_features = selected_features

        logging.info(f"Best neighbors for {sic_mapping[industry]} industry: {best_neighbors} with best hit rate for top 30: {best_hit_rate_top_30}")
        industry_selected_features[sic_mapping[industry]] = best_selected_features
        industry_best_neighbors[sic_mapping[industry]] = best_neighbors

    results_df.to_excel(output_dir / "rfecv_summary_results929prop.xlsx", index=False)
    logging.info("All RFECV results saved to Excel.")

    # Create a summary DataFrame for selected features
    max_length = max(len(features) for features in industry_selected_features.values())
    summary_data = {industry: features + [''] * (max_length - len(features)) for industry, features in industry_selected_features.items()}
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(output_dir / "all_selected_features_summary929prop.xlsx", index=False)

    # Create and save a summary DataFrame for best neighbors
    neighbors_df = pd.DataFrame.from_dict(industry_best_neighbors, orient='index', columns=['Best Neighbors'])
    neighbors_df.to_excel(output_dir / "best_neighbors_summary929prop.xlsx", index_label='Industry')
    hit_rates.to_csv(output_dir / 'hit_rate_summary929prop.csv', index=False)

if __name__ == "__main__":
    main()


