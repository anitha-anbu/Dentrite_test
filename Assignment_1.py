from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import re
import json
from striprtf.striprtf import rtf_to_text
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to parse JSON
def parse_json(file_path):
    with open(file_path, 'r') as file:
        rtf_content = file.read()

    # Convert RTF to plain text
    plain_text = rtf_to_text(rtf_content)

    # Extract JSON string using regex
    match = re.search(r'\{.*\}', plain_text, re.DOTALL)

    if match is not None:
        json_string = match.group(0)
        # Parse JSON
        data = json.loads(json_string)
        return data
    else:
        print("No JSON content found in the RTF file.")

# Load JSON data
json_data = parse_json('algoparams_from_ui.json.rtf')

# Extract relevant parts from JSON
target_column = json_data['design_state_data']['target'].get('target')
prediction_type = json_data['design_state_data']['target'].get('prediction_type')
feature_handling = json_data['design_state_data'].get('feature_handling')
feature_reduction = json_data['design_state_data']['feature_reduction'].get('feature_reduction_method')
hyper_params = json_data['design_state_data'].get('hyperparameters')
algorithms = json_data['design_state_data'].get('algorithms')

# Load dataset
data = pd.read_csv('iris.csv')

# Printing the dataset in its original form
print(data.head())

# Handle missing values
for feature in feature_handling:
    if feature_handling[feature].get('is_selected') and feature_handling[feature].get('impute_with') == "Average of values":
        imputer = SimpleImputer(strategy="mean")
        data[feature_handling[feature].get('feature_name')] = imputer.fit_transform(data[feature_handling[feature].get('feature_name')])
        

# Printing the dataset with imputed missing values
print(data.head())     

# Define feature reduction methods
def get_feature_reduction_method(method):
    if method == 'No Reduction':
        return None
    elif method == 'Corr with Target':
        return SelectKBest(score_func=f_regression)
    elif method == 'Tree-based':
        return SelectKBest(score_func=mutual_info_regression)
    elif method == 'PCA':
        return PCA(n_components=0.95)  # Keep 95% of variance
    else:
        raise ValueError(f"Unknown feature reduction method: {method}")

# Define models based on prediction type
def get_model():
    models = {
            'LinearRegression': LinearRegression(),
            'RidgeRegression': Ridge(),
            'LassoRegression': Lasso(),
            'RandomForestRegressor': RandomForestRegressor()
        }
    return models

# Set up pipeline
pipeline_steps = []

# Feature reduction step
feature_reduction_method = get_feature_reduction_method(feature_reduction)
if feature_reduction_method:
    pipeline_steps.append(('feature_reduction', SelectKBest(k=3)))

# Model building step
models = get_model()
for model_name, model in models.items():
    if algorithms[model_name]['is_selected']:
        print(model_name)
        param_grid = {
                    "model__n_estimators": [10, 20, 30],
                    "model__max_features": ["auto", "sqrt", "log2"],
                    "model__min_samples_split": [2, 4, 8],
                    "model__bootstrap": [True, False],
                    }
        pipeline_steps.append(('model', model))
        pipeline = Pipeline(pipeline_steps)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        
        X = data.drop(columns=[target_column, "species"])
        y = data[target_column]
        
        # Fit the model
        grid_search.fit(X, y)
        
        # Predict and evaluate
        y_pred = grid_search.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Best Params: {grid_search.best_params_}")
        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"R^2 Score: {r2}")



