from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn import metrics
import re

import argparse
import logging
import os
import sys
import warnings
from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, RunInfo
import mlflow.sklearn



if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('lpp.mltraining')


def get_or_create_experiment(experiment_name) -> Experiment:
    """
    Creates an mlflow experiment
    :param experiment_name: str. The name of the experiment to be set in MLFlow
    :return: the experiment created if it doesn't exist, experiment if it is already created.
    """
    try:
        client = MlflowClient()
        experiment: Experiment = client.get_experiment_by_name(name=experiment_name)
        if experiment and experiment.lifecycle_stage != 'deleted':
            return experiment
        else:
            experiment_id = client.create_experiment(name=experiment_name)
            return client.get_experiment(experiment_id=experiment_id)
    except Exception as e:
        logger.error(f'Unable to get or create experiment {experiment_name}: {e}')






def mse_score(y_pred, y_true):
    """
    Computes the Mean Squared Error (MSE) between model predictions and known labels
    :param y_pred: the predictions output by the model
    :param y_true: the ground truth (aka known labels)
    :return: the Mean Squared Error
    """
    return metrics.mean_squared_error(y_true, y_pred)

def mse_score_model(model, X_test, y_test):
    """
    Computes the MSE score of a model on a test dataset
    :param model: a ML model with a 'predict' method
    :param X_test: the test features
    :param y_test: the test labels
    :return: the MSE between model predictions and ground truth
    """
    return mse_score(model.predict(X_test), y_test)




def train(data_path, test_size):
    
    # Load dataset
    data = pd.read_csv(data_path)
    logger.info(f'Data shape: {data.shape}')


    # Drop unnecessary columns
    data.drop(['Laptop Name', 'Old Price', 'Link', 'Seller Name', 'Current Prices',
           'Door Delivery Fees', 'Door Delivery Date', 'Door Delivery Ready Time', 
           'Pickup Fees', 'Pickup Date', 'Pickup Ready Time'], axis=1, inplace=True)

    # Create Weighted_Rating column and drop original rating columns
    data['Weighted_Rating'] = (data['Rating_out_of_5'] * data['Number of Ratings']) / data['Number of Ratings'].sum()
    data.drop(['Rating_out_of_5', 'Number of Ratings'], axis=1, inplace=True)

    # Fill missing values in Seller Score with median
    data['Seller Score'].fillna(data['Seller Score'].median(), inplace=True)

    # Extract Storage and isSSD columns using regex
    data['Storage'] = data['ROM'].str.extract(r'(\d+)').astype(float)
    data['isSSD'] = data['ROM'].str.contains('SSD', case=False, na=False).astype(int)
    data.drop('ROM', axis=1, inplace=True)

    # One-Hot Encode Sale Type and Brand
    categorical_features = ['Sale Type', 'Brand']
    encoded_features = pd.get_dummies(data[categorical_features], drop_first=True)
    data = pd.concat([data.drop(categorical_features, axis=1), encoded_features], axis=1)

    # Clean RAM column to retain numeric values
    data['RAM'] = data['RAM'].str.extract(r'(\d+)').astype(float)


    # Define preprocessing steps
    numerical_features = [
    'Discount Percentages', 'Rating_out_of_5', 'Number of Ratings',
    'Weighted_Rating', 'Unit Available', 'Seller Score'
]
    
    # Fill missing values for numerical columns
    numerical_features = ['RAM', 'Storage']
    data[numerical_features] = data[numerical_features].fillna(data[numerical_features].median())

    # Define numerical features for scaling
    numerical_features = [
        'Weighted_Rating', 'Seller Score', 'RAM', 'Storage', 
        'Discount Percentages', 'Unit Available', 'isSSD'
    ]

    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    # Combine preprocessors in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ]
    )
    

    # Split data into training and testing sets
    X = data.drop('USD Price', axis=1)  
    y = data['USD Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    logger.info(f"""
        X_train size: {X_train.shape}
        X_test size: {X_test.shape} """)


    # Define the model
    rf_model = RandomForestRegressor(random_state=42)

    # Define hyperparameter grid for Random Forest
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

    # Wrap the RandomForestRegressor with GridSearchCV
    rf_cv = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1
)

    rf_cv.fit(X_train, y_train)

    # Best model and score
    logger.info(f"Best Cross-Validation Score: {rf_cv.best_score_}")
    logger.info(f"Best Parameters:{rf_cv.best_params_}")
 


    # Create the pipeline
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('random_forest_cv', rf_cv)
])
    
    # Train the model
    logger.info('Starting model training...')
    pipeline.fit(X_train, y_train)
    logger.info('Training done, starting model evaluation...')

    test_score = mse_score_model(pipeline, X_test, y_test)
    logger.info(f"Test Set R2 Score with hyperparameter search and cross-validation: {test_score}")

    return pipeline, "test_score", test_score


def log_metrics_and_model(pipeline, test_metric_name: str, test_metric_value: float):

    experiment_name = 'laptop_prices_estimator_training'
    experiment: Experiment = get_or_create_experiment(experiment_name)

    model = pipeline.steps[1][1]
    best_params = model.best_params_

    def build_local_model_path(relative_artifacts_uri: str, model_name: str):
        import os
        absolute_artifacts_uri = relative_artifacts_uri.replace('./', f'{os.getcwd()}/')
        return os.path.join(absolute_artifacts_uri, model_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='training') as run:
        model_name = 'sklearn_random_forest'
        run_info: RunInfo = run.info
        mlflow.log_params(best_params)
        mlflow.log_metric(test_metric_name, test_metric_value)
        mlflow.sklearn.log_model(pipeline, model_name)
        model_path = build_local_model_path(run_info.artifact_uri, model_name)

        logger.info(f'Model for run_id {run_info.run_id} exported at {model_path}')




if __name__ == '__main__':
    """
    Lauches the training script
    :param data_path: the path to the full dataset
    :param test_size: the ratio of test data vs training data
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path')
    parser.add_argument('--test-size')
    args = parser.parse_args()

    if not args.data_path:
        raise Exception('argument --data-path should be specified')

    if not args.test_size:
        raise Exception('argument --test-size should be specified')

    pipeline, test_metric_name, test_metric_value = train(args.data_path, float(args.test_size))
    if os.getenv('MLFLOW_TRACKING_URI', None):
        log_metrics_and_model(pipeline=pipeline, test_metric_name=test_metric_name, test_metric_value=test_metric_value)
    else:
        logger.info(f'Env var MLFLOW_TRACKING_URI not set, skipping mlflow logging.')


