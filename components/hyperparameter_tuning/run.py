#!/usr/bin/env python
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import wandb
import hydra
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@hydra.main(config_path=".", config_name="config")
def go(config: DictConfig):
    # Get the Optuna trial number from Hydra and convert to int
    trial_num = int(os.environ.get('HYDRA_SWEEP_JOB', '0').split('_')[-1])
    
    # Initialize W&B run with incremental trial number
    run = wandb.init(
        job_type="hyperparameter_tuning",
        name=f"trial_{trial_num}",
        reinit=True,
        id=f"trial_{trial_num}"
    )

    # Log config to W&B - convert DictConfig to regular dict
    wandb.config.update({
        "n_estimators": config.modeling.random_forest.n_estimators,
        "max_depth": config.modeling.random_forest.max_depth,
        "min_samples_split": config.modeling.random_forest.min_samples_split,
        "min_samples_leaf": config.modeling.random_forest.min_samples_leaf,
        "max_features": config.modeling.random_forest.max_features
    })
    
    # Log trial information
    logger.info("=" * 80)
    logger.info(f"Trial #{trial_num}")
    logger.info("Parameters:")
    logger.info(f"n_estimators: {config.modeling.random_forest.n_estimators}")
    logger.info(f"max_depth: {config.modeling.random_forest.max_depth}")
    logger.info(f"min_samples_split: {config.modeling.random_forest.min_samples_split}")
    logger.info(f"min_samples_leaf: {config.modeling.random_forest.min_samples_leaf}")
    logger.info(f"max_features: {config.modeling.random_forest.max_features}")
    logger.info("=" * 80)

    # Get the data
    logger.info("Loading trainval data")
    trainval_local_path = run.use_artifact("trainval_data.csv:latest").file()
    df = pd.read_csv(trainval_local_path)
    
    # Handle missing values and data types
    df = df.dropna()
    
    # Separate features and target
    y = df.pop("price")
    X = df.copy()
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Set random seeds for reproducibility
    np.random.seed(config.modeling.random_seed)
    
    # Split the data with fixed random state
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=config.modeling.val_size,
        stratify=X[config.modeling.stratify_by],
        random_state=config.modeling.random_seed  # Ensure this is set
    )

    # Create full pipeline with fixed random state
    full_pipeline = get_inference_pipeline(
        config.modeling.random_forest, 
        config.modeling.max_tfidf_features,
        config.modeling.random_seed
    )

    # Fit and evaluate
    logger.info("Fitting pipeline")
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Log metrics to W&B
    wandb.log({
        "trial": trial_num,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "n_estimators": config.modeling.random_forest.n_estimators,
        "max_depth": config.modeling.random_forest.max_depth,
        "min_samples_split": config.modeling.random_forest.min_samples_split,
        "min_samples_leaf": config.modeling.random_forest.min_samples_leaf,
        "max_features": config.modeling.random_forest.max_features
    })


    
    # After calculating metrics...
    logger.info("Exporting model")
    
    # Get feature names after preprocessing
    processed_features = X_train.columns.tolist()
    
    # Create model signature
    signature = mlflow.models.infer_signature(
        X_val[processed_features],
        y_pred
    )

    # Export the model
    model_export_path = "random_forest_export"
    
    mlflow.sklearn.save_model(
        full_pipeline,
        model_export_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=X_val[processed_features].iloc[0:2]
    )

    # Log the model to W&B
    artifact = wandb.Artifact(
        name="random_forest_model",
        type="model_export",
        description="Random Forest pipeline with preprocessors",
    )
    artifact.add_dir(model_export_path)
    run.log_artifact(artifact)

    # Make sure to finish the run
    wandb.finish()
    
    # Return metric for Hydra optimization
    return mae

def get_inference_pipeline(rf_config, max_tfidf_features, random_seed):
    # Categorical features
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    
    ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder()
    )

    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"), 
        OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    )

    # Numerical features
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # Date features
    date_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="2010-01-01"),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    # Text features
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        FunctionTransformer(reshape_to_1d, check_inverse=False),
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",
    )

    # Create full pipeline
    rf = RandomForestRegressor(
        n_estimators=rf_config.n_estimators,
        max_depth=rf_config.max_depth,
        min_samples_split=rf_config.min_samples_split,
        min_samples_leaf=rf_config.min_samples_leaf,
        max_features=rf_config.max_features,
        random_state=random_seed,
        n_jobs=-1
    )

    return Pipeline([('preprocessor', preprocessor), ('regressor', rf)])

def delta_date_feature(dates):
    """
    Calculate the difference in days between each date and the oldest date in the dataset
    """
    date_min = datetime.strptime("2010-01-01", "%Y-%m-%d")
    # Flatten 2D array if necessary and handle NaN values
    dates = dates.ravel() if hasattr(dates, 'ravel') else dates
    dates = pd.Series(dates).fillna("2010-01-01")
    dates = pd.to_datetime(dates)
    days = np.array([(date - date_min).days for date in dates])
    return days.reshape(-1, 1)  # Return 2D array

def reshape_to_1d(data):
    """Reshape data to 1D."""
    return data.reshape(-1)

if __name__ == "__main__":
    go() 