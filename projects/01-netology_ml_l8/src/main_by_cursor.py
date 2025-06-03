#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Boston Housing Price Prediction
This script performs regression analysis on the Boston Housing dataset to predict house prices.
It includes data loading, preprocessing, model training, and evaluation.
"""

import sys
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load the Boston Housing dataset and return features and target.
    
    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the target Series
    """
    try:
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        
        # Combine the data from alternating rows
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        # Create DataFrame with proper column names
        df = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                                       'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
        df['MEDV'] = target
        
        logger.info("Data loaded successfully")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
        else:
            logger.info("No missing values found")
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data by splitting into features and target, then into train and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data split into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Train multiple regression models and return them.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"{name} trained successfully")
    
    return trained_models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate models using multiple metrics and create visualizations.
    
    Args:
        models (dict): Dictionary of trained models
        X_train, X_test (pd.DataFrame): Training and test features
        y_train, y_test (pd.Series): Training and test targets
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R2': train_r2,
            'Test R2': test_r2
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        results[name]['CV MSE'] = -cv_scores.mean()
        
        # Feature importance for Random Forest
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance (Random Forest)')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{name} - Predictions vs Actual')
        plt.tight_layout()
        plt.savefig(f'{name.lower().replace(" ", "_")}_predictions.png')
        plt.close()
    
    # Print results
    for name, metrics in results.items():
        logger.info(f"\n{name} Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(description='Boston Housing Price Prediction')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(
            df, test_size=args.test_size, random_state=args.random_state
        )
        
        # Train models
        models = train_models(X_train, y_train)
        
        # Evaluate models
        evaluate_models(models, X_train, X_test, y_train, y_test)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 