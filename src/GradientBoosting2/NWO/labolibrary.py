import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler,PowerTransformer
import math

#DATOS_DIR = '~/buckets/b1/datasets/'
DATOS_DIR = '../data/'

# Function to center, scale, and return a series
def scale_group(group):
    scaler = RobustScaler(),
    #scaler = PowerTransformer()
    scaled_values = scaler.fit_transform(group.values.reshape(-1, 1)).flatten()
    scalers[group.name] = scaler  # Store the scaler for this group
    return pd.Series(scaled_values, index=group.index, name=group.name)

# Function to inverse transform (de-scale) and decenter, and return a series
def inverse_scale_group(group):
    group_name = group.name
    scaler = scalers[group_name]
    inversed_centered_values = scaler.inverse_transform(group.values.reshape(-1, 1)).flatten()
    original_values = inversed_centered_values
    return pd.Series(original_values, index=group.index, name=group_name)

# Custom metric function
def multinacional_metric(y_pred,y_true):
    y_true = y_true.get_label()
    metric = abs(sum(y_true - y_pred)) / sum(y_true)
    return 'multinacional_metric', metric, False


def train_lightgbm_model(data, lgboost_params={},col='tn_2',metric='multinacional_metric',weights=""):
    X = data.drop(columns=[col])
    y = data[col]
    tscv = TimeSeriesSplit(5)
    best_model = None
    best_score = np.inf

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        weights_train = weights.iloc[train_index]
        #train_data = lgb.Dataset(X_train, label=y_train,weight=weights_train)
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data =  lgb.Dataset(X_test, label=y_test, reference=train_data)
        params  = lgboost_params
        
        evals_result = {}
        if metric == 'multinacional_metric':
            model = lgb.train(
                params, 
                train_data, 
                num_boost_round=1000, 
                valid_sets=[test_data], 
                valid_names=['validation'], 
                feval=multinacional_metric,
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
        else:
             model = lgb.train(
                params, 
                train_data, 
                num_boost_round=1000, 
                valid_sets=[test_data], 
                valid_names=['validation'],
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
        
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        #score = abs(sum(y_test - y_pred)) / sum(y_test)
        #score = sum(y_test**2 - y_pred**2) / sum(y_test)
        score = mean_squared_error(y_test, y_pred) 
        if score < best_score:
            best_score = score
            best_model = model

    return best_model, best_score

def predict_next_month(model, last_data_points,col='tn_2'):
    predictions = []
    last_month = last_data_points.index.max() + 1

    last_data_points.index = [last_month] * len(last_data_points)  # Set index to the next month
    predictions = model.predict(last_data_points, num_iteration=model.best_iteration)
    
    prediction_df = last_data_points[['product_id']].copy()
    prediction_df[col] = predictions
    prediction_df.index = [last_month] * len(last_data_points)

    return prediction_df

def train_lightgbm_model_classic(data, lgboost_params={}, col='tn_2', metric='multinacional_metric', weights=""):
    # Ensure weights is a numpy array
    weights = weights.to_numpy()

    # Sort data by date (assuming the index is the date)
    data = data.sort_index()

    X = data.drop(columns=[col])
    y = data[col]

    best_model = None
    best_score = np.inf

    # Define train-validation split date (using the last year-month in the data as the end of the training period)
    split_date = data.index[-1]
    
    # Split the data into training and testing sets based on the split date
    X_train = X.loc[:split_date]
    y_train = y.loc[:split_date]
    weights_train = weights[:len(X_train)]

    X_test = X.loc[split_date:]
    y_test = y.loc[split_date:]
    weights_test = weights[len(X_train):]

    # Create LightGBM datasets
    #train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    train_data = lgb.Dataset(X_train, label=y_train)
   
    test_data = lgb.Dataset(X_test, label=y_test, weight=weights_test, reference=train_data)
    
    params = lgboost_params
    
    evals_result = {}
    if metric == 'multinacional_metric':
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            valid_names=['validation'],
            feval=multinacional_metric,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
    else:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            valid_names=['validation'],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    score = mean_squared_error(y_test, y_pred)
    if score < best_score:
        best_score = score
        best_model = model

    return best_model, best_score