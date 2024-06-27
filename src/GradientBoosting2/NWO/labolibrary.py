import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit


# Define the custom metric function
def multinacional_metric(y_pred, dataset):
    y_true = dataset.get_label()
    metric_value = abs(sum(y_true - y_pred)) / sum(y_true)
    return 'multinacional_metric', metric_value, False

def train_lightgbm_model(data, lgboost_params={},col='tn_2',metric='multinacional_metric'):
    X = data.drop(columns=[col])
    y = data[col]
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = np.inf

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
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
        score = abs(sum(y_test - y_pred)) / sum(y_test)

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

