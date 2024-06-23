import numpy as np
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
import logging
import pandas as pd

# Disable LightGBM warnings
logging.getLogger("lightgbm").setLevel(logging.ERROR)

"""
This class performs feature bagging and sample bagging with hyperparameter tuning using Bayesian Optimization
for time series data. The goal is to train multiple models on different subsets of features and samples,
and aggregate their predictions.

Parameters:
- X: pd.DataFrame, the feature set
- y: pd.Series, the target variable
- n_models: int, number of models to train
- feature_fraction: float, fraction of features to use for each model
- sample_fraction: float, fraction of samples to use for each model
- param_bounds: dict, parameter bounds for Bayesian optimization
- init_points: int, number of initial points for Bayesian optimization
- n_iter: int, number of iterations for Bayesian optimization
- random_state: int, random seed for reproducibility

Methods:
- lgb_cv: Cross-validation function for LightGBM with the given hyperparameters using TimeSeriesSplit.
- fit: Fit the bagging model with hyperparameter tuning.
- predict: Predict using the bagging model.
- fit_multiple_seeds: Fit the bagging model with multiple random seeds and aggregate predictions.
"""


class FeatureBaggingWithHyperparamTuning:
    def __init__(self, X, y, n_models, feature_fraction, sample_fraction, param_bounds, init_points=5, n_iter=25, random_state=0,optimization_target=None)    :
        self.X = X
        self.y = y
        self.n_models = n_models
        self.feature_fraction = feature_fraction
        self.sample_fraction = sample_fraction
        self.param_bounds = param_bounds
        self.init_points = init_points
        self.n_iter = n_iter
        self.models = []
        self.feature_subsets = []
        self.best_params = None
        self.random_state = random_state
        self.optimization_target = optimization_target or 'mse'  
        np.random.seed(random_state)
              
    def lgb_cv(self, num_leaves, learning_rate, n_estimators, min_child_samples, subsample, colsample_bytree, max_depth):
        scores = []
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            for feature_subset in self.feature_subsets:
                model = lgb.LGBMRegressor(
                    num_leaves=int(num_leaves),
                    learning_rate=learning_rate,
                    n_estimators=int(n_estimators),
                    min_child_samples=int(min_child_samples),
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    max_depth=int(max_depth),
                    random_state=self.random_state,
                    verbose=-1,                
                )  
                model.fit(X_train[feature_subset], y_train, eval_metric=self.optimization_target)
                score = model.score(X_test[feature_subset], y_test)
                scores.append(score)
        return np.mean(scores)

    def fit(self):
        n_features = self.X.shape[1]
        used_features = set()
        while len(used_features) < n_features:
            feature_subset = np.random.choice(self.X.columns, size=int(self.feature_fraction * n_features), replace=False)
            self.feature_subsets.append(feature_subset)
            used_features.update(feature_subset)

        for _ in range(self.n_models - len(self.feature_subsets)):
            feature_subset = np.random.choice(self.X.columns, size=int(self.feature_fraction * n_features), replace=False)
            self.feature_subsets.append(feature_subset)

        optimizer = BayesianOptimization(
            f=self.lgb_cv,
            pbounds=self.param_bounds,
            random_state=self.random_state
        )
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        self.best_params = optimizer.max['params']
        self.best_params['num_leaves'] = int(self.best_params['num_leaves'])
        self.best_params['n_estimators'] = int(self.best_params['n_estimators'])
        self.best_params['min_child_samples'] = int(self.best_params['min_child_samples'])
        self.best_params['max_depth'] = int(self.best_params['max_depth'])

        for _ in range(self.n_models):
            sample_indices = np.random.choice(self.X.index, size=int(len(self.X) * self.sample_fraction), replace=False)
            sample_X = self.X.loc[sample_indices]
            sample_y = self.y.loc[sample_indices]
            feature_subset = np.random.choice(sample_X.columns, size=int(self.feature_fraction * n_features), replace=False)
            model = lgb.LGBMRegressor(**self.best_params, verbose=-1, random_state=self.random_state)
            model.fit(sample_X[feature_subset], sample_y,eval_metric=self.optimization_target)
            self.models.append((model, feature_subset))

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model, feature_subset in self.models:
            predictions += model.predict(X[feature_subset])
        return predictions / len(self.models)
    

    def fit_multiple_seeds(self, seeds):
        all_predictions = np.zeros((len(self.X), len(seeds)))
        
        for i, seed in enumerate(seeds):
            print(f"Training with seed {seed}")
            self.random_state = seed
            np.random.seed(seed)
            self.models = []
            self.feature_subsets = []
            self.fit()
            all_predictions[:, i] = self.predict(self.X)
        
        return np.mean(all_predictions, axis=1)
    
    def feature_importance(self):
        feature_importances = np.zeros(self.X.shape[1])
        for model, feature_subset in self.models:
            importances = model.feature_importances_
            for i, feature in enumerate(feature_subset):
                feature_importances[self.X.columns.get_loc(feature)] += importances[i] / self.n_models
        feature_importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        return feature_importance_df
    
    def forecast(self, X_last, n_periods):
        forecasts = []
        current_X = X_last.copy()
        current_date = current_X.index[-1].to_timestamp()
        current_date += pd.DateOffset(months=1)
        for _ in range(n_periods):
            pred = np.zeros(len(current_X))
            for model, feature_subset in self.models:
                pred += model.predict(current_X[feature_subset])
            pred /= len(self.models)
            forecasts.append(np.mean(pred, axis=0))  # Assuming single step forecast
            print(f"Forecast for {current_date}: {np.mean(pred, axis=0)}")
            # Update current_X for the next step forecast
            current_date += pd.DateOffset(months=1)
            current_X = self.update_features(current_X, np.mean(pred, axis=0), current_date)           
        return np.array(forecasts)
    
    def update_features(self, current_X, new_value, new_date):
        updated_X = current_X.copy()
        # Create a new row with the new predicted value
        new_row = updated_X.iloc[-1].copy()
        new_row['tn_lag_1'] = new_value  # Set the new prediction as the new lagged value
        # Shift existing lagged values
        for lag in range(2, 13):  # Assuming 12 lags
            if f'tn_lag_{lag}' in updated_X.columns:
                new_row[f'tn_lag_{lag}'] = updated_X.iloc[-2][f'tn_lag_{lag-1}']
            if f'tn_lag_{lag}' in updated_X.columns:
                new_row[f'cust_request_tn_lag_{lag}'] = updated_X.iloc[-2][f'cust_request_tn_lag_{lag-1}']
            if f'tn_lag_{lag}' in updated_X.columns:
                new_row[f'stock_final_lag_{lag}'] = updated_X.iloc[-2][f'stock_final_lag_{lag-1}']
        # Append the new row to the DataFrame using pd.concat
        new_row_df = pd.DataFrame([new_row], index=[new_date.to_period('M')])
        updated_X = pd.concat([updated_X, new_row_df])
        updated_X.fillna(0, inplace=True)
        return updated_X
