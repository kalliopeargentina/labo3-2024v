import pandas as pd
from labolibrary import train_lightgbm_model, predict_next_month

def generate_sample_dataframe(n_products=3, n_months=24, start_date='2022-01'):
    date_range = pd.period_range(start=start_date, periods=n_months, freq='M')
    data = []
    for product_id in range(1, n_products + 1):
        for date in date_range:
            data.append({
                'date': date,
                'product_id': product_id,
                'feature_1': np.random.rand(),
                'feature_2': np.random.rand(),
                'tn_2': np.random.rand()
            })
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


# Generate sample dataframe
df = generate_sample_dataframe()

# Ensure the index is a PeriodIndex with monthly frequency
df.index = pd.PeriodIndex(df.index, freq='M')

# Train a single LightGBM model using all data
model, best_score = train_lightgbm_model(df)

# Prepare last data points for prediction
last_data_points = df[df.index == df.index.max()].copy()
last_data_points.drop(columns=['tn_2'], inplace=True)

# Predict the next month's value using the trained model
predictions = predict_next_month(model, last_data_points)

# Display predictions
print(predictions)

# Compare with actual values
actual_values = df[df.index == df.index.max()][['product_id', 'tn_2']]
print(actual_values)
