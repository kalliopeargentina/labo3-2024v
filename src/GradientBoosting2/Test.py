

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data into DataFrames
df_sell_in = pd.read_csv('src/GradientBoosting2/data/sell-in.txt', sep='\t')
df_tb_productos = pd.read_csv('src/GradientBoosting2/data/tb_productos.txt', sep='\t')
df_tb_stocks = pd.read_csv('src/GradientBoosting2/data/tb_stocks.txt', sep='\t')

# Fill any missing values in categorical columns
categorical_columns = ['cat1', 'cat2', 'cat3', 'brand', 'sku_size']
for col in categorical_columns:
    df_tb_productos[col].fillna('Unknown', inplace=True)

# Convert relevant columns in df_tb_productos to categorical
for col in categorical_columns:
    df_tb_productos[col] = df_tb_productos[col].astype('category')

# Join tb_productos to sell_in on product_id
df_sell_in_merged = pd.merge(df_sell_in, df_tb_productos, on='product_id', how='left')

# Convert 'periodo' to datetime for better feature extraction
df_sell_in_merged['periodo'] = pd.to_datetime(df_sell_in_merged['periodo'], format='%Y%m')

# Join tb_stocks to sell_in_merged on both product_id and periodo
df_final = pd.merge(df_sell_in_merged, df_tb_stocks, on=['product_id', 'periodo'], how='left')

# Convert 'periodo' to datetime features
df_final['year'] = df_final['periodo'].dt.year
df_final['month'] = df_final['periodo'].dt.month

# Fill any remaining missing values
df_final.fillna(0, inplace=True)

# Encode categorical variables
df_final_encoded = pd.get_dummies(df_final, columns=categorical_columns)

# Define the target variable and features
X = df_final_encoded.drop(columns=['cust_request_qty', 'cust_request_tn', 'tn', 'periodo'])
y = df_final_encoded['tn']  # Assuming 'tn' is the sales column we want to predict

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'Root Mean Squared Error: {rmse}')

# Display the first few predictions alongside actual values
df_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_predictions.head())
