import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from feature_baggingV2 import FeatureBaggingWithHyperparamTuning
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
#import lightgbm as lgb
import datetime
from sklearn.preprocessing import MinMaxScaler

import os, sys, gc, time, warnings, pickle, psutil, random

# Función para escalar y devolver una serie
def minmax_scale_group(group):
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(group.values.reshape(-1, 1)).flatten()
    scalers[group.name] = scaler  # Almacenar el escalador para este grupo
    return pd.Series(scaled_values, index=group.index)

# Función para desescalar y devolver una serie
def inverse_minmax_scale_group(group):
    scaler = scalers[group.name]
    inversed_values = scaler.inverse_transform(group.values.reshape(-1, 1)).flatten()
    return pd.Series(inversed_values, index=group.index)

# Definir la métrica personalizada
def multinacional_metric(y_true, y_pred):
    return abs(sum(y_true - y_pred)) / sum(y_true)

DATOS_DIR = '~/buckets/b1/datasets/'
df_sell_in = pd.read_csv(DATOS_DIR+'sell-in.txt', sep='\t')
df_predecir = pd.read_csv(DATOS_DIR+'productos_a_predecir.txt', sep='\t')
df_tb_stocks = pd.read_csv(DATOS_DIR+'tb_stocks.txt', sep='\t')
df_tb_productos = pd.read_csv(DATOS_DIR+'tb_productos_descripcion.txt', sep='\t')

df_tb_stocks['periodo'] = pd.to_datetime(df_tb_stocks['periodo'], format='%Y%m')
df_tb_stocks['periodo'] = df_tb_stocks['periodo'] - pd.DateOffset(months=1) #mes diferente en stock


df_tb_stocks['product_id'] = df_tb_stocks['product_id'].astype(int)
df_tb_stocks['stock_final'] = df_tb_stocks['stock_final'].astype(float)
df_tb_productos['product_id'] = df_tb_productos['product_id'].astype(int)
df_tb_productos['sku_size'] = df_tb_productos['sku_size'].astype(int)
df_sell_in['periodo'] = pd.to_datetime(df_sell_in['periodo'], format='%Y%m')
df_sell_in['product_id'] = df_sell_in['product_id'].astype(int)
df_sell_in['customer_id'] = df_sell_in['customer_id'].astype(int)
df_sell_in['cust_request_qty'] = df_sell_in['cust_request_qty'].astype(int)
df_sell_in['cust_request_tn'] = df_sell_in['cust_request_tn'].astype(float)
df_sell_in['tn'] = df_sell_in['tn'].astype(float)
df_sell_in['plan_precios_cuidados'] = df_sell_in['plan_precios_cuidados'].astype(bool)

df_sell_in['tn_2'] = df_sell_in['tn'].shift(-2) #OBTENGO TN 2 MESES ADELANTE (CLASE FINAL)

# Join tb_productos to sell_in on product_id
df_sell_in_merged = pd.merge(df_sell_in, df_tb_productos, on='product_id', how='left')
# Join tb_stocks to sell_in_merged on both product_id and periodo
df_final = pd.merge(df_sell_in_merged, df_tb_stocks, on=['product_id', 'periodo'], how='left')

#Convertir 'periodo' a formato de fecha y Calcular el trimestre desde 'periodo'
df_final['fecha'] = pd.to_datetime(df_final['periodo'], format='%Y%m')
df_final["trimestre"] = df_final.periodo.dt.quarter

#Establecer 'fecha' como índice y convertir a período mensual
df_final.set_index('fecha', inplace=True)
df_final.index = df_final.index.to_period('M')

df_final['diff_tn_tn2'] =  df_final['tn_2'] - df_final['tn'] #NUEVA CLASE

# Create lag variables for 'cust_request_qty', 'cust_request_tn', and 'tn'
n_lags = 36
for lag in range(1, n_lags + 1):
    df_final[f'cust_request_qty_lag_{lag}'] = df_final['cust_request_qty'].shift(lag)
    df_final[f'cust_request_tn_lag_{lag}'] = df_final['cust_request_tn'].shift(lag)
    df_final[f'stock_final_lag_{lag}'] = df_final['stock_final'].shift(lag)
    df_final[f'tn_lag_{lag}'] = df_final['tn'].shift(lag)

#DELTA DE LAGS 2 ANTERIORES

for lag in range(1, n_lags-1):
    df_final[f'delta_cust_request_tn_{lag}_{lag+2}'] = (
        df_final[f'cust_request_tn_lag_{lag+2}'] - df_final[f'cust_request_tn_lag_{lag}']
    )
    df_final[f'delta_stock_final_{lag}_{lag+2}'] = (
        df_final[f'stock_final_lag_{lag+2}'] - df_final[f'stock_final_lag_{lag}']
    )
    df_final[f'delta_tn_{lag}_{lag+2}'] = (
        df_final[f'tn_lag_{lag+2}'] - df_final[f'tn_lag_{lag}']
    )
#DELTA DE LAGS 1 ANTERIOR

for lag in range(1, n_lags):
    df_final[f'delta_cust_request_tn_{lag}_{lag+1}'] = (
        df_final[f'cust_request_tn_lag_{lag+1}'] - df_final[f'cust_request_tn_lag_{lag}']
    )
    df_final[f'delta_stock_final_{lag}_{lag+1}'] = (
        df_final[f'stock_final_lag_{lag+1}'] - df_final[f'stock_final_lag_{lag}']
    )
    df_final[f'delta_tn_{lag}_{lag+1}'] = (
        df_final[f'tn_lag_{lag+1}'] - df_final[f'tn_lag_{lag}']
    )
#Mes actual / (lag 2 + lag 3) 

# Calcular el ratio del valor actual con respecto a la suma de los dos lags anteriores
for lag in range(3, n_lags + 1):
    df_final[f'ratio_cust_request_tn_{lag}'] = (
        df_final[f'cust_request_tn_lag_{lag}'] / (
            df_final[f'cust_request_tn_lag_{lag-1}'] + df_final[f'cust_request_tn_lag_{lag-2}']
        )
    )
    df_final[f'ratio_stock_final_{lag}'] = (
        df_final[f'stock_final_lag_{lag}'] / (
            df_final[f'stock_final_lag_{lag-1}'] + df_final[f'stock_final_lag_{lag-2}']
        )
    )
    df_final[f'ratio_tn_{lag}'] = (
        df_final[f'tn_lag_{lag}'] / (
            df_final[f'tn_lag_{lag-1}'] + df_final[f'tn_lag_{lag-2}']
        )
    )
df_final.reset_index(inplace=True)

#MEDIAS MOVILES
rolling_windows = [3, 6, 9, 12, 24, 36]

# Agrupamos por 'product_id' y calculamos las medias móviles para 'tn'
for window in rolling_windows:
    df_final[f'rolling_mean_tn_{window}'] = df_final.groupby('product_id')['tn'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
# Datetime features
df_final['year'] = df_final['periodo'].dt.year
df_final['month'] = df_final['periodo'].dt.month
df_final['quarter'] = df_final.periodo.dt.quarter

#Variables Dummies si es el max o el min de cierta cantidad de meses
months = [3, 6, 9, 12]

# Agrupamos por 'product_id' y calculamos las medias móviles para 'tn'
for i in months:
    df_final[f'max_{i}m'] = df_final.groupby('product_id')['tn'].transform(lambda x: x.rolling(i, min_periods=1).max())
    df_final[f'min_{i}m'] = df_final.groupby('product_id')['tn'].transform(lambda x: x.rolling(i, min_periods=1).min())
    # Crear las dummies
    df_final[f'dummy_max_{i}m'] = np.where(df_final['tn'] == df_final[f'max_{i}m'], 1, 0)
    df_final[f'dummy_min_{i}m'] = np.where(df_final['tn'] == df_final[f'min_{i}m'], 1, 0)
    
#Calcular ventas por trimestre
df_final['tn_trimestre'] = df_final.groupby(['trimestre', 'product_id'])['tn'].transform('sum')
#Calcular ventas por trimestre por cliente
df_final['tn_trimestre_customer'] = df_final.groupby(['trimestre','customer_id', 'product_id'])['tn'].transform('sum')


df_final['tn_product_id'] = df_final.groupby(['periodo', 'product_id'])['tn'].transform('sum')


# Leer el archivo exportado
df_exported = pd.read_excel('23variables_externas.xlsx')

# Asegúrate de que las columnas de fecha estén en el formato datetime
df_exported['fecha'] = pd.to_datetime(df_exported['fecha'])

# Unir los DataFrames por la columna de fecha
df_merged = pd.merge(df_final, df_exported, on='fecha', how='left')

df_final['plan_precios_cuidados'] =df_final['plan_precios_cuidados'].astype(bool)
#df_final['dias_fin_trimestre'] = df_final['dias_fin_trimestre'].dt.days.astype(int) #TARDA!
df_final = df_final.drop(columns=['fecha'])

# #Cambiar las variables categoricas y hacer one-hot encoding

df_final["cat1"] = df_final["cat1"].astype("category")
df_final["cat2"] = df_final["cat2"].astype("category")
df_final["cat3"] = df_final["cat3"].astype("category")
df_final["brand"] = df_final["brand"].astype("category")
df_final["descripcion"] = df_final["descripcion"].astype("category")

# # Encode categorical variables explicitly. One-hot encoding
cat1_dummies = pd.get_dummies(df_final['cat1'], prefix='cat1', drop_first=True)
cat2_dummies = pd.get_dummies(df_final['cat2'], prefix='cat2', drop_first=True)
cat3_dummies = pd.get_dummies(df_final['cat3'], prefix='cat3', drop_first=True)
brand_dummies = pd.get_dummies(df_final['brand'], prefix='brand', drop_first=True)
descripcion_dummies = pd.get_dummies(df_final['descripcion'], prefix='descripcion', drop_first=True)

# # Concatenate the dummy variables to the DataFrame and drop the original categorical columns
df_final= pd.concat([df_final, cat1_dummies, cat2_dummies, cat3_dummies, brand_dummies, descripcion_dummies], axis=1)
df_final.drop(columns=['cat1', 'cat2', 'cat3', 'brand'], inplace=True)

df_final.set_index('periodo', inplace=True)
df_final.index = df_final.index.to_period('M')
df_final.sort_index(inplace=True)



df_final.to_parquet('./FE_dataset-CARLA.parquet', engine='pyarrow')  


