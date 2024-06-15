import importlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostRegressor, cv, Pool
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
# 1.1. Load dataset

def load_excel(excel_path, sheet, skip_rows, header_row):
        try:
            data = pd.read_excel(excel_path, sheet_name=sheet, skiprows=skip_rows, header=header_row)
            print("Dataset")
            return data
        except Exception as e:
            print(f"Nao deu {e}")
            return None


excel_path = r"C:\\Users\\Naja Informatica\\Desktop\\TCC\\database\\tst\\WorldEnergyBalancesHighlights2023.xlsx"
skip_rows = 1
header_row = 0

raw_data = load_excel(excel_path, 3, skip_rows, header_row)

# 1.2.1  Selecting only Brazil data and creating a new dataset
def filter_brazil(raw_data):
    filtered_df = raw_data[raw_data['Country'] == 'Brazil']
    return filtered_df

brasil_df = filter_brazil(raw_data)

#1.2.2 select only renewables production from brasil df

def filter_renewable(brasil_df):
    filtered_type = brasil_df[brasil_df['Product'].isin(['Renewables and waste','Renewable sources'])]
    return filtered_type

renew_br_df = filter_renewable(brasil_df)


# 1.3.1 Unpivot time column

melted_df = pd.melt(renew_br_df, id_vars=['Country', 'Product', 'Flow', 'NoCountry', 'NoProduct', 'NoFlow'],
                    var_name='Year', value_name='Value')


# 1.3.2 The flow column should be an attribute variable
pd.set_option('mode.chained_assignment', None)

def pivot_flow(melted_df):
    pivoted_df = melted_df.pivot(index='Year', columns='Flow', values='Value')
    return pivoted_df

pivoted_df = pivot_flow(melted_df)

# 1.4.1 problema na data
years_to_drop = range(1971, 1990)
pivoted_df = pivoted_df.drop(years_to_drop)
pivoted_df.replace("..", 0, inplace=True)
final_df = pivoted_df.astype(float)

final_df.reset_index(drop=False, inplace=True)
final_df = final_df[~final_df['Year'].astype(str).str.contains('Provisional')]
final_df.set_index('Year', inplace=True)
final_df.index = pd.to_datetime(final_df.index, format='%Y')
final_df = final_df.asfreq('YS')

if not pd.api.types.is_datetime64_any_dtype(final_df.index):
    final_df.index = pd.to_datetime(final_df.index, errors='coerce')
if final_df.index.isna().any():
    print("Cabecalho com na")

####################ETL  ACABOU############################

print(final_df.shape)
print(final_df.head)


#######VALIDAÇÃO##CRUZADA########VALIDAÇÃO##CRUZADA########VALIDAÇÃO##CRUZADA##########VALIDAÇÃO###CRUZADA############
y = final_df['Production (PJ)']
n_splits = len(y) - 3

tscv = TimeSeriesSplit(n_splits=n_splits)
for train_index, test_index in tscv.split(y):
    train, test = y.iloc[train_index], y.iloc[test_index]

# teste de overlap (deletado)

##########ARIMA############ARIMA#########ARIMA############ARIMA#################ARIMA############ARIMA########

# Auto arima (deletado)
# modelo
p, d, q = 2, 2, 1

model = sm.tsa.ARIMA(train, order=(p, d, q))
fitted_model = model.fit()

#residuals de treino
arima_residuals = fitted_model.resid
arima_resid = pd.DataFrame({'Actual': final_df['Production (PJ)'] , 'Predicted': fitted_model.fittedvalues, 'Residual': arima_residuals})

print(arima_resid)

#forecast
forecast_years = pd.date_range(start='2023', periods=12, freq='YE')
forecast_result = fitted_model.get_forecast(steps=12)
test_predictions = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()
arima_predicted = fitted_model.fittedvalues

results_arima_df = pd.DataFrame({
    'Ano': forecast_years.year,
    'Forecast': test_predictions,
})
print(results_arima_df)



#AVALIAÇÃO
train_rmse = mean_squared_error(train, arima_predicted, squared=False)
train_mae = mean_absolute_error(train, arima_predicted)
train_mape = np.mean(np.abs((train - arima_predicted) / train)) * 100
train_r2 = r2_score(train, arima_predicted)

print(f'RMSE Treino: {train_rmse}')
print(f'MAE Treino: {train_mae}')
print(f'MAPE Treino: {train_mape}%')

# graficos
plt.figure(figsize=(12, 8))
plt.plot(final_df.index, final_df['Production (PJ)'], label='Dados originais', color='blue')
plt.plot(results_arima_df.index, results_arima_df['Forecast'], label='Previsão', color='red')
plt.title('Original e Previsto')
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

#grafico de residuals ########## grafico horrendo arrumar
plt.figure(figsize=(10, 6))
plt.plot(arima_residuals)
plt.title('residuals')
plt.xlabel('anos')
plt.ylabel('Residuals')
plt.show()


#######CATBOOST#############CATBOOST#############CATBOOST#############CATBOOST#############CATBOOST
final_df.reset_index(inplace=True)
print(final_df.columns)
final_df['Year'] = final_df['Year'].dt.year
final_df = final_df.sort_values('Year')

final_df.dropna(inplace=True)

def create_lagged_features(df, target_column, lags=3):
    for lag in range(1, lags + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df.dropna()

lags = 3

final_df = create_lagged_features(final_df, 'Production (PJ)', lags)

features = final_df[['Commercial and public services (PJ)', 'Electricity output (GWh)',
                     'Electricity, CHP and heat plants (PJ)',
                     'Industry (PJ)',
                     'Total energy supply (PJ)', 'Total final consumption (PJ)'] +
                    [f'Production (PJ)_lag_{i}' for i in range(1, lags + 1)]]
target = final_df['Production (PJ)']

data_pool = Pool(features, target)

train_size = int(0.7 * len(final_df))
train_df = final_df[:train_size]
test_df = final_df[train_size:]

train_features = train_df.drop(columns=['Production (PJ)'])
train_target = train_df['Production (PJ)']

test_features = test_df.drop(columns=['Production (PJ)'])
test_target = test_df['Production (PJ)']

train_pool = Pool(train_features, train_target)


params = {
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 3,
    'l2_leaf_reg': 1,
    'one_hot_max_size': 5,
    'colsample_bylevel': 0.8,
    'bagging_temperature': 0.2,
    'random_strength': 1,
    'subsample': 1,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'task_type': 'CPU',
    'verbose': 300,
    'random_state': 42
}

# 3-fold
cv_results = cv(
    params=params,
    dtrain=data_pool,
    fold_count=3,
    shuffle=True,
    partition_random_seed=42,
    verbose=False
)

model = CatBoostRegressor(**params)
model.fit(data_pool)

def iterative_forecast(model, initial_features, steps=13):
    forecasted_values = []
    current_features = initial_features.copy()

    for _ in range(steps):
        prediction = model.predict([current_features])[0]
        forecasted_values.append(prediction)
        current_features = current_features[1:] + [prediction]

    return forecasted_values

last_known_values = features.iloc[-1].tolist()

forecast = iterative_forecast(model, last_known_values, steps=13)

# Create a DataFrame with the forecasted values
years = list(range(2023, 2023 + len(forecast)))
forecast_df = pd.DataFrame(forecast, index=years, columns=['Forecasted Production (PJ)'])

print("Previsão de 13 passos:", forecast_df)

# GRAFICO
plt.figure(figsize=(12, 6))
plt.plot(forecast_df.index, forecast_df['Forecasted Production (PJ)'], label='Forecasted', marker='x')
plt.xlabel('Ano')
plt.ylabel('Production (PJ)')
plt.title('Produção de 13 passos prevista com Catboost')
plt.legend()
plt.show()

# AVALIAÇÃO
train_predictions = model.predict(train_features)
train_rmse = mean_squared_error(train_target, train_predictions, squared=False)
train_mae = mean_absolute_error(train_target, train_predictions)
train_mape = np.mean(np.abs((train_target - train_predictions) / train_target)) * 100
train_r2 = r2_score(train_target, train_predictions)

print(f'RMSE Treino: {train_rmse}')
print(f'MAE Treino: {train_mae}')
print(f'MAPE Treino: {train_mape}%')
print(f'R² Treino: {train_r2}')

test_predictions = model.predict(test_features)
test_rmse = mean_squared_error(test_target, test_predictions, squared=False)
test_mae = mean_absolute_error(test_target, test_predictions)
test_mape = np.mean(np.abs((test_target - test_predictions) / test_target)) * 100
test_r2 = r2_score(test_target, test_predictions)

print(f'RMSE Teste: {test_rmse}')
print(f'MAE Teste: {test_mae}')
print(f'MAPE Teste: {test_mape}%')
print(f'R² Teste: {test_r2}')

##################################stacking#######################################
''' junta os modelos as predições num df só pra facilitar antes, dps dos  resultados 
 preliminares e arrumar o problema do overfiting tentar rodar o negocio de 
 stacking
 '''



forecast_arimaaa = np.array([5955.54, 6077.56, 6189.61, 6298.57, 6405.47, 6511.51, 6617.08, 6722.44, 6827.68, 6932.88, 7038.04, 7143.19])
forecast_catboost = np.array([5556.76, 4531.83, 4498.15, 4765.71, 4639.15, 4663.96, 4567.11, 4570.11, 4581.84, 4551.25, 4583.18, 4546.58])

combinado = (forecast_arimaaa + forecast_catboost) / 2

forecast_years = list(range(2023, 2023 + len(combinado)))
combinado_df = pd.DataFrame({
    'Year': forecast_years,
    'Junção Forecast': combinado
})

print(combinado_df)



plt.figure(figsize=(12, 6))
plt.plot(final_df['Year'], final_df['Production (PJ)'], label='Dados originais', color='blue')
plt.plot(combinado_df['Year'], combinado_df['Combined Forecast'], label='Combinado', color='green', marker='x')
plt.xlabel('Ano')
plt.ylabel('Produção')
plt.title('Junção de previsões')
plt.legend()
plt.grid(True)
plt.show()

















