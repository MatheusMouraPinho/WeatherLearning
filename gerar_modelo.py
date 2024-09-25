import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('csv/dados_treinamento.csv')

df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
df['dia_semana'] = df['data'].dt.dayofweek  # 0 = Segunda-feira, 6 = Domingo
df['mes'] = df['data'].dt.month
df['ano'] = df['data'].dt.year

features = ['umidade', 'velocidade_vento', 'dia_semana', 'mes', 'ano']
targets = ['temp_min', 'temp_max']

X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Melhores parâmetros encontrados:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

# teste model
y_pred_rf = best_rf_model.predict(X_test)

mse_min_rf = mean_squared_error(y_test['temp_min'], y_pred_rf[:, 0])
r2_min_rf = r2_score(y_test['temp_min'], y_pred_rf[:, 0])
mse_max_rf = mean_squared_error(y_test['temp_max'], y_pred_rf[:, 1])
r2_max_rf = r2_score(y_test['temp_max'], y_pred_rf[:, 1])

print(f'MSE (temp_min): {mse_min_rf}')
print(f'R² (temp_min): {r2_min_rf}')
print(f'MSE (temp_max): {mse_max_rf}')
print(f'R² (temp_max): {r2_max_rf}')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(y_test['temp_min'].values, label='Temp Min Real', marker='o')
plt.plot(y_pred_rf[:, 0], label='Temp Min Prevista', marker='x')
plt.title('Comparação de Temperatura Mínima (Real vs Prevista)')
plt.xlabel('Amostras')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_test['temp_max'].values, label='Temp Max Real', marker='o')
plt.plot(y_pred_rf[:, 1], label='Temp Max Prevista', marker='x')
plt.title('Comparação de Temperatura Máxima (Real vs Prevista)')
plt.xlabel('Amostras')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
