import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

modelo_path = 'modelos/modelo_temperatura.pkl'

df = pd.read_csv('csv/dados_treinamento.csv')

df['data'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['dia_semana'] = df['data'].dt.dayofweek
df['mes'] = df['data'].dt.month
df['ano'] = df['data'].dt.year

features = ['wdir', 'wspd', 'pres', 'dia_semana', 'mes', 'ano']
targets = ['tmin', 'tmax']

X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

xgb_model = xgb.XGBRegressor(
    tree_method='hist',
    random_state=42,
    # device='cuda',
    nthread=12
)

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [10, 30],
    'min_child_weight': [1, 3],
    'learning_rate': [0.01, 0.05],
    'colsample_bytree': [0.8],
    'subsample': [0.8],
    'gamma': [0, 0.1]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=4,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=4
)

grid_search.fit(X_train, y_train)

modelo_rf = grid_search.best_estimator_

os.makedirs('modelos', exist_ok=True)
joblib.dump(modelo_rf, modelo_path)
print(f"Modelo treinado e salvo em '{modelo_path}'.")

y_pred_xgb = modelo_rf.predict(X_test)

mse_min_xgb = mean_squared_error(y_test['tmin'], y_pred_xgb[:, 0])
r2_min_xgb = r2_score(y_test['tmin'], y_pred_xgb[:, 0])
mse_max_xgb = mean_squared_error(y_test['tmax'], y_pred_xgb[:, 1])
r2_max_xgb = r2_score(y_test['tmax'], y_pred_xgb[:, 1])

print(f'MSE (tmin): {mse_min_xgb}')
print(f'R² (tmin): {r2_min_xgb}')
print(f'MSE (tmax): {mse_max_xgb}')
print(f'R² (tmax): {r2_max_xgb}')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(y_test['tmin'].values, label='Temp Min Real', marker='o')
plt.plot(y_pred_xgb[:, 0], label='Temp Min Prevista', marker='x')
plt.title('Comparação de Temperatura Mínima (Real vs Prevista)')
plt.xlabel('Amostras')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_test['tmax'].values, label='Temp Max Real', marker='o')
plt.plot(y_pred_xgb[:, 1], label='Temp Max Prevista', marker='x')
plt.title('Comparação de Temperatura Máxima (Real vs Prevista)')
plt.xlabel('Amostras')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
