import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('csv/dados_treinamento.csv')

# Separar as features (X) e o target (y)
X = df[['humidity_percent', 'wind_speed_mps', 'feels_like_celsius']]
y = df['temp_celsius']

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R²: {r2}')

# Visualizar as previsões
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Valores Reais')
plt.plot(y_pred, label='Previsões')
plt.legend()
plt.title('Comparação entre Previsões e Valores Reais')
plt.xlabel('Amostras')
plt.ylabel('Temperatura (°C)')
plt.show()

# Salvar o modelo treinado
joblib.dump(model, 'modelos/modelo_temperatura.pkl')
