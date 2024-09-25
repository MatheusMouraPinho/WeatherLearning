import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import boto3
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

modelo_path = 'modelos/modelo_temperatura.pkl'
modelo_rf = joblib.load(modelo_path)
print("Modelo Random Forest carregado com sucesso!")

df_historico = pd.read_csv('csv/dados_atual.csv')
print("\nDados de previsão carregados:")
print(df_historico)

df_historico['data'] = pd.to_datetime(df_historico['data'], format='%Y-%m-%d')
df_historico['dia_semana'] = df_historico['data'].dt.dayofweek  # 0 = Segunda-feira, 6 = Domingo
df_historico['mes'] = df_historico['data'].dt.month
df_historico['ano'] = df_historico['data'].dt.year

features = ['umidade', 'velocidade_vento', 'dia_semana', 'mes', 'ano']
X_futuros = df_historico[features]

previsoes_rf = modelo_rf.predict(X_futuros)

df_historico['temp_min_prevista'] = previsoes_rf[:, 0]
df_historico['temp_max_prevista'] = previsoes_rf[:, 1]

print("\nDataFrame com Previsões:")
print(df_historico)

plt.figure(figsize=(10, 5))
plt.plot(df_historico['data'], df_historico['temp_min_prevista'], label='Temp Min Prevista', marker='o')
plt.plot(df_historico['data'], df_historico['temp_max_prevista'], label='Temp Max Prevista', marker='x')
plt.title('Previsões de Temperatura (Random Forest)')
plt.xlabel('Data')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

previsoes_dir = 'csv/previsoes'
os.makedirs(previsoes_dir, exist_ok=True)
data_atual = datetime.now().strftime('%d_%m_%Y')
arquivo_previsao = f'previsao_{data_atual}.csv'

df_historico.to_csv(os.path.join(previsoes_dir, arquivo_previsao), index=False)
print(f"\nPrevisões salvas em '{previsoes_dir}/{arquivo_previsao}'.")

aws_access_key_id = os.getenv('aws_access_key_id')
aws_secret_access_key = os.getenv('aws_secret_access_key')
aws_session_token = os.getenv('aws_session_token')
aws_region = os.getenv('aws_region')

s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=aws_region
)

bucket_name = 'weather-learning-bucket'
arquivo_saida = f'previsoes/{arquivo_previsao}'

csv_buffer = StringIO()
df_historico.to_csv(csv_buffer, index=False)

s3_client.put_object(Bucket=bucket_name, Key=arquivo_saida, Body=csv_buffer.getvalue())
print(f"Arquivo salvo no s3 em '{arquivo_saida}' no bucket '{bucket_name}'.")
