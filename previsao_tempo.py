import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import boto3
from io import StringIO
from dotenv import load_dotenv
from botocore.exceptions import ClientError

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
X_historico = df_historico[features]

dias_futuros = 30
data_inicial = df_historico['data'].max() + timedelta(days=1)
datas_futuras = [data_inicial + timedelta(days=i) for i in range(dias_futuros)]

df_futuros = pd.DataFrame({
    'data': datas_futuras,
    'umidade': df_historico['umidade'].values[-dias_futuros:],
    'velocidade_vento': df_historico['velocidade_vento'].values[-dias_futuros:]
})

df_futuros['dia_semana'] = df_futuros['data'].dt.dayofweek
df_futuros['mes'] = df_futuros['data'].dt.month
df_futuros['ano'] = df_futuros['data'].dt.year

X_futuros = df_futuros[features]

previsoes_rf = modelo_rf.predict(X_futuros)

df_futuros['temp_min_prevista'] = previsoes_rf[:, 0]
df_futuros['temp_max_prevista'] = previsoes_rf[:, 1]

plt.figure(figsize=(10, 5))
plt.plot(df_futuros['data'], df_futuros['temp_min_prevista'], label='Temp Min Prevista', marker='o')
plt.plot(df_futuros['data'], df_futuros['temp_max_prevista'], label='Temp Max Prevista', marker='x')
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
data_inicial = (df_historico['data'].max() + timedelta(days=1)).strftime('%d_%m_%Y')
arquivo_previsao = f'previsao_{data_inicial}.csv'

df_futuros.to_csv(os.path.join(previsoes_dir, arquivo_previsao), index=False)

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
df_futuros.to_csv(csv_buffer, index=False)

RED = "\033[91m"
RESET = "\033[0m"

try:
    s3_client.put_object(Bucket=bucket_name, Key=arquivo_saida, Body=csv_buffer.getvalue())
    print(f"Arquivo salvo no S3 em '{arquivo_saida}' no bucket '{bucket_name}'.")
except ClientError as e:
    if e.response['Error']['Code'] == 'InvalidAccessKeyId' or e.response['Error']['Code'] == 'AccessDenied':
        print(f"Erro: As {RED}Credenciais{RESET} do {RED}AWS{RESET} são inválidas.")
    else:
        print(f"Erro ao tentar salvar o arquivo no S3: {e}")
