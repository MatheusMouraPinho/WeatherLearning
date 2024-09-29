import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import boto3
from io import StringIO
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import glob
import re

load_dotenv()

modelo_path = 'modelos/modelo_temperatura.pkl'
modelo_rf = joblib.load(modelo_path)
print("Modelo carregado com sucesso!")

diretorio_historico = 'csv/historico'

padrao_arquivos = '*_a_*.csv'

arquivos = glob.glob(os.path.join(diretorio_historico, padrao_arquivos))

if not arquivos:
    print("Sem dados de hoje aqui.")
    exit()

arquivos_sorted = sorted(
    arquivos,
    key=lambda x: datetime.strptime(re.search(r'_(\d{2}_\d{2}_\d{4})\.csv$', x).group(1), '%d_%m_%Y'),
    reverse=True
)

arquivo_mais_recente = arquivos_sorted[0]
print(f"Arquivo histórico encontrado: {arquivo_mais_recente}")

df_historico = pd.read_csv(arquivo_mais_recente)
print("\nDados de previsão carregados:")
print(df_historico)

df_historico['data'] = pd.to_datetime(df_historico['date'], format='%Y-%m-%d')
df_historico['dia_semana'] = df_historico['data'].dt.dayofweek
df_historico['mes'] = df_historico['data'].dt.month
df_historico['ano'] = df_historico['data'].dt.year

features = ['wdir', 'wspd', 'pres', 'dia_semana', 'mes', 'ano']
X_historico = df_historico[features]

dias_futuros = 30
data_inicial = df_historico['data'].max() + timedelta(days=1)
datas_futuras = [data_inicial + timedelta(days=i) for i in range(dias_futuros)]
data_final = datas_futuras[-1]

df_futuros = pd.DataFrame({
    'data': datas_futuras,
    'wdir': df_historico['wdir'].values[-dias_futuros:],
    'wspd': df_historico['wspd'].values[-dias_futuros:],
    'pres': df_historico['pres'].values[-dias_futuros:]
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
plt.title('Previsões de Temperatura')
plt.xlabel('Data')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

previsoes_dir = 'csv/previsoes'
os.makedirs(previsoes_dir, exist_ok=True)

data_inicial_str = data_inicial.strftime('%d_%m_%Y')
data_final_str = data_final.strftime('%d_%m_%Y')

arquivo_previsao = f'{data_inicial_str}_a_{data_final_str}.csv'

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
