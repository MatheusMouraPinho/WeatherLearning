import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

df_novos_dados = pd.read_csv('csv/historico/dados_base.csv')

df_novos_dados['datetime'] = pd.to_datetime(df_novos_dados['datetime'])

# Simula "hoje" como 6 dias adiante para testar, pois
# a API traz a previsão de 5 dias (previsão baseada na previsão)

hoje = datetime.now().date() + timedelta(days=6)

definir_periodo_do_dia = lambda hora: 'Manhã' if 5 <= hora < 12 else 'Tarde' if 12 <= hora < 18 else 'Noite'
df_novos_dados['periodo'] = df_novos_dados['datetime'].dt.hour.apply(definir_periodo_do_dia)

df_historico = df_novos_dados[df_novos_dados['datetime'].dt.date < hoje].copy()

model = LinearRegression()
previsoes = {}

# Treinar e prever para cada período (manhã, tarde, noite)
for periodo in ['Manhã', 'Tarde', 'Noite']:
    df_periodo_historico = df_historico[df_historico['periodo'] == periodo]

    # Separar as features e o target
    X_historico = df_periodo_historico[['humidity_percent', 'wind_speed_mps', 'feels_like_celsius']]
    y_historico = df_periodo_historico['temp_celsius']

    # Treinar o modelo para este período
    if not X_historico.empty:
        model.fit(X_historico, y_historico)

        mean_values = X_historico.mean().to_frame().T
        previsao = model.predict(mean_values)[0]
        previsoes[periodo] = previsao
    else:
        print(f"Sem dados históricos suficientes para treinar o modelo para {periodo}.")

df_previsao = pd.DataFrame({
    'datetime': [datetime.combine(hoje, datetime.min.time()) + timedelta(hours=8),
                 datetime.combine(hoje, datetime.min.time()) + timedelta(hours=14),
                 datetime.combine(hoje, datetime.min.time()) + timedelta(hours=20)],
    'periodo': ['Manhã', 'Tarde', 'Noite'],
    'previsao_temp': [previsoes.get('Manhã', None),
                      previsoes.get('Tarde', None),
                      previsoes.get('Noite', None)]
})

print(df_previsao)
data_formatada = hoje.strftime('%d_%m_%Y')
df_previsao.to_csv(f'csv/previsao/previsao_tempo_{data_formatada}.csv', index=False)
