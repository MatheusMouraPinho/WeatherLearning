# Tech Challenge Fiap - Desenvolvedores

- Gabriel Sargeiro ([LinkedIn](https://www.linkedin.com/in/gabriel-sargeiro/))
- Guilherme Lobo ([LinkedIn](https://www.linkedin.com/in/guilhermegclobo/))
- Matheus Moura ([LinkedIn](https://www.linkedin.com/in/matheus-moura-pinho-55a25b186/))

# Projeto WeatherLearning

O projeto **WeatherLearning** é um sistema de aprendizado de máquina para previsão de temperaturas mínimas e máximas com base em dados meteorológicos, utilizando o modelo **Gradient Boosting** (XGBoost).

- Fonte de dados:
   - https://meteostat.net/

## Estrutura do Projeto

    WeatherLearning/
    ├── .venv/                                # Ambiente virtual
    ├── csv/
    │   ├── historico/                        # Dados para previsão
    │   │   └── 30_08_2024_a_29_09_2024.csv
    │   ├── previsoes/                        # Previsões geradas pelo modelo
    │   │   └── 30_09_2024_a_29_10_2024.csv
    │   └── dados_treinamento.csv             # Dados históricos para treinamento do modelo
    ├── modelos/
    │   └── modelo_temperatura.pkl            # Modelo Gradient Boosting treinado para previsão
    ├── .env                                  # Arquivo com credenciais da AWS
    ├── .gitignore
    ├── api.py                                # API que coleta os dados da fonte
    ├── gerar_modelo.py                       # Script para treinar o modelo
    ├── previsao_tempo.py                     # Script para gerar previsões
    ├── README.md
    └── requirements.txt                      # Lista de dependências do projeto


## Configuração do Projeto

### 1. Ativar o ambiente virtual

No PowerShell (Windows):
```
.\.venv\Scripts\Activate
```

No Linux ou macOS:
```
source .venv/bin/activate
```

### 2. Instalar as dependências

Com o ambiente virtual ativado, execute o seguinte comando para instalar as dependências do projeto:

```
pip install -r requirements.txt
```

### 3. Executar o servidor local

Após instalar as dependências, inicie o servidor local com o comando abaixo:

```
uvicorn api:app --reload
```

Você pode acessar a documentação interativa da API via FastAPI no endereço:

```
http://127.0.0.1:8000/docs
```
