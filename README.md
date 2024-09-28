# Tech Challenge Fiap - Desenvolvedores

- Gabriel Sargeiro ([LinkedIn](https://www.linkedin.com/in/gabriel-sargeiro/))
- Guilherme Lobo ([LinkedIn](https://www.linkedin.com/in/guilhermegclobo/))
- Matheus Moura ([LinkedIn](https://www.linkedin.com/in/matheus-moura-pinho-55a25b186/))

# Projeto WeatherLearning

O projeto **WeatherLearning** é um sistema de aprendizado de máquina para previsão de temperaturas mínimas e máximas com base em dados meteorológicos, utilizando o modelo **Gradient Boosting** (XGBoost).

- Fonte de dados:
   - (a decidir)

## Estrutura do Projeto (pendente)

    WeatherLearning/
    ├── .venv/                          # Ambiente virtual
    ├── csv/
    │   ├── previsoes/                  # Previsões geradas pelo modelo
    │   │   └── previsao_24_09_2024.csv # Exemplo de previsão gerada
    │   ├── dados_atual.csv             # Dados atuais para previsão
    │   └── dados_treinamento.csv       # Dados históricos para treinamento do modelo
    ├── modelos/
    │   └── modelo_temperatura.pkl      # Modelo Gradient Boosting treinado para previsão
    ├── .env                            # Arquivo com credenciais da AWS
    ├── .gitignore
    ├── gerar_modelo.py                 # Script para treinar o modelo
    ├── previsao_tempo_real.py          # Script para gerar previsões
    ├── README.md
    └── requirements.txt                # Lista de dependências do projeto

