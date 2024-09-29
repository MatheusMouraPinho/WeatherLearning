from fastapi import FastAPI, Query
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta
import logging
import os
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import time

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

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

BASE_URL = os.getenv("BASE_URL", "https://meteostat.net/pt/place/br/sao-paulo")

DOWNLOAD_DIR = os.path.abspath("csv/historico")

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": DOWNLOAD_DIR,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def fetch_data_from_csv(download_path: str):
    try:
        df = pd.read_csv(download_path)
        logging.info(f"CSV carregado com sucesso de '{download_path}'.")
        return df
    except Exception as e:
        logging.error(f"Erro ao ler o CSV: {str(e)}")
        return pd.DataFrame()

def upload_to_s3_file(file_path, bucket_name, arquivo_saida):
    try:
        with open(file_path, 'rb') as data:
            s3_client.upload_fileobj(data, bucket_name, arquivo_saida)
        logging.info(f"Arquivo '{arquivo_saida}' enviado para o bucket '{bucket_name}'.")
    except ClientError as e:
        if e.response['Error']['Code'] in ['InvalidAccessKeyId', 'AccessDenied']:
            logging.error("Erro: Credenciais AWS inválidas.")
        else:
            logging.error(f"Erro ao enviar o arquivo para o S3: {e}")

def clear_download_dir():
    try:
        export_file = os.path.join(DOWNLOAD_DIR, "export.csv")
        if os.path.exists(export_file):
            os.remove(export_file)
            logging.debug(f"Arquivo removido: {export_file}")
            logging.info("Arquivo temporário 'export.csv' removido com sucesso.")
        else:
            logging.debug("Nenhum arquivo temporário 'export.csv' encontrado para remover.")
    except Exception as e:
        logging.error(f"Falha ao limpar o arquivo temporário 'export.csv': {str(e)}")

@app.get(
    "/scrape_wind_pressure",
    summary="Velocidade do Vento e Pressão do Ar",
    response_description="Retorna a Velocidade do Vento e Pressão do Ar extraídos de um intervalo de dias.",
    response_model=dict
)
def get_wind_pressure(
    days: int = Query(
        30,
        ge=30,
        description="Número de dias para buscar a partir da data de hoje. Mínimo de 30 dias."
    )
):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start_date_str = start_date.strftime("%d_%m_%Y")
    end_date_str = end_date.strftime("%d_%m_%Y")

    arquivo_historico = f'{start_date_str}_a_{end_date_str}.csv'
    logging.info(f"Nome do arquivo histórico: {arquivo_historico}")

    url = f"{BASE_URL}?s=83779&t={start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    logging.info(f"Acessando a URL: {url}")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    try:
        driver.get(url)

        wait = WebDriverWait(driver, 20)

        try:
            accept_button = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[@class='btn btn-primary' and @data-bs-dismiss='modal']")
                )
            )
            accept_button.click()
            logging.info("Botão 'Aceitar' clicado com sucesso.")
        except Exception as e:
            logging.warning("Botão 'Aceitar' não encontrado ou já foi clicado.")

        try:
            export_button = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//*[@data-bs-toggle='modal' and @data-bs-target='#export-modal']")
                )
            )
            export_button.click()
            logging.info("Botão de exportação clicado com sucesso.")
        except Exception as e:
            logging.error(f"Não foi possível clicar no botão de exportação: {str(e)}")
            return {"erro": "Falha ao abrir o modal de exportação."}

        try:
            format_select_element = wait.until(
                EC.visibility_of_element_located((By.ID, "formatSelect"))
            )
            select = Select(format_select_element)
            select.select_by_value("csv")
            logging.info("Formato 'CSV' selecionado com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao selecionar o formato CSV: {str(e)}")
            return {"erro": "Falha ao selecionar o formato CSV."}

        try:
            save_button = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//div[@class='modal-footer']//button[@class='btn btn-primary']")
                )
            )
            save_button.click()
            logging.info("Botão 'Salvar' clicado com sucesso. Iniciando download...")
        except Exception as e:
            logging.error(f"Erro ao clicar no botão 'Salvar': {str(e)}")
            return {"erro": "Falha ao iniciar o download do CSV."}

        timeout = 20
        polling_interval = 1
        elapsed = 0
        download_file = None

        while elapsed < timeout:
            files = os.listdir(DOWNLOAD_DIR)
            logging.debug(f"Arquivos encontrados: {files}")
            for file in files:
                if file.endswith(".csv"):
                    download_file = os.path.join(DOWNLOAD_DIR, file)

                    if os.path.getsize(download_file) > 0:
                        logging.info(f"Download concluído: {download_file}")
                        break
            if download_file:
                break
            time.sleep(polling_interval)
            elapsed += polling_interval

        if not download_file or not os.path.exists(download_file):
            logging.error("O download do CSV não foi concluído no tempo esperado.")
            return {"erro": "Falha ao baixar o CSV dentro do tempo esperado."}

        arquivo_final_path = os.path.join(DOWNLOAD_DIR, arquivo_historico)
        try:
            os.rename(download_file, arquivo_final_path)
            logging.info(f"Arquivo renomeado para '{arquivo_final_path}'.")
        except Exception as e:
            logging.error(f"Erro ao renomear o arquivo: {str(e)}")
            return {"erro": "Falha ao renomear o arquivo baixado."}

        for file in os.listdir(DOWNLOAD_DIR):
            if file == "export.csv":
                try:
                    os.remove(os.path.join(DOWNLOAD_DIR, file))
                    logging.info("Arquivo 'export.csv' removido com sucesso.")
                except Exception as e:
                    logging.error(f"Erro ao remover 'export.csv': {str(e)}")

        df = fetch_data_from_csv(arquivo_final_path)

        if df.empty:
            return {"erro": "Nenhum dado foi encontrado no CSV baixado."}

        arquivo_saida = f'historico/{arquivo_historico}'
        upload_to_s3_file(arquivo_final_path, bucket_name, arquivo_saida)

        return {"mensagem": "Dados processados e enviados com sucesso.", "dados": df.to_dict(orient='records')}

    except Exception as e:
        logging.error(f"Erro durante o processo: {str(e)}")
        return {"erro": "Ocorreu um erro durante o processamento."}

    finally:
        driver.quit()
        clear_download_dir()