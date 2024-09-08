import requests
import csv

csv_file_name = 'csv/weather_training_data.csv'

api_key = '62eb45a3a5fc6d10375784af8dd6530a'
# Coordenadas de São Sebastião, SP
lat = '-23.7656'
lon = '-45.4090'

url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric'
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['datetime', 'temp_celsius', 'feels_like_celsius', 'humidity_percent', 'wind_speed_mps',
                         'weather_description'])

        for forecast in data['list']:
            datetime = forecast['dt_txt']
            temp_celsius = forecast['main']['temp']
            feels_like_celsius = forecast['main']['feels_like']
            humidity_percent = forecast['main']['humidity']
            wind_speed_mps = forecast['wind']['speed']
            weather_description = forecast['weather'][0]['description']

            writer.writerow(
                [datetime, temp_celsius, feels_like_celsius, humidity_percent, wind_speed_mps, weather_description])
else:
    print(f'Erro na requisição: {response.status_code}')
    print(response.text)
