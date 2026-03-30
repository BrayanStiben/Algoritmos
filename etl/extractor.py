import os
import json
import time
import requests
from config import RAW_DIR, TICKERS

def extract_data():

    print("Iniciando Extractor ETL.....")

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept': '*/*'}

    for ticker in TICKERS:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=5y&interval=1d"
        try:
            print(f"Descargando datos raw de: {ticker}")


            #Se utiliza la librería requests estándar para realizar la petición HTTP
            # directa construyendo la URL manualmente.
            response = requests.get(url, headers=headers, timeout=10)

            # Si el ticker no existe o falla, YF suele devolver 404
            if response.status_code == 200:
                data = response.json()

                # Verificamos si realmente trajo resultados válidos y con datos históricos
                result = data.get("chart", {}).get("result")
                if result and result[0].get("indicators", {}).get("adjclose", [{}])[0]:
                    file_path = os.path.join(RAW_DIR, f"{ticker}.json")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    print(f" -> Guardado exitoso: {file_path}")
                else:
                    print(f" -> Datos vacíos para: {ticker} (Tickers inválido o sin histórico)")
            else:
                print(f" -> Fallo HTTP {response.status_code} al buscar: {ticker}")

        except requests.exceptions.RequestException as e:
            print(f" -> Error de conexión con {ticker}: {e}")

            # Delay estricto según requerimientos de "políticas de scraping justas/respeto"
        time.sleep(2)

        print("Extracción completada.\n")

    if __name__ == "__main__":
        extract_data()


