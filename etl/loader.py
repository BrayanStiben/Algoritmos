import os
import csv
from config import CLEAN_DIR, MASTER_DIR

def load_data():
    """
    Consolida todos los CSV individuales de la carpeta clean en un único "master_dataset.csv".
    Se ajustan las fechas (desalineamiento de calendarios bursátiles) utilizando 
    diccionarios anidados para un Outer Join explícito.
    """
    print("Iniciando Loader ETL (Módulo de Carga Puro Python)...")
    archivos = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.csv')]

    if not archivos:
        print("No hay archivos CSV en la carpeta clean para cargar.")
        return None

    # Leer todos los datos a diccionarios
    all_dates = set()
    tickers_data = {} # { 'AAPL': { '2023-01-01': {...row...} }, ... }
    tickers_list = []

    for archivo in archivos:
        ticker = archivo.replace(".csv", "")
        tickers_list.append(ticker)
        tickers_data[ticker] = {}
        
        with open(os.path.join(CLEAN_DIR, archivo), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fecha = row['fecha']
                all_dates.add(fecha)
                # Convertir numéricos
                row['open'] = float(row['open']) if row['open'] else None
                row['high'] = float(row['high']) if row['high'] else None
                row['low'] = float(row['low']) if row['low'] else None
                row['close'] = float(row['close']) if row['close'] else None
                row['volumen'] = int(float(row['volumen'])) if row['volumen'] else 0
                tickers_data[ticker][fecha] = row

    sorted_dates = sorted(list(all_dates))
    
    # Outer Join + Forward fill (LOCF) & Backward fill para sanear el calendario asíncrono
    master_rows = []
    
    for ticker in tickers_list:
        last_valid = {'open': None, 'high': None, 'low': None, 'close': None, 'volumen': 0}
        ticker_rows = []
        
        for fecha in sorted_dates:
            if fecha in tickers_data[ticker]:
                row = tickers_data[ticker][fecha]
                last_valid = row.copy()
                ticker_rows.append(row)
            else:
                # Si falta la fecha por festividad en este país, copiamos el último valor válido (ffill explícito)
                new_row = last_valid.copy()
                new_row['fecha'] = fecha
                new_row['ticker'] = ticker
                ticker_rows.append(new_row)
                
        # Backward fill simple por si las primeras fechas estaban vacías
        first_valid = {'open': None, 'high': None, 'low': None, 'close': None, 'volumen': 0}
        for row in reversed(ticker_rows):
            if row['open'] is not None:
                first_valid = row.copy()
            elif first_valid['open'] is not None:
                row['open'] = first_valid['open']
                row['high'] = first_valid['high']
                row['low'] = first_valid['low']
                row['close'] = first_valid['close']
                row['volumen'] = first_valid['volumen']
                
        master_rows.extend(ticker_rows)

    # Ordenar máster dataset por fecha y luego por ticker
    master_rows = sorted(master_rows, key=lambda x: (x['fecha'], x['ticker']))

    # Escribir CSV Unificado
    master_file_path = os.path.join(MASTER_DIR, "master_dataset.csv")
    fieldnames = ['fecha', 'ticker', 'open', 'high', 'low', 'close', 'volumen']
    
    with open(master_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master_rows)

    print(f" -> Carga y alineación de calendarios (Sin Pandas) finalizada en {master_file_path}")
    print(f" -> Total registros en Master Dataset: {len(master_rows)}\n")
    
    return master_file_path

if __name__ == "__main__":
    load_data()
