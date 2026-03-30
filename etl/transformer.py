import os
import json
import csv
from datetime import datetime
from config import RAW_DIR, CLEAN_DIR

def transform_data():
    print("Iniciando Transformador ETL (Puro Python O(N))...")
    archivos = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]

    for archivo in archivos:
        ticker = archivo.replace(".json", "")
        with open(os.path.join(RAW_DIR, archivo), "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]
            
            opens = quote.get('open', [None]*len(timestamps))
            highs = quote.get('high', [None]*len(timestamps))
            lows = quote.get('low', [None]*len(timestamps))
            closes = quote.get('close', [None]*len(timestamps))
            volumes = quote.get('volume', [0]*len(timestamps))
            
            # Construir lista de diccionarios (Dataframe casero)
            rows = []
            for i in range(len(timestamps)):
                # Convertir timestamp Unix a YYYY-MM-DD
                dt = datetime.utcfromtimestamp(timestamps[i]).strftime('%Y-%m-%d')
                rows.append({
                    'fecha': dt,
                    'ticker': ticker,
                    'open': opens[i],
                    'high': highs[i],
                    'low': lows[i],
                    'close': closes[i],
                    'volumen': volumes[i] if volumes[i] is not None else 0
                })
        except (KeyError, IndexError, TypeError):
            print(f" -> JSON con estructura anómala, saltando: {ticker}")
            continue

        # Ordenar por fecha por seguridad
        rows = sorted(rows, key=lambda x: x['fecha'])
        
        # Evaluar Nulos y Forward-Fill (Carry-Forward)
        last_valid = {'open': None, 'high': None, 'low': None, 'close': None}
        for row in rows:
            for col in ['open', 'high', 'low', 'close']:
                if row[col] is not None:
                    last_valid[col] = row[col]
                    row[col] = round(row[col], 4)
                else:
                    row[col] = last_valid[col] # ffill
                    
        # Backward-Fill (para nulos al inicio que el ffill no agarró)
        first_valid = {'open': None, 'high': None, 'low': None, 'close': None}
        for row in reversed(rows):
            for col in ['open', 'high', 'low', 'close']:
                if row[col] is not None:
                    first_valid[col] = row[col]
                elif first_valid[col] is not None:
                    row[col] = first_valid[col] # bfill

        # Exportar a Clean CSV
        csv_path = os.path.join(CLEAN_DIR, f"{ticker}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=['fecha', 'ticker', 'open', 'high', 'low', 'close', 'volumen'])
            writer.writeheader()
            writer.writerows(rows)

        print(f" -> Transformado: {ticker}.csv ({len(rows)} filas)")

    print("Transformación completada.\n")


if __name__ == "__main__":
    transform_data()