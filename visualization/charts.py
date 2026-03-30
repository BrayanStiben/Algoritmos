import os
import csv
from config import MASTER_DIR, OUTPUT_DIR


def generate_bar_chart():
    """
    Lee sorting_results.csv, ordena con burbuja y genera HTML de barras horizontales.
    Complejidad Temporal: O(n^2) burbuja justificado por n=12 algoritmos.
    Complejidad Espacial: O(n) por la lista de registros en memoria.
    """
    print("Iniciando Módulo de Visualización (Charts)...")

    # Verificar existencia del CSV de benchmark
    res_path = os.path.join(MASTER_DIR, "sorting_results.csv")
    if not os.path.exists(res_path):
        print(f"Error: {res_path} no existe. Ejecuta el benchmark primero.")
        return

    # Leer resultados del benchmark
    records = []
    with open(res_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "algoritmo": row["algoritmo"],
                "tiempo": float(row["tiempo_segundos"])
            })

    # Ordenar ascendente por tiempo con burbuja (n=12, costo insignificante)
    for i in range(len(records)):
        for j in range(len(records) - 1):
            if records[j]["tiempo"] > records[j + 1]["tiempo"]:
                records[j], records[j + 1] = records[j + 1], records[j]

    # Tiempo máximo como referencia para escalar barras
    max_time = max(r["tiempo"] for r in records) if records else 1.0
    if max_time == 0:
        max_time = 0.001

    # HTML base con estilos oscuros, sin librerías gráficas externas
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Benchmark Algoritmos de Ordenamiento</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; }}
            .chart-container {{ max-width: 900px; margin: auto; background: #161b22; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            h1 {{ text-align: center; color: #58a6ff; }}
            .bar-row {{ display: flex; align-items: center; margin-bottom: 12px; }}
            .label {{ width: 140px; font-weight: bold; font-size: 14px; text-align: right; margin-right: 15px; }}
            .bar-wrapper {{ flex-grow: 1; background: #21262d; border-radius: 4px; overflow: hidden; height: 24px; position: relative; }}
            .bar {{ height: 100%; background: linear-gradient(90deg, #238636, #2ea043); transition: width 0.5s ease; }}
            .time-label {{ position: absolute; right: 10px; top: 0; line-height: 24px; font-size: 12px; font-weight: bold; color: #fff; text-shadow: 1px 1px 2px #000; }}
        </style>
    </head>
    <body>
        <div class="chart-container">
            <h1>Tiempos de Ejecución: 12 Algoritmos de Ordenamiento</h1>
    """

    # Barra por algoritmo con ancho proporcional al tiempo máximo
    for r in records:
        w_pct = (r["tiempo"] / max_time) * 100
        if w_pct < 0.5:  # Mínimo visible
            w_pct = 0.5

        html_content += f"""
            <div class="bar-row">
                <div class="label">{r['algoritmo']}</div>
                <div class="bar-wrapper">
                    <div class="bar" style="width: {w_pct}%;"></div>
                    <div class="time-label">{r['tiempo']:.5f} s</div>
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Guardar HTML en outputs/
    chart_file = os.path.join(OUTPUT_DIR, "sorting_chart.html")
    with open(chart_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f" -> Diagrama de Barras (Algoritmos) renderizado y guardado en {chart_file}\n")


def generate_bar_chart_sin_lentos():
    """
    Igual que generate_bar_chart() pero excluye los 3 más lentos.
    Permite comparar visualmente mejor los algoritmos eficientes.
    Complejidad Temporal: O(n^2) burbuja, n=12.
    Complejidad Espacial: O(n) por lista en memoria.
    """
    print("Iniciando Renderizado de Gráfica Sin Algoritmos Lentos...")

    # Verificar existencia del CSV de benchmark
    res_path = os.path.join(MASTER_DIR, "sorting_results.csv")
    if not os.path.exists(res_path):
        print(f"Error: {res_path} no existe.")
        return

    # Leer resultados del benchmark
    records = []
    with open(res_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "algoritmo": row["algoritmo"],
                "tiempo": float(row["tiempo_segundos"])
            })

    # Ordenar ascendente con burbuja
    for i in range(len(records)):
        for j in range(len(records) - 1):
            if records[j]["tiempo"] > records[j + 1]["tiempo"]:
                records[j], records[j + 1] = records[j + 1], records[j]

    # Separar los 3 más lentos para nota al pie y excluirlos del gráfico
    excluidos = records[-3:]
    records = records[:-3]

    # Tiempo máximo del subconjunto eficiente como referencia
    max_time = max(r["tiempo"] for r in records) if records else 1.0
    if max_time == 0:
        max_time = 0.001

    # HTML en azul para diferenciar visualmente de la gráfica completa (verde)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Benchmark - Algoritmos Eficientes</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; }}
            .chart-container {{ max-width: 900px; margin: auto; background: #161b22; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            h1 {{ text-align: center; color: #58a6ff; }}
            .bar-row {{ display: flex; align-items: center; margin-bottom: 12px; }}
            .label {{ width: 140px; font-weight: bold; font-size: 14px; text-align: right; margin-right: 15px; }}
            .bar-wrapper {{ flex-grow: 1; background: #21262d; border-radius: 4px; overflow: hidden; height: 24px; position: relative; }}
            .bar {{ height: 100%; background: linear-gradient(90deg, #1f6feb, #58a6ff); transition: width 0.5s ease; }}
            .time-label {{ position: absolute; right: 10px; top: 0; line-height: 24px; font-size: 12px; font-weight: bold; color: #fff; text-shadow: 1px 1px 2px #000; }}
            .nota {{ max-width: 900px; margin: 20px auto 0; font-size: 12px; color: #8b949e; border-top: 1px solid #30363d; padding-top: 12px; }}
        </style>
    </head>
    <body>
        <div class="chart-container">
            <h1>Tiempos de Ejecución: 9 Algoritmos Eficientes</h1>
    """

    # Barra por algoritmo del subconjunto eficiente
    for r in records:
        w_pct = (r["tiempo"] / max_time) * 100
        if w_pct < 0.5:  # Mínimo visible
            w_pct = 0.5

        html_content += f"""
            <div class="bar-row">
                <div class="label">{r['algoritmo']}</div>
                <div class="bar-wrapper">
                    <div class="bar" style="width: {w_pct}%;"></div>
                    <div class="time-label">{r['tiempo']:.5f} s</div>
                </div>
            </div>
        """

    # Nota al pie con los 3 excluidos y sus tiempos reales
    excluidos_str = " | ".join([f"{e['algoritmo']}: {e['tiempo']:.5f}s" for e in excluidos])
    html_content += f"""
        </div>
        <div class="nota">
            * Algoritmos excluidos por escala (O(n²)): {excluidos_str}
        </div>
    </body>
    </html>
    """

    # Guardar HTML en outputs/ con nombre distinto al gráfico completo
    chart_file = os.path.join(OUTPUT_DIR, "sorting_chart_eficientes.html")
    with open(chart_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f" -> Gráfica algoritmos eficientes guardada en {chart_file}\n")


def generate_volume_chart(activos_top_volumen):
    """
    Recibe el top 15 días por volumen de benchmark.py y genera HTML de barras.
    Se limita a 5 activos para no saturar la visualización.
    Complejidad Temporal: O(A * 15) donde A = activos mostrados (máx 5).
    Complejidad Espacial: O(1) construcción iterativa sin acumular datos.
    """
    print("Iniciando Renderizado de Gráfica de Volumen...")

    # HTML base con estilos oscuros
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Top 15 Días Mayor Volumen por Activo</title>
        <style>
            body { font-family: Arial, sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; }
            .chart-container { max-width: 900px; margin: auto; background: #161b22; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 30px; }
            h1 { text-align: center; color: #d2a8ff; }
            h2 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 5px; }
            .bar-row { display: flex; align-items: center; margin-bottom: 8px; }
            .label { width: 120px; font-weight: bold; font-size: 12px; text-align: right; margin-right: 15px; color: #8b949e; }
            .bar-wrapper { flex-grow: 1; background: #21262d; border-radius: 4px; overflow: hidden; height: 20px; position: relative; }
            .bar { height: 100%; background: linear-gradient(90deg, #8957e5, #d2a8ff); transition: width 0.5s ease; }
            .time-label { position: absolute; right: 10px; top: 0; line-height: 20px; font-size: 11px; font-weight: bold; color: #fff; text-shadow: 1px 1px 2px #000; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center; padding-bottom: 20px;">Top 15 Días Mayor Volumen por Activo (Orden Ascendente)</h1>
    """

    # Recorrer activos limitando a 5 para no saturar la vista
    count = 0
    for ticker, datos in activos_top_volumen.items():
        if count >= 5:
            break

        # Volumen máximo del activo para escalar barras proporcionalmente
        max_vol = max(d["volumen"] for d in datos) if datos else 1
        if max_vol == 0:
            max_vol = 1

        # Sección por activo
        html_content += f'<div class="chart-container">\n<h2>{ticker}</h2>\n'

        # Barra por cada uno de los 15 días
        for d in datos:
            w_pct = (d["volumen"] / max_vol) * 100
            if w_pct < 0.5:  # Mínimo visible
                w_pct = 0.5

            html_content += f"""
                <div class="bar-row">
                    <div class="label">{d['fecha'][:10]}</div>
                    <div class="bar-wrapper">
                        <div class="bar" style="width: {w_pct}%;"></div>
                        <div class="time-label">{d['volumen']:,}</div>
                    </div>
                </div>
            """

        html_content += '</div>\n'
        count += 1

    html_content += """
    </body>
    </html>
    """

    # Guardar HTML en outputs/
    chart_file = os.path.join(OUTPUT_DIR, "volume_chart.html")
    with open(chart_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f" -> Diagrama de Barras (Volumen) renderizado y guardado en {chart_file}\n")


if __name__ == "__main__":
    generate_bar_chart()