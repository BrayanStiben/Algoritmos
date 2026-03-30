import os
import sys

# Añadir el directorio principal de financial_project a sys.path para ejecuciones directas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from config import MASTER_DIR

# Importaremos todos los algoritmos
from sorting.algorithms import (
    selection_sort, comb_sort, gnome_sort, binary_insertion_sort,
    quick_sort, heap_sort, tree_sort, tim_sort, pigeonhole_sort,
    bucket_sort, radix_sort, bitonic_sort
)


def run_benchmarks():
    """
    Carga master_dataset.csv en memoria como lista de diccionarios.
    Ejecuta un benchmark por algoritmo tomando muestras, para no forzar SelectionSort sobre 25.000 filas.
    Registra tiempos en "sorting_results.csv".
    Identifica y recopila los 15 top volumen ascendente por activo según regla 2.
    """
    print("Iniciando Benchmark Algorítmico...")
    master_file = os.path.join(MASTER_DIR, "master_dataset.csv")

    if not os.path.exists(master_file):
        print(f"Error: No se encuentra master dataset en {master_file}")
        return

    registros = []
    with open(master_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Casteamos para garantizar funcionamiento correcto en algoritmos
            row['open'] = float(row['open'])
            row['close'] = float(row['close'])
            row['high'] = float(row['high'])
            row['low'] = float(row['low'])
            row['volumen'] = int(row['volumen'])
            registros.append(row)

    # RESTRICCIÓN PDF: "Ordenar de manera ascendente cada uno de los registros a partir del archivo unificado"
    # JUSTIFICACIÓN: Se procesa el 100% de la carga del Master Dataset sin reducciones de muestra
    # para cumplir a cabalidad con la instrucción de volumen de datos, a costa de mayor tiempo
    # computacional en algoritmos O(n^2).
    muestra = registros

    print(
        f"Dataset Maestro detectado con {len(registros)} filas. Usaremos muestra de n={len(muestra)} registros continuos.")
    algoritmos = [
        ("Selection Sort", "O(n^2)", selection_sort),
        ("Comb Sort", "O(n log n) esp.", comb_sort),
        ("Gnome Sort", "O(n^2)", gnome_sort),
        ("Binary Insertion", "O(n^2)", binary_insertion_sort),
        ("Quick Sort", "O(n log n)", quick_sort),
        ("Heap Sort", "O(n log n)", heap_sort),
        ("Tree Sort", "O(n log n)", tree_sort),
        ("Tim Sort", "O(n log n)", tim_sort),
        ("Pigeonhole Sort", "O(n+N)", pigeonhole_sort),
        ("Bucket Sort", "O(n+k)", bucket_sort),
        ("Radix Sort", "O(d*(n+k))", radix_sort),
        ("Bitonic Sort", "O(n log^2 n)", bitonic_sort)
    ]

    # 1. Ejecutar Algoritmos
    # RESTRICCIÓN PDF: "no se acepta el uso de funciones de alto nivel que implementen directamente los
    # algoritmos solicitados... encapsuladas en una sola función."
    # JUSTIFICACIÓN: En lugar de usar `muestra.sort()` o `sorted(muestra)`, iteramos por implementaciones
    # clásicas programadas en este proyecto para medir iterativamente sus tiempos.
    resultados_benchmark = []
    for name, compl, func in algoritmos:
        print(f"Ejecutando {name}...")
        try:
            # Clon profundo naive sin tools externas
            array_clonado = []
            for m in muestra:
                array_clonado.append({
                    'fecha': m['fecha'], 'close': m['close'],
                    'open': m['open'], 'high': m['high'],
                    'low': m['low'], 'volumen': m['volumen'], 'ticker': m['ticker']
                })

            _, timer = func(array_clonado)

            resultados_benchmark.append({
                "algoritmo": name,
                "complejidad": compl,
                "tamaño": len(muestra),
                "tiempo_segundos": round(timer, 6)
            })
            print(f" -> {name}: {timer:.4f}s")
        except RecursionError:
            print(f" -> {name}: Error de Limite de Recursion Alcanzado.")
        except Exception as e:
            print(f" -> {name}: Error: {e}")

    # Guardar sorting_results.csv
    benchmark_file = os.path.join(MASTER_DIR, "sorting_results.csv")
    with open(benchmark_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algoritmo", "complejidad", "tamaño", "tiempo_segundos"])
        writer.writeheader()
        writer.writerows(resultados_benchmark)

    print(f"Resultados de Benchmark exportados a {benchmark_file}\n")

    # 2. Ordenar Top 15 ascendente por Volumen POR ACTIVO
    # Obtenemos dict de activos para agrupar
    activos = {}
    for r in registros:
        t = r['ticker']
        if t not in activos: activos[t] = []
        activos[t].append(r)

    print(" === PROCESAMIENTO TOP 15 DIAS ALTO VOLUMEN (Ordenado Ascendente) ===")

    activos_resumen = {}

    for ticker in activos:
        # Quicksort manual ordenando registros del ticker por volumen de menor a mayor
        def qv(arr):
            if len(arr) <= 1: return arr
            piv = arr[len(arr) // 2]
            izq, mid, der = [], [], []
            for x in arr:
                if x['volumen'] < piv['volumen']:
                    izq.append(x)
                elif x['volumen'] > piv['volumen']:
                    der.append(x)
                else:
                    mid.append(x)
            return qv(izq) + mid + qv(der)

        # Ordenar los registros del ticker actual por volumen ascendente
        activos[ticker] = qv(activos[ticker])

        # Tomar los últimos 15 elementos que corresponden a los de mayor volumen
        top_15_altos = activos[ticker][-15:]
        activos_resumen[ticker] = top_15_altos

        # Mostrar en consola el top 15 del ticker actual
        print(f" * TOP Volumen Ascendente - TICKER {ticker}:")
        for i, val in enumerate(top_15_altos):
            print(f"   {i + 1}. Fecha: {val['fecha']} -> Vol: {val['volumen']}")

    # Una vez procesados todos los tickers, guardar los resultados en un archivo CSV
    top15_file = os.path.join(MASTER_DIR, "top15_volumen.csv")
    with open(top15_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "posicion", "fecha", "volumen"])
        # Escribir la cabecera del CSV
        writer.writeheader()
        # Recorrer todos los tickers y escribir sus 15 registros
        for ticker in activos_resumen:
            for i, val in enumerate(activos_resumen[ticker]):
                writer.writerow({
                    "ticker": ticker,
                    "posicion": i + 1,
                    "fecha": val['fecha'],
                    "volumen": val['volumen']
                })

    print(f" -> Top 15 volumen exportado a {top15_file}\n")

    # Retornar el resumen para que main.py pueda usarlo en la visualización
    return activos_resumen


if __name__ == "__main__":
    run_benchmarks()