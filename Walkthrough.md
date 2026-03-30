# Walkthrough del Sistema de Análisis Financiero

Se ha completado exitosamente la refactorización arquitectónica del proyecto, integrando módulos algorítmicos complejos y un pipeline ETL moderno, logrando desplegar todo en un Dashboard Interactivo de Streamlit. **Esta iteración cumple estrictamente con el requerimiento de no utilizar librerías analíticas de alto nivel como Pandas o SciPy**, basándose exclusivamente en librerías estándar y estructuras de datos nativas (listas y diccionarios).

## 🛠️ Cambios Realizados y Características

1. **Gestión de Entornos**: 
   - Modificamos el `requirements.txt` para eliminar `pandas` y `scipy`.
   - Modificación al punto de entrada `main.py` para inicializar el servidor web en un solo comando sin necesidad de escribir en consola comandos complejos.

2. **Refactorización ETL Algorítmica (Python Puro)**:
   - `transformer.py` ya no usa DataFrames. Implementamos iteradores y comprensiones de listas (`List Comprehensions`) para procesar el JSON con una complejidad determinista O(N), usando un *Carry-Forward* explícito.
   - `loader.py` fue reimplementado mediante diccionarios para cruzar las fechas de los activos (BVC de Colombia vs NYSE Globales) garantizando integridad temporal mediante un outer-join manual O(N), limpiando diferencias entre feriados y festividades regionales.
   - Todo exporta usando el módulo estándar `csv`.

3. **Módulo de Algoritmos Financieros Avanzados (`analysis/` & `app.py`)**:
   - **Similitud:** Se añadieron 4 modelos calculados nativamente: *Euclidiana*, *Pearson*, *Coseno* y *Dynamic Time Warping*. Todo funciona con O(N) basándose puramente en Python.
   - **Patrones:** Motor de *Sliding Windows* para contar los días consecutivos alcistas y detector matemático para buscar formaciones en "V" (derivado de Morning Star) escaneando las aperturas vs. cierres de 4 posiciones adyacentes a nivel de lista.
   - **Riesgo y Perfilado:** Generador de Portafolio ordenado evaluando Volatilidad Historizada Anualizada ($RetornosLog \times \sqrt{252}$). Calculado usando el módulo `math`. Perfila Activos como Conservador (<15%), Moderado o Agresivo (>25%).
   - **⚖️ Optimizador de Portafolio (Brute-Force Matemático):** Ubicado en la Pestaña 3, permite al usuario elegir hasta 4 acciones. Lanza 2000 iteraciones calculando la Matriz de Covarianzas manual y los retornos esperados, encontrando la combinación de "Pesos" óptimos que maximice el *Sharpe Ratio*, sin usar Scipy ni Pandas.
   - **📉 Retrocesos de Fibonacci (Análisis Técnico):** Implementado en la Pestaña 1 sobre el gráfico de velas. A través de un interruptor, busca de forma iterativa y directa el Máximo y Mínimo del rango renderizado, trazando líneas horizontales precisas de soportes y resistencias algorítmicos en los puntos exactos: 23.6%, 38.2%, 50.0% y 61.8%.
   - **📈 Predictor Trivial por KNN:** Desarrollado nativamente en `analysis/ml.py` en Python puro. Encuentra la cercanía de Z-Scores de rendimiento de los últimos 3 días (vectores en base Euclidiana) contra toda la historia del instrumento. Toma los "5 vecinos más cercanos" y realiza votación por mayoría simple para arrojar una predicción en la Pestaña 7.

4. **Despliegue de Interfaz (`app.py` & UI)**:
   - Streamlit y Plotly ahora consumen una matriz en memoria mediante iteradores basados en `csv.DictReader` anulando los DataFrames por completo.
   - Las matrices de correlación globales (cruce de precios Ticker x Ticker) operan reestructurando arreglos O(C^2) de base empírica computacional y son enviadas nativamente a `px.imshow`.
   - **HUD Futurista**: El CSS ha sido remodelado fuertemente imitando los tableros cibernéticos de control. Usa fuentes "Share Tech Mono", bordes `cyan/neon`, contornos y botones reactivos.
   - **Impacto COVID-19 (Pre/Post 2021)**: Un 5to módulo permite comparar múltiples activos en ventanas divisorias. Filtra O(N) las muestras temporales para dictaminar correlación y retornos aislando el evento de pandemia.
   - **Terminal de Datos Limpios**: Incorporación del 6to módulo (`tab6`) llamado "[DATASET MAESTRO]" que permite inspeccionar la totalidad de los ~36,000 registros del ETL alineados.
   - Inyección de Componente de Exportación Técnica conectada a la librería `fpdf`.

## ☑️ Verificación de Resultados

El proyecto ahora respeta todos los requerimientos algorítmicos sin Pandas (Puro Python) incluyendo Optimizadores, Simuladores Monte Carlo, KNN Predictors y Análisis Técnico como Fibonacci. Se puede lanzar mediante:
`uv run main.py`
