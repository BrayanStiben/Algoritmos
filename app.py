import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import csv
from analysis.risk import evaluate_portfolio_risk
from analysis.patterns import detect_consecutive_up_days, detect_reversal_v_pattern
from analysis.similarity import euclidean_distance, pearson_correlation, cosine_similarity, dtw_distance
from analysis.ml import knn_predict, monte_carlo_simulation, brute_force_portfolio
from config import MASTER_DIR
from visualization.pdf_generator import generate_technical_report

st.set_page_config(page_title="Fintech Analytics", layout="wide", page_icon="📈")

# Estilos CSS Futuristas / Cyberpunk (HUD Interface)
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Fondo principal y textos */
    .stApp { background-color: #031015; font-family: 'Share Tech Mono', monospace; }
    h1, h2, h3, h4 { color: #00f3ff !important; font-family: 'Share Tech Mono', monospace; text-shadow: 0 0 10px rgba(0, 243, 255, 0.8), 0 0 20px rgba(0, 243, 255, 0.4); text-transform: uppercase; letter-spacing: 2px;}
    p, span, div, label { color: #8ab4f8; font-family: 'Share Tech Mono', monospace; }
    
    /* Contenedores Inputs */
    .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
        border: 1px solid #00f3ff !important;
        background-color: rgba(0, 243, 255, 0.05) !important;
        box-shadow: inset 0 0 8px rgba(0, 243, 255, 0.3) !important;
        border-radius: 0px !important;
    }
    .stSlider > div { color: #00f3ff; }
    
    /* Métricas */
    [data-testid="stMetric"] {
        border: 1px solid #00f3ff;
        background: radial-gradient(circle, rgba(0,243,255,0.1) 0%, rgba(3,16,21,1) 100%);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.3) inset, 0 0 5px rgba(0, 243, 255, 0.5);
        border-radius: 0px; 
        padding: 15px;
        position: relative;
    }
    [data-testid="stMetric"]::before { content: ''; position: absolute; top: 0; left: 0; width: 10px; height: 10px; border-top: 2px solid #00f3ff; border-left: 2px solid #00f3ff;}
    [data-testid="stMetric"]::after { content: ''; position: absolute; bottom: 0; right: 0; width: 10px; height: 10px; border-bottom: 2px solid #00f3ff; border-right: 2px solid #00f3ff;}
    
    .metric-value, [data-testid="stMetricValue"] { font-size: 32px !important; color: #ff00ff !important; text-shadow: 0 0 15px rgba(255, 0, 255, 0.8); }
    [data-testid="stMetricLabel"] {color: #00f3ff !important; font-size: 14px; text-transform: uppercase;}
    
    /* Botones */
    div.stButton > button:first-child {
        background-color: transparent; border: 1px solid #00f3ff; color: #00f3ff;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.5) inset; border-radius: 0px; transition: all 0.3s ease; text-transform: uppercase; font-weight: bold; letter-spacing: 1px;
    }
    div.stButton > button:first-child:hover {
        background-color: #00f3ff; color: #031015; box-shadow: 0 0 25px rgba(0, 243, 255, 1); border-color: #00f3ff;
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] { background-color: rgba(3, 16, 21, 0.8); border-bottom: 3px solid #00f3ff; padding: 0px; gap: 5px; }
    .stTabs [data-baseweb="tab"] { color: #00f3ff; border: 1px solid transparent; border-bottom: none; padding: 10px 20px; transition: all 0.3s; text-transform: uppercase; font-weight: bold;}
    .stTabs [aria-selected="true"] {
        color: #031015 !important; background-color: #00f3ff !important; 
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.8); 
    }
    
    /* Tablas CSS */
    table { border-collapse: collapse; width: 100%; border: 1px solid #00f3ff; box-shadow: 0 0 15px rgba(0, 243, 255, 0.15);}
    th { background-color: rgba(0, 243, 255, 0.2); color: #00f3ff !important; text-transform: uppercase; padding: 10px; text-align: left; border-bottom: 2px solid #00f3ff;}
    td { padding: 8px; border-bottom: 1px solid rgba(0, 243, 255, 0.2); color: #8ab4f8; }
    tr:hover { background-color: rgba(0, 243, 255, 0.05); }
    </style>""", unsafe_allow_html=True
)
# Plantilla base HUD para plotly
hud_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Share Tech Mono", color="#00f3ff"),
    xaxis=dict(showgrid=True, gridcolor='rgba(0, 243, 255, 0.1)', zerolinecolor='rgba(0, 243, 255, 0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(0, 243, 255, 0.1)', zerolinecolor='rgba(0, 243, 255, 0.3)')
)

@st.cache_data
def load_master_data():
    master_file = os.path.join(MASTER_DIR, "master_dataset.csv")
    if not os.path.exists(master_file):
        st.warning("Ejecutando proceso ETL completo por primera vez...")
        from etl.extractor import extract_data
        from etl.transformer import transform_data
        from etl.loader import load_data
        extract_data()
        transform_data()
        load_data()
        
    master_rows = []
    with open(master_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['open'] = float(row['open']) if row['open'] else np.nan
            row['high'] = float(row['high']) if row['high'] else np.nan
            row['low'] = float(row['low']) if row['low'] else np.nan
            row['close'] = float(row['close']) if row['close'] else np.nan
            row['volumen'] = float(row['volumen']) if row['volumen'] else 0
            master_rows.append(row)
            
    return master_rows

st.title("Sistema Avanzado de Análisis Financiero 📊")
try:
    master_rows = load_master_data()
    # Extraer tickers únicos puramente iterando
    tickers = list(dict.fromkeys([row['ticker'] for row in master_rows]))
    
    # Generador de reportes en la barra lateral
    with st.sidebar:
        st.header("Generación de Reportes 📄")
        if st.button("Generar Reporte Técnico PDF", use_container_width=True):
            with st.spinner('Procesando PDF...'):
                risk_results = evaluate_portfolio_risk(master_rows)
                pdf_path = os.path.join(MASTER_DIR, "Reporte_Tecnico.pdf")
                generate_technical_report(risk_results, pdf_path)
                
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Descargar Archivo PDF generado", f, file_name="Reporte_Tecnico_Financiero.pdf", mime="application/pdf", use_container_width=True)
                    
except Exception as e:
    st.error(f"Falta el dataset maestro o hubo un error ETL: {e}")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["[VELAS & SMA]", "[CORRELACIÓN]", "[RIESGOS & PATRONES]", "[SIMILITUD]", "[IMPACTO COVID-19]", "[DATASET MAESTRO]", "[PREDICCIÓN IA & MONTE CARLO]"])

with tab1:
    st.header("Análisis de Tendencias (Candlestick & Medias Móviles)")
    col_a, col_b = st.columns([1, 4])
    with col_a:
        asset = st.selectbox("Seleccione Activo", tickers)
        sma_window = st.slider("Ventana de SMA", min_value=10, max_value=200, value=50, step=10)
        show_fibo = st.checkbox("Trazar Niveles Fibonacci Automáticos")
    with col_b:
        asset_rows = [r for r in master_rows if r['ticker'] == asset]
        asset_rows = sorted(asset_rows, key=lambda x: x['fecha'])
        
        # Calculo SMA algorítmico empírico (Sliding Window O(N))
        closes = [r['close'] for r in asset_rows]
        sma_manual = [np.nan] * len(closes)
        
        if len(closes) >= sma_window:
            # Ventana inicial (Ignorando NaNs si los hay)
            valid_closes = [c if not np.isnan(c) else 0 for c in closes]
            current_sum = sum(valid_closes[:sma_window])
            sma_manual[sma_window - 1] = current_sum / sma_window
            
            # Recorrido de ventana deslizante
            for i in range(sma_window, len(valid_closes)):
                current_sum = current_sum - valid_closes[i - sma_window] + valid_closes[i]
                sma_manual[i] = current_sum / sma_window
                
        # Graficar sin pandas
        fechas = [r['fecha'] for r in asset_rows]
        opens = [r['open'] for r in asset_rows]
        highs = [r['high'] for r in asset_rows]
        lows = [r['low'] for r in asset_rows]
        
        fig = go.Figure(data=[go.Candlestick(x=fechas,
                        open=opens, high=highs,
                        low=lows, close=closes, name='Precio Velas',
                        increasing_line_color='#00ffcc', decreasing_line_color='#ff00ff')])
        fig.add_trace(go.Scatter(x=fechas, y=sma_manual, line=dict(color='yellow', width=2), name=f'SMA {sma_window}'))
        
        # Retrocesos de Fibonacci Dinámicos O(N)
        if show_fibo and highs and lows:
            max_price = max([h for h in highs if h and not np.isnan(h)])
            min_price = min([l for l in lows if l and not np.isnan(l)])
            diff = max_price - min_price
            
            levels = {
                "0.0% (Mínimo)": min_price,
                "23.6%": max_price - diff * 0.764,
                "38.2%": max_price - diff * 0.618,
                "50.0%": max_price - diff * 0.5,
                "61.8%": max_price - diff * 0.382,
                "100.0% (Máximo)": max_price
            }
            colors_fibo = ["#33ff33", "#ffff00", "#ff9900", "#ff3399", "#ff0000", "#33ff33"]
            
            for (name, val), c in zip(levels.items(), colors_fibo):
                fig.add_hline(y=val, line_dash="solid" if name in ["0.0% (Mínimo)", "100.0% (Máximo)"] else "dot", 
                              line_color=c, line_width=1, opacity=0.7, 
                              annotation_text=f"Fibo {name}", annotation_position="right", annotation_font_color=c)

        fig.update_layout(title=f"Evolución de {asset}", height=600, yaxis_title="Precio Cierre ($)", **hud_layout)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Matriz de Correlación del Portafolio")
    st.markdown("Calcula las relaciones cruzadas entre los cierres de todos los activos, revelando posibles diversificaciones de riesgo.")
    
    # 1. Agrupar series por ticker dictado desde lists nativas
    ticker_series = {t: [] for t in tickers}
    for row in master_rows:
        ticker_series[row['ticker']].append(row['close'])
        
    n_tickers = len(tickers)
    corr_matrix_np = np.zeros((n_tickers, n_tickers))
    
    for i in range(n_tickers):
        for j in range(n_tickers):
            if i == j:
                corr_matrix_np[i, j] = 1.0
            elif j > i:
                s_i = np.array(ticker_series[tickers[i]], dtype=float)
                s_j = np.array(ticker_series[tickers[j]], dtype=float)
                
                # Eliminamos los NaNs compartidos (fechas desalineadas) usando bitwise masking rápido
                valid_mask = ~np.isnan(s_i) & ~np.isnan(s_j)
                val_i = s_i[valid_mask]
                val_j = s_j[valid_mask]
                
                # Usamos nuestra propia función matemática pura de Similitud Pearson O(N)
                corr, _ = pearson_correlation(val_i.tolist(), val_j.tolist())
                
                # Espejado simétrico de matriz relacional
                corr_matrix_np[i, j] = corr
                corr_matrix_np[j, i] = corr
                
    # Heatmap Futurista Cyberpunk Paleta
    fig_corr = px.imshow(corr_matrix_np, text_auto=".2f", aspect="auto", x=tickers, y=tickers, color_continuous_scale='electric', 
                         title="Mapa de Calor de Correlaciones de Pearson (Algoritmo Explícito)")
    fig_corr.update_layout(height=800, **hud_layout)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header("Clasificación de Riesgos y Algoritmos de Detección de Patrones")
    st.markdown("### 1. Perfil de Riesgo (Volatilidad Histórica Anualizada)")
    
    risk_results = evaluate_portfolio_risk(master_rows)
    # Mostrar riesgo manualmente creando tabla HTML (Sin Pandas df.style)
    st.markdown("""
        <style>
        .risk-conservador { background-color: rgba(46, 125, 50, 0.4); color: #00f3ff; border: 1px solid #2e7d32; padding: 4px; display:inline-block; min-width:80px; text-align:center;}
        .risk-moderado { background-color: rgba(239, 108, 0, 0.4); color: #ffd700; border: 1px solid #ef6c00; padding: 4px; display:inline-block; min-width:80px; text-align:center;}
        .risk-agresivo { background-color: rgba(198, 40, 40, 0.4); color: #ff00ff; border: 1px solid #c62828; padding: 4px; display:inline-block; min-width:80px; text-align:center;}
        </style>
    """, unsafe_allow_html=True)
    
    table_html = "<table><tr><th>Ticker</th><th>Volatilidad Anual</th><th>Perfil</th></tr>"
    for r in risk_results:
        pct = f"{r['volatilidad_anual']*100:.2f}%" if not np.isnan(r['volatilidad_anual']) else "N/A"
        cls = f"risk-{str(r['perfil']).lower()}"
        table_html += f"<tr><td>{r['ticker']}</td><td>{pct}</td><td><span class='{cls}'>{r['perfil']}</span></td></tr>"
    table_html += "</table><br>"
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown("### 2. Detección de Patrones mediante Sliding Window")
    patron_asset = st.selectbox("Evaluar Patrones en:", tickers, key="patron_select")
    
    asset_rows = [r for r in master_rows if r['ticker'] == patron_asset]
    asset_closes = [r['close'] for r in asset_rows]
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric(f"Días Consecutivos (3) al Alza", detect_consecutive_up_days(asset_closes, 3))
    col_p2.metric(f"Patrón de Reversión en V (Morning Star)", detect_reversal_v_pattern(asset_rows))
    st.info("La reversión en V ocurre cuando un activo sufre 3 días consecutivos de caídas (cierres por debajo de la apertura) seguido de un día alcista fuerte (cierre superior a la apertura anterior).")
    
    st.markdown("---")
    st.markdown("### 3. Optimizador de Portafolio O(N) Fuerza Bruta (Sharpe Ratio)")
    st.markdown("Sistema Markowitz sin Scipy u Optimizadores encapsulados. Encuentra los pesos calculando 1000 iteraciones aleatorias masivas de combinaciones cruzando Matrices de Covarianza O(N).")
    
    port_tickers = st.multiselect("Selecciona HASTA 4 activos para armar tu portafolio:", tickers, default=tickers[:3] if len(tickers)>2 else tickers)
    if st.button("Buscar Óptimo mediante IA Monte Carlo", type="primary"):
        if 2 <= len(port_tickers) <= 4:
            with st.spinner("Computando Covarianzas manuales N x N y lanzando caminos Markovianos..."):
                series_dict = {t: [r['close'] for r in master_rows if r['ticker']==t and r['close']] for t in port_tickers}
                opt_res = brute_force_portfolio(series_dict, num_portfolios=2000)
                
                if opt_res:
                    st.success(f"💎 **Optimizador de Markowitz Finalizado.** Mejor Sharpe Ratio Encontrado: {opt_res['sharpe']:.2f}")
                    
                    # Mostrar pesos sugeridos
                    p_col1, p_col2 = st.columns(2)
                    peso_str = ""
                    for tick, weight in opt_res['weights'].items():
                        peso_str += f"- **{tick}:** {weight*100:.1f}%\n"
                        
                    p_col1.info(f"**Distribución Ideal de Inversión (Pesos):**\n" + peso_str)
                    
                    p_col2.metric("Retorno Histórico Esperado", f"{opt_res['return']*100:.2f}% (Anual)")
                    p_col2.metric("Nivel de Riesgo (Volatilidad Combinada)", f"{opt_res['volatility']*100:.2f}%")
        else:
            st.warning("Para optimizar cruzado necesitas seleccionar entre 2 y 4 activos.")

with tab4:
    st.header("Laboratorio Matemático de Similitud Series de Tiempo")
    st.markdown("Compara dos activos para determinar la alineación espacial o temporal usando sus rendimientos o precios normalizados.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    s1_ticker = col_s1.selectbox("Activo 1", tickers, index=0)
    s2_ticker = col_s2.selectbox("Activo 2", tickers, index=1 if len(tickers)>1 else 0)
    algoritmo = col_s3.selectbox("Algoritmo", ["Distancia Euclidiana", "Correlación de Pearson", "Similitud Coseno", "DTW (Dynamic Time Warping)"])
    
    # Obtener arrays numéricos nativos
    arr1 = np.array([r['close'] for r in master_rows if r['ticker'] == s1_ticker])
    arr2 = np.array([r['close'] for r in master_rows if r['ticker'] == s2_ticker])
    
    # Limitar para que tengan el mismo tamaño forzosamente y normalizar (MinMax estándar para comparar forma libre de escala)
    min_len = min(len(arr1), len(arr2))
    a1 = arr1[-min_len:]
    a2 = arr2[-min_len:]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        a1_norm = (a1 - np.nanmean(a1)) / np.nanstd(a1)
        a2_norm = (a2 - np.nanmean(a2)) / np.nanstd(a2)
        
    a1_norm = np.nan_to_num(a1_norm)
    a2_norm = np.nan_to_num(a2_norm)
    
    if st.button("Calcular Similitud", type="primary"):
        val, req_complexy = 0, ""
        math_formula, algo_desc = "", ""
        
        if algoritmo == "Distancia Euclidiana":
            val, req_complexy = euclidean_distance(a1_norm.tolist(), a2_norm.tolist())
            math_formula = r"$$ d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$"
            algo_desc = "Itera sobre ambos vectores tramo a tramo $\mathcal{O}(N)$, acumulando el cuadrado de las diferencias de las observaciones normalizadas y aplicando la raíz cuadrada al sumatorio total. Útil para medir cercanía absoluta."
        elif algoritmo == "Correlación de Pearson":
            val, req_complexy = pearson_correlation(a1_norm.tolist(), a2_norm.tolist())
            math_formula = r"$$ r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}} $$"
            algo_desc = "Mide la relación lineal paralela de los rendimientos. Calcula las co-variaciones cruzadas restando las medias aritméticas y dividiendo por sus desviaciones cruzadas iterando secuencialmente $\mathcal{O}(N)$."
        elif algoritmo == "Similitud Coseno":
            val, req_complexy = cosine_similarity(a1_norm.tolist(), a2_norm.tolist())
            math_formula = r"$$ \text{cos}(\theta) = \frac{\sum x_i y_i}{\sqrt{\sum x_i^2} \sqrt{\sum y_i^2}} $$"
            algo_desc = "Evalúa la direccionalidad pura ignorando la magnitud calculando el producto punto de los vectores y normalizándolo $\mathcal{O}(N)$. Perfecto para evaluar momentos direccionales paralelos."
        elif algoritmo == "DTW (Dynamic Time Warping)":
            st.warning("Calculando Costo DTW en los últimos 250 días para evitar saturación de memoria O(N²).")
            # DTW en matrix nativa requiere listas planas
            val, req_complexy = dtw_distance(a1_norm[-250:].tolist(), a2_norm[-250:].tolist())
            math_formula = r"$$ DTW(X, Y) = \min_W \sqrt{\sum_{k=1}^K w_k} \quad (\text{Camino de Costo Mínimo}) $$"
            algo_desc = "Resuelve mediante Programación Dinámica $\mathcal{O}(N^2)$ creando una matriz de distancias locales acopladas temporalmente permitiendo elasticidad (las curvas pueden estar desfasadas)."
            
        st.success(f"**Cálculo Completado**: Métrica de {algoritmo} obtenida.")
        
        # Bloque expansivo explicativo (Requisito 2 Universitario)
        with st.expander("📚 Explicación Matemática y Algorítmica", expanded=True):
            st.latex(math_formula)
            st.markdown(f"**Fundamento Algorítmico:** {algo_desc}")
            st.markdown(f"**Complejidad Big-O:** `{req_complexy}` calculada sobre las bases canónicas nativas de Python.")
            st.metric(f"Valor Resultante ({algoritmo})", f"{val:.4f}")
        
        # Graficamos la serie
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(y=a1_norm[-250:], name=s1_ticker))
        fig_sim.add_trace(go.Scatter(y=a2_norm[-250:], name=s2_ticker))
        fig_sim.update_layout(title="Comparación Gráfica de Forma (Z-Score) - Últimos 250 días", height=400, **hud_layout)
        st.plotly_chart(fig_sim, use_container_width=True)
        
with tab5:
    st.header("Análisis de Quiebre Dinámico: Efecto COVID-19 (2021+)")
    st.markdown("Analiza la bifurcación temporal aislando y correlacionando activos pre y post 2021 mediante puras `List Comprehensions` O(N).")
    
    selected_assets = st.multiselect("Seleccione hasta 4 activos a comparar", tickers, default=[tickers[0]] if len(tickers)>0 else [])
    if len(selected_assets) > 4:
        st.warning("Seleccione máximo 4 activos para mantener visible el panel HUD.")
    elif len(selected_assets) > 0:
        col_c1, col_c2 = st.columns(2)
        
        # Filtros de diccionarios puros
        for i, asset in enumerate(selected_assets[:4]):
            asset_rows = [r for r in master_rows if r['ticker'] == asset]
            asset_rows = sorted(asset_rows, key=lambda x: x['fecha'])
            
            # Quiebre en 2021-01-01
            pre_covid = [r for r in asset_rows if r['fecha'] < '2021-01-01']
            post_covid = [r for r in asset_rows if r['fecha'] >= '2021-01-01']
            
            # Rentabilidades y Puntos crudos
            closes_pre = [r['close'] for r in pre_covid if r['close'] is not None]
            closes_post = [r['close'] for r in post_covid if r['close'] is not None]
            
            ret_pre = (closes_pre[-1] / closes_pre[0] - 1)*100 if len(closes_pre)>1 and closes_pre[0] > 0 else 0
            ret_post = (closes_post[-1] / closes_post[0] - 1)*100 if len(closes_post)>1 and closes_post[0] > 0 else 0
            
            # Promedios
            avg_pre = sum(closes_pre)/len(closes_pre) if closes_pre else 0
            avg_post = sum(closes_post)/len(closes_post) if closes_post else 0
            
            target_col = col_c1 if i % 2 == 0 else col_c2
            with target_col:
                st.markdown(f"#### [ACTIVO: {asset}]")
                colA, colB = st.columns(2)
                colA.metric("RETORNO PRE-2021", f"{ret_pre:.1f}%", f"Avg: ${avg_pre:.2f}")
                colB.metric("RETORNO POST-2021", f"{ret_post:.1f}%", f"Avg: ${avg_post:.2f}")

        # Gráfico Temporal Bifurcado
        fig_covid = go.Figure()
        # Colores cian/neon holográficos 
        colors = ['#00e5ff', '#ff00ff', '#fdd835', '#00e676']
        
        for i, asset in enumerate(selected_assets[:4]):
            asset_rows = [r for r in master_rows if r['ticker'] == asset]
            fechas = [r['fecha'] for r in asset_rows]
            closes_norm = np.array([r['close'] if r['close'] else 0 for r in asset_rows])
            
            # Normalizar para visualización justa si hay varios
            with np.errstate(divide='ignore', invalid='ignore'):
                c_norm = (closes_norm - np.nanmean(closes_norm)) / np.nanstd(closes_norm)
            
            fig_covid.add_trace(go.Scatter(x=fechas, y=c_norm, name=asset, line=dict(color=colors[i % len(colors)], width=2)))
            
        # Marcador divisor del quiebre (Línea V)
        fig_covid.add_vline(x='2021-01-01', line_width=2, line_dash="dash", line_color="#ff5252")
        fig_covid.add_annotation(x='2021-01-01', y=0, text="QUICK_RECOVERY_POINT (2021)", showarrow=True, arrowhead=1, font=dict(color="#ff5252", size=14))
        
        fig_covid.update_layout(title="HUD COMPARADOR TIMELINE: PRE VS POST", height=500, **hud_layout)
        st.plotly_chart(fig_covid, use_container_width=True)

        # Inferencias Automáticas de Comportamiento 
        if len(selected_assets) >= 2:
            st.markdown("### 📝 CONCLUSIONES SISTÉMICAS ALGORÍTMICAS")
            # Extraemos series POST-COVID del primer par
            asset1 = selected_assets[0]
            asset2 = selected_assets[1]
            c1_post = [r['close'] for r in master_rows if r['ticker'] == asset1 and r['fecha'] >= '2021-01-01']
            c2_post = [r['close'] for r in master_rows if r['ticker'] == asset2 and r['fecha'] >= '2021-01-01']
            
            c1_pre = [r['close'] for r in master_rows if r['ticker'] == asset1 and r['fecha'] < '2021-01-01']
            c2_pre = [r['close'] for r in master_rows if r['ticker'] == asset2 and r['fecha'] < '2021-01-01']
            
            c1_post = [c if c else 0 for c in c1_post]
            c2_post = [c if c else 0 for c in c2_post]
            min_post = min(len(c1_post), len(c2_post))
            
            # Usamos nuestra def pearson_correlation
            corr_post, _ = pearson_correlation(c1_post[-min_post:], c2_post[-min_post:])
            
            c1_pre = [c if c else 0 for c in c1_pre]
            c2_pre = [c if c else 0 for c in c2_pre]
            min_pre = min(len(c1_pre), len(c2_pre))
            corr_pre, _ = pearson_correlation(c1_pre[-min_pre:], c2_pre[-min_pre:])
            
            # Generar texto
            def evaluar_corr(c):
                if c > 0.7: return "altamente positiva (se movieron en tándem)"
                if c < -0.7: return "fuertemente inversa (bienes refugio / divergencia)"
                if c > 0: return "ligeramente positiva"
                return "débil/inversa o neutral"

            st.info(f"👉 **Evolución Correlacional ({asset1} vs {asset2}):**\n"
                    f"Antes de 2021, la correlación estadística entre ambos instrumentos era **{corr_pre:.2f}** ({evaluar_corr(corr_pre)}). "
                    f"Para la ventana temporal a partir de 2021 en adelante (Recuperación / Post-COVID), su correlación es de **{corr_post:.2f}** ({evaluar_corr(corr_post)}). "
                    f"Esto indica estructuralmente de manera matemática que su comportamiento {'cambió drásticamente debido a dinámicas sectoriales asíncronas.' if abs(corr_post - corr_pre) > 0.5 else 'mantuvo un flujo relacional relativamente constante a pesar del evento disruptivo mundial.'}")

with tab6:
    st.header("Terminal de Datos Limpios (Master Dataset)")
    st.markdown("Inspecciona la matriz de registros consolidados del ETL. Todas las series han sido alineadas en su fecha (Outer Join algorítmico) y purgadas algorítmicamente sin dependencias de terceros.")
    
    # Selector de filtrado sin DataFrames
    show_dataset_ticker = st.selectbox("Filtrar Registros por Activo HUD:", ["[TODOS LOS TICKERS]"] + tickers)
    
    if show_dataset_ticker == "[TODOS LOS TICKERS]":
        filtered = master_rows[-1000:]
        st.caption("Visor HUD Estricto: Mostrando los últimos 1000 registros para evitar latencia HTML global.")
    else:
        filtered = [r for r in master_rows if r['ticker'] == show_dataset_ticker][-1000:]
        st.caption(f"Visor HUD Estricto: Mostrando los últimos 1000 registros estandarizados de {show_dataset_ticker}.")
        
    # Renderizamos en HTML puro para heredar estrictamente los degradados y bordes neón del HUD CSS
    table_html = "<table><tr><th>Fecha</th><th>Ticker</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volumen</th></tr>"
    for r in filtered:
        o = f"{r['open']:.2f}" if not np.isnan(r['open']) else "N/A"
        h = f"{r['high']:.2f}" if not np.isnan(r['high']) else "N/A"
        l = f"{r['low']:.2f}" if not np.isnan(r['low']) else "N/A"
        c = f"{r['close']:.2f}" if not np.isnan(r['close']) else "N/A"
        v = f"{r['volumen']:.0f}" if not np.isnan(r['volumen']) else "0"
        table_html += f"<tr><td>{r['fecha']}</td><td>{r['ticker']}</td><td>{o}</td><td>{h}</td><td>{l}</td><td>{c}</td><td>{v}</td></tr>"
    table_html += "</table>"
    # Contenedor con scroll adaptativo para el HUD HTML
    st.markdown(f"<div style='height: 600px; overflow-y: scroll; border: 1px solid #00f3ff; padding: 5px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.15) inset;'>{table_html}</div>", unsafe_allow_html=True)

with tab7:
    st.header("Módulo Extremo Avanzado: Inteligencia Predictiva O(N)")
    st.markdown("Cálculos probabilísticos realizados puramente mediante Iteraciones y Matemáticas de Listas bajo inferencia Random Walks Inyectadas (Scikit-Learn Simulation).")
    
    ml_asset = st.selectbox("Seleccionar Activo Objetivo para Predicción e Inferencia:", tickers, key="ml_select")
    asset_rows = [r for r in master_rows if r['ticker'] == ml_asset]
    asset_rows = sorted(asset_rows, key=lambda x: x['fecha'])
    
    colML1, colML2 = st.columns([1, 2])
    
    with colML1:
        st.subheader("1. K-Nearest Neighbors O(N)")
        st.write("Busca los 5 días históricos con una racha de retornos equivalente a los últimos 3 días (Distancia Euclidiana O(N)) y somete el comportamiento del día posterior a votación mayoritaria.")
        
        if st.button("Ejecutar Inferencias KNN"):
            with st.spinner("Escaneando el firmamento histórico..."):
                pred, prob = knn_predict(asset_rows, k=5, window_size=3)
                
                direction = "ALZA 🚀" if pred == 1 else "BAJA 📉"
                color = "#33ff33" if pred == 1 else "#ff3333"
                
                st.markdown(f"#### PREDICCIÓN CONSOLIDADA (MAÑANA): <span style='color:{color}'>{direction}</span>", unsafe_allow_html=True)
                st.metric("Confianza de IA Replicada (Mayoritaria):", f"{prob*100:.1f}%")

    with colML2:
        st.subheader("2. Simulador de Wall Street (Fuerza N-Monte Carlo)")
        st.write("Dispara 50 caminos log-normales probables para este activo a 30 días, fundamentados heurísticamente en su Volatilidad de serie.")
        
        if st.button("Lanzar Proyección Monte Carlo (30 Días)"):
            with st.spinner("Ejecutando sorteos Gaussianos y construyendo matrices predictivas..."):
                caminos_mc, min_mc, max_mc = monte_carlo_simulation(asset_rows, days_ahead=30, n_simulations=50)
                
                if caminos_mc:
                    fig_mc = go.Figure()
                    for c in caminos_mc:
                        # Dibujamos las líneas de proyecciones holográficamente (Cyan tenue)
                        fig_mc.add_trace(go.Scatter(y=c, mode='lines', line=dict(color='rgba(0, 243, 255, 0.15)', width=1), showlegend=False))
                        
                    fig_mc.update_layout(title=f"Proyecciones a futuro (50 Random Walks) - {ml_asset}", height=450, xaxis_title="Días a Futuro", yaxis_title="Precio Simulado", **hud_layout)
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Peor Escenario (Worst Case) a 30 días", f"${min_mc:.2f}")
                    mc2.metric("Mejor Escenario (Best Case) a 30 días", f"${max_mc:.2f}")
