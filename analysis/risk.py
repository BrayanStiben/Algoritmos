import numpy as np
import math

def calculate_volatility(asset_rows):
    """
    Calcula volatilidad histórica anualizada (252 días de operación).
    Utiliza retornos logarítmicos sobre una lista plana de diccionarios.
    """
    n = len(asset_rows)
    if n < 2: return np.nan
    
    # Cálculo manual O(N) de retornos logarítmicos: ln(S_t / S_{t-1})
    log_returns = []
    for i in range(1, n):
        prev_close = asset_rows[i-1]['close']
        curr_close = asset_rows[i]['close']
        if prev_close and curr_close and prev_close > 0:
            log_returns.append(math.log(curr_close / prev_close))
            
    if not log_returns: return np.nan
            
    # Desviación estándar muestral O(N)
    mean_ret = sum(log_returns) / len(log_returns)
    variance = sum((r - mean_ret)**2 for r in log_returns) / (len(log_returns) - 1) if len(log_returns) > 1 else 0
    daily_vol = math.sqrt(variance)
    
    # Anualizar
    annual_vol = daily_vol * math.sqrt(252)
    
    return annual_vol

def classify_risk(annual_vol):
    """
    Clasifica el activo según su volatilidad histórica.
    - Conservador < 0.15 (15%)
    - Moderado entre 0.15 y 0.25 (25%)
    - Agresivo > 0.25
    """
    if np.isnan(annual_vol): 
        return "Indeterminado"
    elif annual_vol < 0.15:
        return "Conservador"
    elif annual_vol <= 0.25:
        return "Moderado"
    else:
        return "Agresivo"

def evaluate_portfolio_risk(master_rows):
    """
    Recorre el master dataset (lista de diccionarios) y devuelve el perfil
    de riesgo algorítmico de todos los activos, ordenados de menor a mayor riesgo.
    """
    risk_results = []
    
    # Extraer tickers únicos usando set O(N)
    tickers = set(row['ticker'] for row in master_rows)
    
    for ticker in tickers:
        # Filtrado O(N) por comprensión de listas
        asset_rows = [row for row in master_rows if row['ticker'] == ticker]
        asset_rows = sorted(asset_rows, key=lambda x: x['fecha'])
        
        volatility = calculate_volatility(asset_rows)
        classification = classify_risk(volatility)
        
        risk_results.append({
            'ticker': ticker,
            'volatilidad_anual': volatility,
            'perfil': classification
        })
        
    # Ordenar estrictamente de menor a mayor riesgo
    # Manejar np.nan comparándolo como infinito para que queden al final
    risk_results = sorted(risk_results, key=lambda x: x['volatilidad_anual'] if not np.isnan(x['volatilidad_anual']) else float('inf'))
    return risk_results
