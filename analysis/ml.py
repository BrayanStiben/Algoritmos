import math
import random

def get_log_returns(closes):
    """Calcula retornos logarítmicos secuenciales O(N)."""
    returns = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0 and closes[i] > 0:
            returns.append(math.log(closes[i]/closes[i-1]))
        else:
            returns.append(0.0)
    return returns

def euclidean_distance_native(vec1, vec2):
    """Distancia Euclidiana O(N) para listas flat."""
    return math.sqrt(sum((a - b)**2 for a, b in zip(vec1, vec2)))

def knn_predict(asset_rows, k=5, window_size=3):
    """
    Predice si el cierre del día siguiente será alcista (+1) o bajista (-1)
    basado en los retornos más cercanos históricamente usando K-Nearest Neighbors.
    """
    closes = [r['close'] for r in asset_rows if r['close']]
    if len(closes) < window_size + k + 2:
        return 0, 0.0 # No hay suficientes datos para entrenamiento y predicción
        
    retornos = get_log_returns(closes)
    
    # Normalizamos el set de retornos para mejorar el predictor euclidiano (Z-Score manual)
    mu = sum(retornos) / len(retornos)
    variance = sum((x - mu)**2 for x in retornos) / len(retornos)
    sigma = math.sqrt(variance) if variance > 0 else 1.0
    ret_norm = [(x - mu) / sigma for x in retornos]
    
    # Extraemos Features X y Targets Y
    # X_i = [ret[i-2], ret[i-1], ret[i]]
    # Y_i = +1 si ret[i+1] > 0 else -1
    
    X = []
    y = []
    
    for i in range(window_size - 1, len(ret_norm) - 1):
        feature_vector = ret_norm[i - window_size + 1 : i + 1]
        target = 1 if ret_norm[i + 1] > 0 else -1
        X.append(feature_vector)
        y.append(target)
        
    # El vector objetivo "Hoy" para predecir "Mañana" (No tiene variable Target conocida aún)
    vector_hoy = ret_norm[-window_size:]
    
    # Computar distancias O(N)
    distancias = []
    for i in range(len(X)):
        dist = euclidean_distance_native(vector_hoy, X[i])
        distancias.append((dist, y[i]))
        
    # Ordenar algoritmicamente por las más cortas (Nearest Neighbors)
    distancias.sort(key=lambda x: x[0])
    vecinos_cercanos = distancias[:k]
    
    # Votación (Majority Vote)
    votos_positivos = sum(1 for dist, target in vecinos_cercanos if target == 1)
    votos_negativos = k - votos_positivos
    
    prediccion = 1 if votos_positivos > votos_negativos else -1
    confianza = max(votos_positivos, votos_negativos) / k
    
    return prediccion, confianza

def monte_carlo_simulation(asset_rows, days_ahead=30, n_simulations=50):
    """
    Proyecta 50 caminos aleatorios (Random Walks) usando distribución log-normal.
    Retorna O(N): Lista de Caminos, precio mínimo proyectado, y máximo proyectado.
    """
    closes = [r['close'] for r in asset_rows if r['close']]
    if len(closes) < 2:
        return [], 0, 0
        
    retornos = get_log_returns(closes)
    mu_diario = sum(retornos) / len(retornos)
    variance = sum((x - mu_diario)**2 for x in retornos) / len(retornos)
    sigma_diario = math.sqrt(variance)
    
    last_close = closes[-1]
    
    caminos = []
    all_final_prices = []
    
    for _ in range(n_simulations):
        camino_actual = [last_close]
        precio_actual = last_close
        for _ in range(days_ahead):
            # random.gauss(mu, sigma) es la librería estándar
            random_shock = random.gauss(mu_diario, sigma_diario)
            precio_actual = precio_actual * math.exp(random_shock)
            camino_actual.append(precio_actual)
            
        caminos.append(camino_actual)
        all_final_prices.append(camino_actual[-1])
        
    return caminos, min(all_final_prices), max(all_final_prices)

def brute_force_portfolio(series_dict, num_portfolios=1000):
    """
    Optimizador de Portafolio O(N) basado en Teoría de Markowitz (Sharpe Ratio).
    Sortea pesos aleatorios para las llaves y preselecciona el portafolio Max Sharpe.
    No usa scipy.optimize, todo es puramente aleatorio o exhaustivo local.
    """
    tickers = list(series_dict.keys())
    n_assets = len(tickers)
    
    if n_assets < 2:
        return None
        
    # Truncamos todas las series a la menor longitud disponible para operar matrices parejas  
    min_len = min(len(s) for s in series_dict.values())
    
    log_returns = {}
    for t in tickers:
        log_returns[t] = get_log_returns(series_dict[t][-min_len:])
        
    # Mean Returns Anualizados
    mean_rets = {}
    for t in tickers:
        mean_rets[t] = (sum(log_returns[t]) / len(log_returns[t])) * 252
        
    # Calcular covarianza manual de la matriz (N x N)
    n_days = len(log_returns[tickers[0]])
    cov_matrix = {}
    for t1 in tickers:
        cov_matrix[t1] = {}
        for t2 in tickers:
            m1 = sum(log_returns[t1]) / n_days
            m2 = sum(log_returns[t2]) / n_days
            
            covar = sum((log_returns[t1][i] - m1) * (log_returns[t2][i] - m2) for i in range(n_days)) / (n_days - 1)
            # Covarianza Anualizada
            cov_matrix[t1][t2] = covar * 252 
            
    # Proceso Bruto de Sorteos (Monte Carlo Weights)
    best_sharpe = -9999
    best_weights = {}
    best_ret = 0
    best_vol = 0
    
    for _ in range(num_portfolios):
        # 1. Pesos random
        weights_random = [random.random() for _ in range(n_assets)]
        total = sum(weights_random)
        w = {tickers[i]: weights_random[i] / total for i in range(n_assets)}
        
        # 2. Retorno esperado
        port_return = sum(w[t] * mean_rets[t] for t in tickers)
        
        # 3. Varianza / Volatilidad O(N²) calculada manualmente sobre el Set
        port_variance = 0.0
        for t1 in tickers:
            for t2 in tickers:
                port_variance += w[t1] * w[t2] * cov_matrix[t1][t2]
                
        port_volatility = math.sqrt(port_variance) if port_variance > 0 else 0.0001
        
        # 4. Asumimos Tasa Libre de Riesgo 0%
        sharpe_ratio = port_return / port_volatility
        
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_weights = w
            best_ret = port_return
            best_vol = port_volatility
            
    return {
        "sharpe": best_sharpe,
        "weights": best_weights,
        "return": best_ret,
        "volatility": best_vol
    }
