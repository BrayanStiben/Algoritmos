import numpy as np

def euclidean_distance(series1, series2):
    """
    Calcula la distancia Euclidiana entre dos series de tiempo.
    Retorna el valor y la complejidad.
    Implementación explícita O(N).
    """
    # Complejidad: O(N)
    n = len(series1)
    if n == 0: return 0.0, "O(N)"
    
    sum_sq = 0.0
    for i in range(n):
        sum_sq += (series1[i] - series2[i]) ** 2
        
    return sum_sq ** 0.5, "O(N)"

def pearson_correlation(series1, series2):
    """
    Calcula la correlación de Pearson.
    Mide la relación lineal entre dos series. Retorna valor de -1 a 1.
    Implementación explícita O(N).
    """
    n = len(series1)
    if n == 0: return 0.0, "O(N)"
    
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    
    for i in range(n):
        x = float(series1[i])
        y = float(series2[i])
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y
        
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5
    
    if denominator == 0:
        return 0.0, "O(N)"
        
    return numerator / denominator, "O(N)"

def cosine_similarity(series1, series2):
    """
    Similitud por coseno aplicada a vectores.
    Retorna valor de -1 (opuestos) a 1 (idénticos direccionalmente).
    Implementación explícita O(N).
    """
    n = len(series1)
    if n == 0: return 0.0, "O(N)"
    
    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    
    for i in range(n):
        a = float(series1[i])
        b = float(series2[i])
        dot_product += a * b
        norm_a_sq += a * a
        norm_b_sq += b * b
        
    norm_a = norm_a_sq ** 0.5
    norm_b = norm_b_sq ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0, "O(N)"
        
    return dot_product / (norm_a * norm_b), "O(N)"

def dtw_distance(s1, s2):
    """
    Dynamic Time Warping (DTW) manual y optimizado con numpy.
    Permite comparar secuencias desalineadas temporalmente.
    Complejidad: O(N*M)
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), fill_value=np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            # Toma el mínimo valor del vecindario
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m], "O(N*M)"
