def detect_consecutive_up_days(closes_list, window=3):
    """
    Utiliza una ventana deslizante ('sliding window') para contar cuántas veces 
    se dio el patrón de N días al alza consecutivos.
    Implementación algorítmica explícita manual: O(N)
    """
    n = len(closes_list)
    
    # Se necesitan al menos window + 1 días para tener 'window' incrementos
    if n <= window: return 0
    
    occurrences = 0
    consecutive_ups = 0
    
    # Recorrido algorítmico O(N)
    for i in range(1, n):
        # Condición de "día al alza"
        if closes_list[i] > closes_list[i-1]:
            consecutive_ups += 1
        else:
            consecutive_ups = 0
            
        # Si acumulamos suficientes rachas
        if consecutive_ups >= window:
            occurrences += 1
            
    return occurrences

def detect_reversal_v_pattern(asset_rows):
    """
    Patrón en V o Reversión Clásica: 
    Requiere 3 velas pasadas bajistas y 1 vela fuertemente alcista en el día actual (reversión).
    Complejidad: O(N) usando una lista pura de diccionarios.
    """
    if len(asset_rows) < 4: return 0
    
    occurrences = 0
    for i in range(3, len(asset_rows)):
        # Extraemos variables para limpieza
        o_3 = asset_rows[i-3]['open']; c_3 = asset_rows[i-3]['close']
        o_2 = asset_rows[i-2]['open']; c_2 = asset_rows[i-2]['close']
        o_1 = asset_rows[i-1]['open']; c_1 = asset_rows[i-1]['close']
        o_0 = asset_rows[i]['open'];   c_0 = asset_rows[i]['close']
        
        # Validación de que no hay None
        if None in (o_3, c_3, o_2, c_2, o_1, c_1, o_0, c_0):
            continue
            
        # Condiciones: 3 días previos bajaron
        baja_1 = c_3 < o_3
        baja_2 = c_2 < o_2
        baja_3 = c_1 < o_1
        
        # Día actual subió fuertemente (abre abajo, cierra arriba superando apertura del previo)
        reversion_alcista = (c_0 > o_0) and (c_0 > o_1)
        
        if baja_1 and baja_2 and baja_3 and reversion_alcista:
            occurrences += 1
            
    return occurrences
