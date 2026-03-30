import time
import sys

# Aumentamos el límite de recursión para evitar cortes abruptos con grandes datos en recursivos como Quick Sort
sys.setrecursionlimit(100000)


class Nodo:
    def __init__(self, dato):
        self.dato = dato
        self.izquierdo = None
        self.derecho = None


def _es_mayor(a, b):
    # Criterio: fecha ascendente. Empate: close ascendente.
    if a['fecha'] == b['fecha']:
        return float(a['close']) > float(b['close'])
    return a['fecha'] > b['fecha']


def _es_menor(a, b):
    if a['fecha'] == b['fecha']:
        return float(a['close']) < float(b['close'])
    return a['fecha'] < b['fecha']


# =====================================================================
# 1. SELECTION SORT
# =====================================================================
def selection_sort(arr):
    """
    Selecciona el elemento mínimo de la lista iterando secuencialmente.
    Complejidad Temporal: O(n^2) siempre
    Complejidad Espacial: O(1) in-place
    """
    start_time = time.time()
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if _es_menor(arr[j], arr[min_idx]):
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr, time.time() - start_time


# =====================================================================
# 2. COMB SORT
# =====================================================================
def comb_sort(arr):
    """
    Mejora del Bubble Sort. Usa un factor de reducción (gap) > 1.
    Complejidad Temporal: O(n^2) promedio, peor caso O(n^2), esperado O(n log n).
    Complejidad Espacial: O(1) in-place.
    """
    start_time = time.time()
    n = len(arr)
    gap = n
    shrink = 1.3
    swapped = True
    while gap > 1 or swapped:
        gap = int(float(gap) / shrink)
        if gap < 1: gap = 1
        swapped = False
        for i in range(0, n - gap):
            if _es_mayor(arr[i], arr[i + gap]):
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                swapped = True
    return arr, time.time() - start_time


# =====================================================================
# 3. GNOME SORT
# =====================================================================
def gnome_sort(arr):
    """
    Similar al insertion sort pero reubicando el elemento con intercambios continuos.
    Complejidad Temporal: O(n^2) peor caso, iteración cuasi-lineal si está ordenado.
    Complejidad Espacial: O(1) in-place.
    """
    start_time = time.time()
    index = 0
    n = len(arr)
    while index < n:
        if index == 0 or not _es_menor(arr[index], arr[index - 1]):
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1
    return arr, time.time() - start_time


# =====================================================================
# 4. BINARY INSERTION SORT
# =====================================================================
def binary_insertion_sort(arr):
    """
    Insertion Sort clásico asistido por Búsqueda Binaria para encontrar la posición correcta.
    Complejidad Temporal: O(n^2) por movimientos, O(n log n) comparaciones.
    Complejidad Espacial: O(1) in-place.
    """
    start_time = time.time()
    for i in range(1, len(arr)):
        val = arr[i]
        left, right = 0, i - 1
        while left <= right:
            mid = (left + right) // 2
            if _es_menor(val, arr[mid]):
                right = mid - 1
            else:
                left = mid + 1
        for j in range(i, left, -1):
            arr[j] = arr[j - 1]
        arr[left] = val
    return arr, time.time() - start_time


# =====================================================================
# 5. QUICK SORT
# =====================================================================
def quick_sort(arr):
    """
    Partición Divide and Conquer iterando listas menores y mayores a un pivote.
    Complejidad Temporal: O(n log n) promedio, O(n^2) peor caso.
    Complejidad Espacial: O(log n) a O(n) por tamaño de recursión o creación de super arreglos extra.
    """
    start_time = time.time()

    def _qs(a):
        if len(a) <= 1:
            return a
        pivot = a[len(a) // 2]
        left = []
        middle = []
        right = []
        for x in a:
            if _es_menor(x, pivot):
                left.append(x)
            elif _es_mayor(x, pivot):
                right.append(x)
            else:
                middle.append(x)
        return _qs(left) + middle + _qs(right)

    result = _qs(arr)
    return result, time.time() - start_time


# =====================================================================
# 6. HEAP SORT
# =====================================================================
def heap_sort(arr):
    """
    Convierte el arreglo en un montículo binario (heap max) y extrae elementos de 1 en 1.
    Complejidad Temporal: O(n log n) siempre.
    Complejidad Espacial: O(1) in-place (iterativo) o O(log n) si max_heapify es recursivo.
    """
    start_time = time.time()
    n = len(arr)

    def heapify(n_size, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n_size and _es_mayor(arr[l], arr[largest]):
            largest = l
        if r < n_size and _es_mayor(arr[r], arr[largest]):
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(n_size, largest)

    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(i, 0)

    return arr, time.time() - start_time


# =====================================================================
# 7. TREE SORT
# =====================================================================
def tree_sort(arr):
    """
    Inserta elementos en un Árbol Binario de Búsqueda (BST) y lee todo in-order.
    Complejidad Temporal: O(n log n) promedio, O(n^2) en caso de lista ordenada y sin auto-balance.
    Complejidad Espacial: O(n) espacio para crear árboles y sub-nodos.
    """
    start_time = time.time()
    if not arr: return arr, 0.0

    def insertar(nodo, elemento):
        if _es_menor(elemento, nodo.dato):
            if nodo.izquierdo is None:
                nodo.izquierdo = Nodo(elemento)
            else:
                insertar(nodo.izquierdo, elemento)
        else:
            if nodo.derecho is None:
                nodo.derecho = Nodo(elemento)
            else:
                insertar(nodo.derecho, elemento)

    def transversa_en_orden(nodo, salida):
        if nodo is not None:
            transversa_en_orden(nodo.izquierdo, salida)
            salida.append(nodo.dato)
            transversa_en_orden(nodo.derecho, salida)

    raiz = Nodo(arr[0])
    longitud = len(arr)
    # Por limitaciones de recursión de python sobre O(n^2), iteraremos la insersión
    # No usaremos la recursión para la insersión estricta si n>1000
    for i in range(1, longitud):
        current = raiz
        elem = arr[i]
        while True:
            if _es_menor(elem, current.dato):
                if current.izquierdo is None:
                    current.izquierdo = Nodo(elem)
                    break
                else:
                    current = current.izquierdo
            else:
                if current.derecho is None:
                    current.derecho = Nodo(elem)
                    break
                else:
                    current = current.derecho

    res = []
    transversa_en_orden(raiz, res)
    return res, time.time() - start_time


# =====================================================================
# 8. TIM SORT (Custom simplified)
# =====================================================================
def tim_sort(arr):
    """
    Mezcla entre Insertion y Merge Sort. Identifica runs estructurados para procesar.
    Complejidad Temporal: O(n log n) peor caso, acercando a O(n) estado cuasi-ordenado.
    Complejidad Espacial: O(n) espacio memoria suplementaria en mezclado.
    """
    start_time = time.time()
    MIN_MERGE = 32
    n = len(arr)

    def calc_min_run(n):
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    min_run = calc_min_run(n)

    def __insercion_ts(izq, der):
        for i in range(izq + 1, der + 1):
            j = i
            while j > izq and _es_menor(arr[j], arr[j - 1]):
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1

    def __mezcla_ts(l, m, r):
        len1, len2 = m - l + 1, r - m
        left_p, right_p = [], []
        for i in range(0, len1): left_p.append(arr[l + i])
        for i in range(0, len2): right_p.append(arr[m + 1 + i])

        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if not _es_mayor(left_p[i], right_p[j]):
                arr[k] = left_p[i]
                i += 1
            else:
                arr[k] = right_p[j]
                j += 1
            k += 1
        while i < len1:
            arr[k] = left_p[i]
            k += 1
            i += 1
        while j < len2:
            arr[k] = right_p[j]
            k += 1
            j += 1

    for i in range(0, n, min_run):
        __insercion_ts(i, min((i + MIN_MERGE - 1), (n - 1)))

    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min((left + size - 1), (n - 1))
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                __mezcla_ts(left, mid, right)
        size *= 2

    return arr, time.time() - start_time


# =====================================================================
# 9. PIGEONHOLE SORT
# =====================================================================
def pigeonhole_sort(arr):
    """
    Agrupa elementos en contenedores para su misma clase.
    Complejidad Temporal: O(n + N) para el rango N. Pésimo para alta diferenciación.
    Complejidad Espacial: O(n + N) requiere arreglos de recolección temporal abundantes.
    """
    start_time = time.time()
    if not arr: return arr, 0.0

    def fecha_int(d):
        return int(d.replace("-", "")[:6])  # YYYYMM para limitar dimension N

    mi = fecha_int(arr[0]['fecha'])
    ma = mi
    for x in arr:
        v = fecha_int(x['fecha'])
        if v < mi: mi = v
        if v > ma: ma = v

    size = ma - mi + 1
    holes = [[] for _ in range(size)]
    for x in arr: holes[fecha_int(x['fecha']) - mi].append(x)

    out = []
    for hole in holes:
        if hole:
            # Reutilizamos quicksort in-bucket por desempate close
            hole_s, _ = quick_sort(hole)
            for e in hole_s: out.append(e)

    return out, time.time() - start_time


# =====================================================================
# 10. BUCKET SORT
# =====================================================================
def bucket_sort(arr):
    """
    Divide elementos en canastas equitativas, ordena locamente y une canastas.
    Complejidad Temporal: O(n + k), O(n^2) en el peor si todos van a uno.
    Complejidad Espacial: O(n) por los diccionarios agrupadores.
    """
    start_time = time.time()
    if not arr: return arr, 0.0

    cubetas = {}
    for elemento in arr:
        ano = int(elemento['fecha'][:4])
        if ano not in cubetas: cubetas[ano] = []
        cubetas[ano].append(elemento)

    claves = []
    for k in cubetas: claves.append(k)
    for i in range(len(claves)):
        for j in range(i + 1, len(claves)):
            if claves[j] < claves[i]: claves[i], claves[j] = claves[j], claves[i]

    out = []
    for k in claves:
        c, _ = quick_sort(cubetas[k])
        for x in c: out.append(x)

    return out, time.time() - start_time


# =====================================================================
# 11. RADIX SORT
# =====================================================================
def radix_sort(arr):
    """
    Sort por cada dígito posicional, empezando del final al principio usando conteo local.
    Complejidad Temporal: O(d*(n+b)) con d digitos, b base cardinal.
    Complejidad Espacial: O(n+b).
    """
    start_time = time.time()
    if not arr: return arr, 0.0

    def dt_val(d):
        return int(d['fecha'].replace("-", ""))

    val_maximo = max(dt_val(x) for x in arr)

    def counting_sort_radix(arreglo_d, exp):
        longitud = len(arreglo_d)
        salida = [0] * longitud
        conteo = [0] * 10
        for i in range(longitud):
            indice = (dt_val(arreglo_d[i]) // exp) % 10
            conteo[indice] += 1
        for i in range(1, 10): conteo[i] += conteo[i - 1]
        for i in range(longitud - 1, -1, -1):
            indice = (dt_val(arreglo_d[i]) // exp) % 10
            salida[conteo[indice] - 1] = arreglo_d[i]
            conteo[indice] -= 1
        return salida

    arreglo_work = arr[:]  # clone temp stack
    arreglo_work = counting_sort_radix(arreglo_work, 1)
    exponente = 10
    while val_maximo // exponente > 0:
        arreglo_work = counting_sort_radix(arreglo_work, exponente)
        exponente *= 10

    # Aplicar corrección tie-break per radix (porque usé enteros simples para buckets sin floats)
    out, _ = tim_sort(arreglo_work)
    return out, time.time() - start_time


# =====================================================================
# 12. BITONIC SORT
# =====================================================================
def bitonic_sort(arr):
    """
    Algoritmo en paralelo generando secuencias ascendentes y descendentes y mezclando.
    Complejidad Temporal: O(n log^2 n) en serial (nuestro caso), O(log^2 n) en hardware nativo paralelo.
    Complejidad Espacial: O(n log^2 n) si se clonan arrays intensamente, O(n) por pad de 2^x arrays.
    """
    start_time = time.time()
    longitud = len(arr)
    if longitud <= 1: return arr, time.time() - start_time

    potencia = 1
    while potencia < longitud: potencia *= 2

    val_relleno = {'fecha': '9999-99-99', 'close': float('inf'), 'volumen': 0, 'ticker': 'PAD'}
    longitud_relleno = potencia - longitud

    arreglo_c = []
    for x in arr: arreglo_c.append(x)
    for _ in range(longitud_relleno): arreglo_c.append(val_relleno)

    def c_swap(a, i, j, d_):
        if (d_ == 1 and _es_mayor(a[i], a[j])) or (d_ == 0 and _es_menor(a[i], a[j])):
            a[i], a[j] = a[j], a[i]

    def _mergest(a, l_idx, count, d_):
        if count > 1:
            k = count // 2
            for i in range(l_idx, l_idx + k):
                c_swap(a, i, i + k, d_)
            _mergest(a, l_idx, k, d_)
            _mergest(a, l_idx + k, k, d_)

    def _bitonic_st(a, l_idx, count, d_):
        if count > 1:
            k = count // 2
            _bitonic_st(a, l_idx, k, 1)
            _bitonic_st(a, l_idx + k, k, 0)
            _mergest(a, l_idx, count, d_)

    _bitonic_st(arreglo_c, 0, potencia, 1)

    out = []
    # Remover dummies
    for i in range(longitud):
        if arreglo_c[i]['fecha'] != '9999-99-99':
            out.append(arreglo_c[i])

    return out, time.time() - start_time
