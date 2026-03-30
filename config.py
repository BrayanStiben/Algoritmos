import os

# Rutas de base de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DIR = os.path.join(DATA_DIR, 'raw')
CLEAN_DIR = os.path.join(DATA_DIR, 'clean')
MASTER_DIR = os.path.join(DATA_DIR, 'master')

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aseguramos la existencia de los directorios
for folder in [RAW_DIR, CLEAN_DIR, MASTER_DIR]:
    os.makedirs(folder, exist_ok=True)

# Lista de activos a descargar (mínimo 20 como requerido)
# Incluimos acciones de BVC (intentando nomenclatura de Yahoo Finance o sufijo hipotético .BVC) y ETFs globales.
# Además incluimos ADRs y líderes mundiales para garantizar al menos 20 activos transables
TICKERS = [

    # ACTIVOS COLOMBIANOS - BVC (Bolsa de Valores de Colombia)
    # Yahoo Finance usa sufijo .CL para representar acciones de Colombia

    "CEMARGOS.CL",  # Cementos Argos - Cementera colombiana, construcción e infraestructura
    "GRUPOARGOS.CL",  # Grupo Argos - Holding de infraestructura, energía y concesiones
    "BOGOTA.CL",  # Banco de Bogotá - Segundo banco más grande de Colombia
    "CORFICOLCF.CL",  # Corficolombiana - Holding financiero y banca de inversión
    "CONCONCRET.CL",  # Conconcreto - Construcción y concesiones viales
    "ISA.CL",  # ISA - Transmisión de energía eléctrica en Latinoamérica
    "GEB.CL",  # Grupo Energía Bogotá - Energía, gas natural y transmisión
    "NUTRESA.CL",  # Nutresa - Alimentos procesados con presencia regional
    "CELSIA.CL",  # Celsia - Generación y comercialización de energía
    "PROMIGAS.CL",  # Promigas - Transporte y distribución de gas natural
    "TERPEL.CL",  # Terpel - Distribución de combustibles líder en Colombia
    "ECOPETROL.CL",  # Ecopetrol - Petrolera estatal colombiana
    "MINEROS.CL",  # Mineros - Minería de oro con operaciones en Latinoamérica
    "BVC",  # Bolsa de Valores de Colombia - Propia casa de bolsa

    # ETFs GLOBALES

    "SPY",  # SPDR S&P 500 ETF - Sigue al índice S&P 500 de EE.UU.
    "VOO",  # Vanguard S&P 500 ETF - S&P 500 con menor comisión de administración
    "QQQ",  # Invesco QQQ Trust - Sigue al Nasdaq-100, enfocado en tecnología
    "VTI",  # Vanguard Total Stock Market ETF - Todo el mercado accionario de EE.UU.
    "VEU",  # Vanguard FTSE All-World ex-US ETF - Mercados globales excluyendo EE.UU.

    # ACCIONES GLOBALES

    "AAPL",  # Apple Inc. - Tecnología y hardware
    "MSFT",  # Microsoft Corporation - Software, cloud computing e IA
    "GOOGL",  # Alphabet Inc. - Google, publicidad digital y tecnología
    "AMZN",  # Amazon.com Inc. - Comercio electrónico y cloud computing (AWS)
    "TSLA",  # Tesla Inc. - Vehículos eléctricos y energía limpia
    "META",  # Meta Platforms Inc. - Redes sociales (Facebook, Instagram) y realidad virtual
    "NVDA",  # NVIDIA Corporation - Semiconductores, GPU e inteligencia artificial
    "JPM",  # JPMorgan Chase & Co. - Banca de inversión y servicios financieros
    "V"  # Visa Inc. - Procesamiento de pagos y tecnología financiera
]

# YEARS_HISTORY = 5 # si lo quiero desde la fecha actual
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"