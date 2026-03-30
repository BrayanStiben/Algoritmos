import os
import sys
import subprocess

def main():
    """
    Punto de entrada para la aplicación web financiera de análisis algorítmico.
    Si el entorno de Streamlit está instalado correctamente, levantará el servidor web.
    """
    print("=========================================================")
    print(" INICIO DEL PROYECTO FINANCIERO ALGORÍTMICO (WEB DASHBOARD)")
    print("=========================================================\n")
    print("Iniciando motor de Streamlit...")
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    main()