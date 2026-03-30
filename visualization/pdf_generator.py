from fpdf import FPDF
import math
import os
from datetime import datetime

class ReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Reporte Técnico: Análisis de Datos Financieros Algorítmico', border=False, ln=1, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def generate_technical_report(risk_results, output_path):
    pdf = ReportPDF()
    pdf.add_page()
    
    # Metadatos del modelo
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 10, txt=f"Fecha de Generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, txt="1. Perfil de Riesgo del Portafolio", ln=1)
    
    pdf.set_font("helvetica", size=10)
    pdf.multi_cell(0, 6, txt="Las siguientes métricas son calculadas mediante la volatilidad histórica anualizada (Desviación Estándar de retornos logarítmicos).")
    pdf.ln(3)
    
    # Tabla simplificada
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(40, 8, "Activo / Ticker", border=1, align='C')
    pdf.cell(50, 8, "Volatilidad Anual", border=1, align='C')
    pdf.cell(50, 8, "Perfil de Riesgo", border=1, align='C')
    pdf.ln()
    
    pdf.set_font("helvetica", "", 10)
    for row in risk_results:
        pdf.cell(40, 8, str(row['ticker']), border=1, align='C')
        
        # Validar NaN manualmente sin pandas
        if math.isnan(row['volatilidad_anual']):
            vol_str = "N/A"
        else:
            vol_str = f"{row['volatilidad_anual']*100:.2f}%"
            
        pdf.cell(50, 8, vol_str, border=1, align='C')
        pdf.cell(50, 8, str(row['perfil']), border=1, align='C')
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, txt="2. Fundamentos Algorítmicos y ETL", ln=1)
    pdf.set_font("helvetica", size=10)
    txt_fundamentos = (
        "DTW (Dynamic Time Warping): Algoritmo de programación dinámica de complejidad O(N*M) "
        "usado para medir la distancia entre dos secuencias temporales que pueden variar en velocidad.\n\n"
        "Matriz de Correlación: Complejidad espacial y temporal dependiente de las combinaciones de pares "
        "O(C^2) calculado algorítmicamente mediante Pearson.\n\n"
        "Ingesta ETL (Iterador Python Puro): Complejidad O(N). Para corregir desalineamiento de calendarios y días feriados, "
        "se utiliza técnica 'Carry-Forward' algorítmica iteradora trasladando la última cotización conocida en el panel global cruzado mediante bucles."
    )
    pdf.multi_cell(0, 6, txt=txt_fundamentos)
    
    pdf.output(output_path)
    return output_path
