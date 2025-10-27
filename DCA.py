import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
from io import BytesIO
import calendar

# Configuración de la página
st.set_page_config(
    page_title="BQuant-DCA Pro Visualizer v3.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .portfolio-preset {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    div[data-testid="stPlotlyChart"] > div {
        transition: none !important;
        animation: none !important;
    }
    .info-container {
        transition: opacity 0.2s ease-in-out;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-new {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📈 BQuant-DCA Pro Visualizer v3.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Análisis Avanzado de Dollar Cost Averaging con Portfolio Management</p>', unsafe_allow_html=True)

# Diccionario de activos predefinidos
ACTIVOS_PREDEFINIDOS = {
    'SPY': {'nombre': 'SPDR S&P 500 ETF', 'color': '#FF6B6B', 'emoji': '🇺🇸'},
    'VTI': {'nombre': 'Vanguard Total Stock Market', 'color': '#4ECDC4', 'emoji': '📊'},
    'QQQ': {'nombre': 'Invesco QQQ Trust', 'color': '#45B7D1', 'emoji': '💻'},
    'VWO': {'nombre': 'Vanguard Emerging Markets', 'color': '#96CEB4', 'emoji': '🌍'},
    'VEA': {'nombre': 'Vanguard Developed Markets', 'color': '#FFEAA7', 'emoji': '🌎'},
    'BND': {'nombre': 'Vanguard Total Bond Market', 'color': '#DDA0DD', 'emoji': '🏦'},
    'GLD': {'nombre': 'SPDR Gold Shares', 'color': '#FFD700', 'emoji': '🥇'},
    'VNQ': {'nombre': 'Vanguard Real Estate', 'color': '#FF7675', 'emoji': '🏠'},
    'BRK-B': {'nombre': 'Berkshire Hathaway', 'color': '#00B894', 'emoji': '💎'},
    'MSFT': {'nombre': 'Microsoft', 'color': '#0984E3', 'emoji': '💼'},
    'AAPL': {'nombre': 'Apple', 'color': '#A29BFE', 'emoji': '🍎'},
    'GOOGL': {'nombre': 'Alphabet', 'color': '#FD79A8', 'emoji': '🔍'},
    'TSLA': {'nombre': 'Tesla', 'color': '#00CED1', 'emoji': '⚡'},
    'NVDA': {'nombre': 'NVIDIA', 'color': '#32CD32', 'emoji': '🎮'},
    'AMZN': {'nombre': 'Amazon', 'color': '#FF8C00', 'emoji': '📦'}
}

# Portfolios predefinidos
PORTFOLIO_PRESETS = {
    'Agresivo': {
        'descripcion': '100% Acciones - Alto crecimiento',
        'allocations': {'SPY': 60, 'QQQ': 30, 'VWO': 10}
    },
    'Moderado': {
        'descripcion': '70/30 - Balance crecimiento/estabilidad',
        'allocations': {'SPY': 50, 'BND': 30, 'VEA': 20}
    },
    'Conservador': {
        'descripcion': '40/60 - Enfoque en preservación',
        'allocations': {'BND': 60, 'SPY': 30, 'GLD': 10}
    },
    'Tech Heavy': {
        'descripcion': 'Enfoque en tecnología',
        'allocations': {'QQQ': 50, 'MSFT': 20, 'NVDA': 15, 'AAPL': 15}
    },
    '60/40 Clásico': {
        'descripcion': 'Portfolio clásico balanceado',
        'allocations': {'VTI': 60, 'BND': 40}
    }
}

# ============================================================================
# FUNCIONES DE CÁLCULO AVANZADAS
# ============================================================================

@st.cache_data(ttl=3600)
def descargar_datos_yfinance(tickers, fecha_inicio, fecha_fin):
    """Descarga datos históricos de Yahoo Finance"""
    try:
        with st.spinner(f"🔄 Descargando datos de {len(tickers)} activos..."):
            datos = yf.download(tickers, start=fecha_inicio, end=fecha_fin, progress=False, multi_level_index=False)
            if len(tickers) == 1:
                datos.columns = pd.MultiIndex.from_product([datos.columns, [tickers[0]]])
            
            precios = datos['Close'].ffill().bfill()
            return precios
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
        return None

def generar_fechas_inversion(fecha_inicio, fecha_fin, frecuencia='monthly', dias_especificos=None, 
                             dia_semana=None, fechas_pausa=None):
    """
    Genera fechas de inversión según configuración avanzada
    
    Args:
        fecha_inicio: Fecha de inicio
        fecha_fin: Fecha de fin
        frecuencia: 'daily', 'weekly', 'biweekly', 'monthly', 'quarterly', 'custom'
        dias_especificos: Lista de días del mes [1, 15] para frecuencia mensual
        dia_semana: 0-6 (Lunes-Domingo) para frecuencia semanal
        fechas_pausa: Lista de tuplas (inicio, fin) para períodos sin inversión
    """
    fechas = []
    fecha_actual = pd.Timestamp(fecha_inicio)
    fecha_final = pd.Timestamp(fecha_fin)
    
    if fechas_pausa is None:
        fechas_pausa = []
    
    def esta_en_pausa(fecha):
        """Verifica si una fecha está en período de pausa"""
        for inicio, fin in fechas_pausa:
            if pd.Timestamp(inicio) <= fecha <= pd.Timestamp(fin):
                return True
        return False
    
    if frecuencia == 'daily':
        # Inversión diaria (días hábiles)
        fecha_range = pd.bdate_range(start=fecha_inicio, end=fecha_fin)
        fechas = [f for f in fecha_range if not esta_en_pausa(f)]
        
    elif frecuencia == 'weekly':
        # Inversión semanal
        dia = dia_semana if dia_semana is not None else 0  # Lunes por defecto
        while fecha_actual <= fecha_final:
            # Ajustar al día de la semana especificado
            dias_hasta = (dia - fecha_actual.weekday()) % 7
            fecha_objetivo = fecha_actual + timedelta(days=dias_hasta)
            if fecha_objetivo <= fecha_final and not esta_en_pausa(fecha_objetivo):
                fechas.append(fecha_objetivo)
            fecha_actual += timedelta(weeks=1)
    
    elif frecuencia == 'biweekly':
        # Inversión quincenal
        while fecha_actual <= fecha_final:
            if not esta_en_pausa(fecha_actual):
                fechas.append(fecha_actual)
            fecha_actual += timedelta(weeks=2)
    
    elif frecuencia == 'monthly':
        # Inversión mensual (primer día hábil o días específicos)
        if dias_especificos and len(dias_especificos) > 0:
            # Días específicos del mes
            while fecha_actual <= fecha_final:
                for dia in dias_especificos:
                    try:
                        fecha_objetivo = fecha_actual.replace(day=dia)
                        if fecha_objetivo <= fecha_final and not esta_en_pausa(fecha_objetivo):
                            fechas.append(fecha_objetivo)
                    except ValueError:
                        # Día no válido para este mes (ej: 31 en febrero)
                        pass
                fecha_actual = fecha_actual + pd.DateOffset(months=1)
                fecha_actual = fecha_actual.replace(day=1)
        else:
            # Primer día del mes
            while fecha_actual <= fecha_final:
                primer_dia = fecha_actual.replace(day=1)
                if primer_dia <= fecha_final and not esta_en_pausa(primer_dia):
                    fechas.append(primer_dia)
                fecha_actual = fecha_actual + pd.DateOffset(months=1)
    
    elif frecuencia == 'quarterly':
        # Inversión trimestral
        while fecha_actual <= fecha_final:
            if not esta_en_pausa(fecha_actual):
                fechas.append(fecha_actual)
            fecha_actual = fecha_actual + pd.DateOffset(months=3)
    
    elif frecuencia == 'custom':
        # Las fechas personalizadas se pasan directamente
        fechas = dias_especificos if dias_especificos else []
    
    return sorted(set(fechas))

def calcular_dca_portfolio_avanzado(precios_dict, config_portfolio):
    """
    Calcula DCA para un portfolio con asignaciones personalizadas y configuración avanzada
    
    Args:
        precios_dict: Dict con precios por ticker
        config_portfolio: Dict con configuración del portfolio
            - allocations: Dict {ticker: porcentaje}
            - inversion_total: Cantidad total a invertir
            - fechas_inversion: Lista de fechas de inversión
            - lump_sum: Cantidad inicial (opcional)
            - aumentos_anuales: % de aumento anual (opcional)
            - rebalanceo: Frecuencia de rebalanceo (opcional)
    """
    allocations = config_portfolio['allocations']
    inversion_por_periodo = config_portfolio['inversion_total']
    fechas_inversion = config_portfolio['fechas_inversion']
    lump_sum = config_portfolio.get('lump_sum', 0)
    aumentos_anuales = config_portfolio.get('aumentos_anuales', 0)
    
    # Normalizar asignaciones si no suman 100%
    total_alloc = sum(allocations.values())
    if total_alloc != 100:
        allocations = {k: (v/total_alloc)*100 for k, v in allocations.items()}
    
    # Obtener todas las fechas de la serie
    todas_las_fechas = None
    for ticker in precios_dict.keys():
        if todas_las_fechas is None:
            todas_las_fechas = precios_dict[ticker].index
        else:
            todas_las_fechas = todas_las_fechas.union(precios_dict[ticker].index)
    
    todas_las_fechas = sorted(todas_las_fechas)
    
    # Inicializar variables
    acciones_por_ticker = {ticker: 0.0 for ticker in allocations.keys()}
    inversion_total_acumulada = 0.0
    año_actual = todas_las_fechas[0].year if todas_las_fechas else datetime.now().year
    inversion_actual = inversion_por_periodo
    
    # Series de resultados
    valor_portfolio_series = pd.Series(index=todas_las_fechas, dtype=float)
    inversion_acumulada_series = pd.Series(index=todas_las_fechas, dtype=float)
    rentabilidad_series = pd.Series(index=todas_las_fechas, dtype=float)
    
    # Resultados detallados por ticker
    resultados_por_ticker = {
        ticker: {
            'acciones': pd.Series(index=todas_las_fechas, dtype=float),
            'valor': pd.Series(index=todas_las_fechas, dtype=float),
            'inversion': pd.Series(index=todas_las_fechas, dtype=float)
        }
        for ticker in allocations.keys()
    }
    
    # Procesar lump sum inicial
    if lump_sum > 0 and todas_las_fechas:
        fecha_inicial = todas_las_fechas[0]
        for ticker, porcentaje in allocations.items():
            if ticker in precios_dict:
                precio_inicial = precios_dict[ticker].loc[fecha_inicial] if fecha_inicial in precios_dict[ticker].index else None
                if precio_inicial and precio_inicial > 0:
                    cantidad_invertir = lump_sum * (porcentaje / 100)
                    acciones_por_ticker[ticker] += cantidad_invertir / precio_inicial
        inversion_total_acumulada += lump_sum
    
    # Procesar cada fecha
    for fecha in todas_las_fechas:
        # Verificar si es fecha de inversión
        if fecha in fechas_inversion:
            # Aplicar aumentos anuales
            if fecha.year > año_actual and aumentos_anuales > 0:
                años_transcurridos = fecha.year - año_actual
                inversion_actual = inversion_por_periodo * ((1 + aumentos_anuales/100) ** años_transcurridos)
            
            # Invertir según asignaciones
            for ticker, porcentaje in allocations.items():
                if ticker in precios_dict and fecha in precios_dict[ticker].index:
                    precio_compra = precios_dict[ticker].loc[fecha]
                    if precio_compra > 0:
                        cantidad_invertir = inversion_actual * (porcentaje / 100)
                        nuevas_acciones = cantidad_invertir / precio_compra
                        acciones_por_ticker[ticker] += nuevas_acciones
                        inversion_total_acumulada += cantidad_invertir
        
        # Calcular valor del portfolio en esta fecha
        valor_total_hoy = 0.0
        for ticker, num_acciones in acciones_por_ticker.items():
            if ticker in precios_dict and fecha in precios_dict[ticker].index:
                precio_actual = precios_dict[ticker].loc[fecha]
                valor_ticker = num_acciones * precio_actual
                valor_total_hoy += valor_ticker
                
                # Guardar detalles por ticker
                resultados_por_ticker[ticker]['acciones'].loc[fecha] = num_acciones
                resultados_por_ticker[ticker]['valor'].loc[fecha] = valor_ticker
        
        # Calcular rentabilidad
        if inversion_total_acumulada > 0:
            rentabilidad = ((valor_total_hoy / inversion_total_acumulada) - 1) * 100
        else:
            rentabilidad = 0.0
        
        # Guardar valores
        valor_portfolio_series.loc[fecha] = valor_total_hoy
        inversion_acumulada_series.loc[fecha] = inversion_total_acumulada
        rentabilidad_series.loc[fecha] = rentabilidad
    
    return {
        'rentabilidad': rentabilidad_series,
        'valor_portfolio': valor_portfolio_series,
        'inversion_acumulada': inversion_acumulada_series,
        'detalles_por_ticker': resultados_por_ticker,
        'acciones_finales': acciones_por_ticker
    }

def calcular_lump_sum_comparison(precios_dict, allocations, monto_inicial, fecha_inicio):
    """Calcula el rendimiento de una inversión lump sum para comparación"""
    # Normalizar asignaciones
    total_alloc = sum(allocations.values())
    if total_alloc != 100:
        allocations = {k: (v/total_alloc)*100 for k, v in allocations.items()}
    
    # Obtener todas las fechas
    todas_las_fechas = None
    for ticker in precios_dict.keys():
        if todas_las_fechas is None:
            todas_las_fechas = precios_dict[ticker].index
        else:
            todas_las_fechas = todas_las_fechas.union(precios_dict[ticker].index)
    
    todas_las_fechas = sorted([f for f in todas_las_fechas if f >= pd.Timestamp(fecha_inicio)])
    
    if not todas_las_fechas:
        return pd.Series(dtype=float)
    
    # Comprar acciones en la fecha inicial
    fecha_compra = todas_las_fechas[0]
    acciones_por_ticker = {}
    
    for ticker, porcentaje in allocations.items():
        if ticker in precios_dict and fecha_compra in precios_dict[ticker].index:
            precio_compra = precios_dict[ticker].loc[fecha_compra]
            if precio_compra > 0:
                cantidad_invertir = monto_inicial * (porcentaje / 100)
                acciones_por_ticker[ticker] = cantidad_invertir / precio_compra
    
    # Calcular valor día a día
    valor_series = pd.Series(index=todas_las_fechas, dtype=float)
    
    for fecha in todas_las_fechas:
        valor_total = 0.0
        for ticker, num_acciones in acciones_por_ticker.items():
            if ticker in precios_dict and fecha in precios_dict[ticker].index:
                precio_actual = precios_dict[ticker].loc[fecha]
                valor_total += num_acciones * precio_actual
        
        valor_series.loc[fecha] = ((valor_total / monto_inicial) - 1) * 100 if monto_inicial > 0 else 0
    
    return valor_series

def calcular_metricas_avanzadas(rentabilidad_series, valores_series, inversion_series):
    """Calcula métricas avanzadas de rendimiento"""
    rentabilidad_clean = rentabilidad_series.dropna()
    valores_clean = valores_series.dropna()
    inversion_clean = inversion_series.dropna()
    
    if len(rentabilidad_clean) < 2:
        return {}
    
    # Rentabilidad final
    rentabilidad_total = rentabilidad_clean.iloc[-1]
    
    # Valor final vs inversión
    valor_final = valores_clean.iloc[-1] if len(valores_clean) > 0 else 0
    inversion_final = inversion_clean.iloc[-1] if len(inversion_clean) > 0 else 1
    
    # Calcular retornos diarios
    retornos_diarios = rentabilidad_clean.pct_change().dropna()
    
    # Volatilidad (desviación estándar anualizada)
    volatilidad = retornos_diarios.std() * np.sqrt(252) * 100 if len(retornos_diarios) > 0 else 0
    
    # Sharpe Ratio (asumiendo tasa libre de riesgo = 0)
    retorno_medio_anual = retornos_diarios.mean() * 252 * 100 if len(retornos_diarios) > 0 else 0
    sharpe_ratio = retorno_medio_anual / volatilidad if volatilidad > 0 else 0
    
    # Maximum Drawdown
    valores_acumulados = valores_clean.cummax()
    drawdowns = (valores_clean - valores_acumulados) / valores_acumulados * 100
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
    
    # CAGR (Compound Annual Growth Rate)
    if inversion_final > 0 and len(rentabilidad_clean) > 30:
        años = len(rentabilidad_clean) / 252  # Asumiendo días hábiles
        cagr = (((valor_final / inversion_final) ** (1/años)) - 1) * 100 if años > 0 else 0
    else:
        cagr = 0
    
    # Mejor y peor año (aproximado)
    rentabilidad_por_año = rentabilidad_clean.groupby(rentabilidad_clean.index.year).last()
    if len(rentabilidad_por_año) > 1:
        retornos_anuales = rentabilidad_por_año.diff().dropna()
        mejor_año = retornos_anuales.max() if len(retornos_anuales) > 0 else 0
        peor_año = retornos_anuales.min() if len(retornos_anuales) > 0 else 0
    else:
        mejor_año = peor_año = 0
    
    # Tiempo en positivo
    dias_positivos = (rentabilidad_clean > 0).sum()
    total_dias = len(rentabilidad_clean)
    porcentaje_positivo = (dias_positivos / total_dias * 100) if total_dias > 0 else 0
    
    return {
        'rentabilidad_total': rentabilidad_total,
        'valor_final': valor_final,
        'inversion_final': inversion_final,
        'volatilidad': volatilidad,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cagr': cagr,
        'mejor_año': mejor_año,
        'peor_año': peor_año,
        'porcentaje_tiempo_positivo': porcentaje_positivo,
        'ganancia_neta': valor_final - inversion_final
    }

def exportar_a_excel(resultados_portfolio, metricas):
    """Exporta resultados a Excel con múltiples hojas"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja 1: Resumen de métricas
        df_metricas = pd.DataFrame([metricas])
        df_metricas.to_excel(writer, sheet_name='Métricas', index=False)
        
        # Hoja 2: Evolución del portfolio
        df_evolucion = pd.DataFrame({
            'Fecha': resultados_portfolio['rentabilidad'].index,
            'Rentabilidad (%)': resultados_portfolio['rentabilidad'].values,
            'Valor Portfolio ($)': resultados_portfolio['valor_portfolio'].values,
            'Inversión Acumulada ($)': resultados_portfolio['inversion_acumulada'].values
        })
        df_evolucion.to_excel(writer, sheet_name='Evolución', index=False)
        
        # Hoja 3: Detalles por activo
        if 'detalles_por_ticker' in resultados_portfolio:
            for ticker, datos in resultados_portfolio['detalles_por_ticker'].items():
                df_ticker = pd.DataFrame({
                    'Fecha': datos['acciones'].index,
                    'Acciones': datos['acciones'].values,
                    'Valor ($)': datos['valor'].values
                })
                df_ticker.to_excel(writer, sheet_name=f'Activo_{ticker}', index=False)
    
    output.seek(0)
    return output

# ============================================================================
# INTERFAZ SIDEBAR
# ============================================================================

st.sidebar.markdown("## ⚙️ Configuración de Análisis")

# Tabs para organizar mejor la configuración
tab_basico, tab_avanzado, tab_portfolio = st.sidebar.tabs(["📊 Básico", "🎯 Avanzado", "💼 Portfolio"])

with tab_basico:
    st.markdown("### 📅 Rango de Fechas")
    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio = st.date_input(
            "Inicio:",
            value=datetime(2020, 1, 1),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now()
        )
    with col2:
        fecha_fin = st.date_input(
            "Fin:",
            value=datetime.now(),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now()
        )
    
    # Configuración de frecuencia de inversión
    st.markdown("### 🗓️ Frecuencia de Inversión")
    frecuencia_dca = st.selectbox(
        "Frecuencia:",
        ['monthly', 'biweekly', 'weekly', 'quarterly', 'daily'],
        format_func=lambda x: {
            'daily': '📆 Diaria (días hábiles)',
            'weekly': '📅 Semanal',
            'biweekly': '📊 Quincenal',
            'monthly': '📈 Mensual',
            'quarterly': '📉 Trimestral'
        }[x],
        index=0
    )
    
    # Opciones específicas según frecuencia
    dias_especificos = None
    dia_semana = None
    
    if frecuencia_dca == 'monthly':
        st.markdown("**Días del mes para invertir:**")
        usar_dias_especificos = st.checkbox("Usar días específicos", value=False)
        
        if usar_dias_especificos:
            dias_mes = st.multiselect(
                "Selecciona días:",
                options=list(range(1, 32)),
                default=[1, 15],
                format_func=lambda x: f"Día {x}"
            )
            dias_especificos = dias_mes if dias_mes else [1]
    
    elif frecuencia_dca == 'weekly':
        dia_semana = st.selectbox(
            "Día de la semana:",
            options=list(range(7)),
            format_func=lambda x: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][x],
            index=0
        )
    
    st.markdown("### 💰 Inversión por Período")
    inversion_por_periodo = st.select_slider(
        "Cantidad ($):",
        options=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
        value=500,
        format_func=lambda x: f"${x:,}"
    )

with tab_avanzado:
    st.markdown("### 💵 Inversión Inicial (Lump Sum)")
    usar_lump_sum = st.checkbox("Agregar inversión inicial", value=False)
    lump_sum_amount = 0
    if usar_lump_sum:
        lump_sum_amount = st.number_input(
            "Monto inicial ($):",
            min_value=0,
            max_value=1000000,
            value=1000,
            step=100
        )
    
    st.markdown("### 📈 Aumentos Anuales")
    usar_aumentos = st.checkbox("Aplicar aumentos anuales", value=False)
    aumento_anual = 0
    if usar_aumentos:
        aumento_anual = st.slider(
            "% de aumento anual:",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            format="%d%%"
        )
    
    st.markdown("### ⏸️ Períodos de Pausa")
    usar_pausas = st.checkbox("Pausar inversiones en períodos específicos", value=False)
    pausas = []
    if usar_pausas:
        st.info("💡 Simula períodos sin inversión (ej: crisis, cambio de trabajo)")
        num_pausas = st.number_input("Número de pausas:", min_value=1, max_value=5, value=1)
        
        for i in range(num_pausas):
            col1, col2 = st.columns(2)
            with col1:
                inicio_pausa = st.date_input(
                    f"Inicio pausa {i+1}:",
                    value=datetime(2022, 1, 1),
                    key=f"pausa_inicio_{i}"
                )
            with col2:
                fin_pausa = st.date_input(
                    f"Fin pausa {i+1}:",
                    value=datetime(2022, 6, 30),
                    key=f"pausa_fin_{i}"
                )
            pausas.append((inicio_pausa, fin_pausa))
    
    st.markdown("### 📊 Comparaciones")
    comparar_lump_sum = st.checkbox("Comparar con inversión Lump Sum", value=True)
    
    st.markdown("### 🎲 Simulación Monte Carlo")
    usar_monte_carlo = st.checkbox("Ejecutar simulación Monte Carlo", value=False)
    if usar_monte_carlo:
        num_simulaciones = st.slider("Número de simulaciones:", 100, 10000, 1000, 100)

with tab_portfolio:
    st.markdown("### 💼 Gestión de Portfolio")
    
    # Selección de preset o custom
    modo_portfolio = st.radio(
        "Modo de configuración:",
        ['Presets', 'Personalizado'],
        horizontal=True
    )
    
    allocations = {}
    
    if modo_portfolio == 'Presets':
        preset_seleccionado = st.selectbox(
            "Selecciona un portfolio predefinido:",
            list(PORTFOLIO_PRESETS.keys()),
            format_func=lambda x: f"{x} - {PORTFOLIO_PRESETS[x]['descripcion']}"
        )
        
        allocations = PORTFOLIO_PRESETS[preset_seleccionado]['allocations'].copy()
        
        st.markdown("**📊 Asignación:**")
        for ticker, pct in allocations.items():
            emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
            st.markdown(f"{emoji} **{ticker}**: {pct}%")
    
    else:  # Personalizado
        st.markdown("**Selecciona activos y asigna porcentajes:**")
        
        # Selector de activos
        activos_seleccionados = st.multiselect(
            "Activos:",
            options=list(ACTIVOS_PREDEFINIDOS.keys()),
            default=['SPY', 'BND'],
            format_func=lambda x: f"{ACTIVOS_PREDEFINIDOS[x]['emoji']} {x} - {ACTIVOS_PREDEFINIDOS[x]['nombre'][:25]}"
        )
        
        # Activos custom
        with st.expander("➕ Agregar activos personalizados"):
            activos_custom = st.text_input(
                "Tickers (separados por comas):",
                placeholder="TSLA, NVDA, AMZN"
            )
            
            if activos_custom:
                custom_list = [ticker.strip().upper() for ticker in activos_custom.split(',') if ticker.strip()]
                activos_seleccionados.extend(custom_list)
        
        # Asignar porcentajes
        if activos_seleccionados:
            st.markdown("**Asigna porcentajes:**")
            
            total_asignado = 0
            for ticker in activos_seleccionados:
                emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                pct = st.slider(
                    f"{emoji} {ticker}:",
                    min_value=0,
                    max_value=100,
                    value=100 // len(activos_seleccionados),
                    step=5,
                    key=f"alloc_{ticker}"
                )
                allocations[ticker] = pct
                total_asignado += pct
            
            # Mostrar total
            if total_asignado != 100:
                st.warning(f"⚠️ Total asignado: {total_asignado}% (se normalizará automáticamente)")
            else:
                st.success(f"✅ Total asignado: {total_asignado}%")
    
    # Guardar/Cargar configuraciones
    st.markdown("### 💾 Guardar/Cargar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Guardar Config"):
            config_actual = {
                'allocations': allocations,
                'inversion': inversion_por_periodo,
                'frecuencia': frecuencia_dca,
                'lump_sum': lump_sum_amount,
                'aumentos': aumento_anual
            }
            st.session_state.config_guardada = config_actual
            st.success("✅ Guardado!")
    
    with col2:
        if st.button("📥 Cargar Config"):
            if 'config_guardada' in st.session_state:
                st.info("✅ Config cargada!")
            else:
                st.warning("No hay config guardada")

# ============================================================================
# CONTROLES DE VISUALIZACIÓN
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎬 Visualización")

velocidad_animacion = st.sidebar.select_slider(
    "Velocidad animación:",
    options=["🐌 Muy Lenta", "🚶 Lenta", "🏃 Normal", "🚀 Rápida", "⚡ Muy Rápida"],
    value="🏃 Normal"
)

velocidad_map = {
    "🐌 Muy Lenta": 0.08,
    "🚶 Lenta": 0.05,
    "🏃 Normal": 0.03,
    "🚀 Rápida": 0.02,
    "⚡ Muy Rápida": 0.01
}

mostrar_marcadores = st.sidebar.checkbox("📍 Marcadores de inversión", value=True)
mostrar_grid = st.sidebar.checkbox("📊 Grid", value=True)
tema_oscuro = st.sidebar.checkbox("🌙 Tema oscuro", value=True)
suavizado = st.sidebar.checkbox("🌊 Suavizado", value=True)

# ============================================================================
# BOTÓN DE ANÁLISIS
# ============================================================================

st.sidebar.markdown("---")

if allocations:
    if st.sidebar.button("🚀 Iniciar Análisis Pro", type="primary", use_container_width=True):
        # Generar fechas de inversión
        fechas_inversion = generar_fechas_inversion(
            fecha_inicio,
            fecha_fin,
            frecuencia=frecuencia_dca,
            dias_especificos=dias_especificos,
            dia_semana=dia_semana,
            fechas_pausa=pausas if usar_pausas else None
        )
        
        # Guardar en session state
        st.session_state.analisis_iniciado = True
        st.session_state.config_analisis = {
            'allocations': allocations,
            'inversion_total': inversion_por_periodo,
            'fechas_inversion': fechas_inversion,
            'lump_sum': lump_sum_amount,
            'aumentos_anuales': aumento_anual,
            'fecha_inicio': fecha_inicio,
            'fecha_fin': fecha_fin,
            'comparar_lump_sum': comparar_lump_sum,
            'usar_monte_carlo': usar_monte_carlo
        }
        st.rerun()

# ============================================================================
# FUNCIONES DE GRAFICACIÓN
# ============================================================================

def crear_grafico_portfolio(resultados, config, tema_oscuro=True, mostrar_grid=True):
    """Crea gráfico principal del portfolio"""
    template = "plotly_dark" if tema_oscuro else "plotly_white"
    
    fig = go.Figure()
    
    # Línea principal de rentabilidad
    rent_data = resultados['rentabilidad'].dropna()
    
    fig.add_trace(go.Scatter(
        x=rent_data.index,
        y=rent_data.values,
        mode='lines',
        name='Portfolio DCA',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>Portfolio</b><br>Fecha: %{x|%d/%m/%Y}<br>Rentabilidad: %{y:.2f}%<extra></extra>'
    ))
    
    # Comparación con lump sum si está habilitada
    if config.get('comparar_lump_sum') and 'lump_sum_comparison' in resultados:
        lump_data = resultados['lump_sum_comparison'].dropna()
        if len(lump_data) > 0:
            fig.add_trace(go.Scatter(
                x=lump_data.index,
                y=lump_data.values,
                mode='lines',
                name='Lump Sum',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                hovertemplate='<b>Lump Sum</b><br>Fecha: %{x|%d/%m/%Y}<br>Rentabilidad: %{y:.2f}%<extra></extra>'
            ))
    
    # Marcadores de inversión
    if mostrar_marcadores and config.get('fechas_inversion'):
        fechas_inv = [f for f in config['fechas_inversion'] if f in rent_data.index]
        if fechas_inv:
            valores_inv = [rent_data.loc[f] for f in fechas_inv]
            
            fig.add_trace(go.Scatter(
                x=fechas_inv[-20:],  # Últimos 20 para no saturar
                y=valores_inv[-20:],
                mode='markers',
                name='Inversiones',
                marker=dict(
                    color='#ffd700',
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>💰 Inversión</b><br>Fecha: %{x|%d/%m/%Y}<extra></extra>'
            ))
    
    # Layout
    fig.update_layout(
        template=template,
        height=500,
        title=dict(
            text=f"📊 Evolución Portfolio DCA - ${config['inversion_total']:,}/período",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title="📅 Fecha",
        yaxis_title="💰 Rentabilidad (%)",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(showgrid=mostrar_grid, gridcolor='rgba(128,128,128,0.1)'),
        yaxis=dict(showgrid=mostrar_grid, gridcolor='rgba(128,128,128,0.1)', ticksuffix="%"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def crear_grafico_composicion(resultados_portfolio, allocations):
    """Crea gráfico de composición del portfolio"""
    detalles = resultados_portfolio.get('detalles_por_ticker', {})
    
    if not detalles:
        return None
    
    # Obtener valores finales
    valores_finales = {}
    for ticker, datos in detalles.items():
        valor_final = datos['valor'].dropna().iloc[-1] if len(datos['valor'].dropna()) > 0 else 0
        valores_finales[ticker] = valor_final
    
    # Crear pie chart
    labels = [f"{ACTIVOS_PREDEFINIDOS.get(t, {}).get('emoji', '📈')} {t}" for t in valores_finales.keys()]
    values = list(valores_finales.values())
    colors = [ACTIVOS_PREDEFINIDOS.get(t, {}).get('color', '#1f77b4') for t in valores_finales.keys()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Valor: $%{value:,.0f}<br>Porcentaje: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="🥧 Composición del Portfolio (Valor Final)",
        height=400,
        showlegend=True
    )
    
    return fig

def crear_grafico_drawdown(resultados_portfolio):
    """Crea gráfico de drawdown"""
    valores = resultados_portfolio['valor_portfolio'].dropna()
    
    if len(valores) < 2:
        return None
    
    # Calcular drawdown
    valores_maximos = valores.cummax()
    drawdown = (valores - valores_maximos) / valores_maximos * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color='#ff6b6b', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.3)',
        hovertemplate='Fecha: %{x|%d/%m/%Y}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="📉 Drawdown del Portfolio",
        xaxis_title="Fecha",
        yaxis_title="Drawdown (%)",
        height=350,
        hovermode='x unified',
        yaxis=dict(ticksuffix="%"),
        showlegend=False
    )
    
    return fig

# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

if 'analisis_iniciado' in st.session_state and st.session_state.analisis_iniciado:
    config = st.session_state.config_analisis
    
    # Descargar datos
    tickers_necesarios = list(config['allocations'].keys())
    precios = descargar_datos_yfinance(tickers_necesarios, config['fecha_inicio'], config['fecha_fin'])
    
    if precios is not None and len(precios) > 0:
        st.success(f"✅ Datos descargados para {len(tickers_necesarios)} activos")
        
        # Preparar diccionario de precios
        precios_dict = {}
        for ticker in tickers_necesarios:
            if ticker in precios.columns:
                precios_dict[ticker] = precios[ticker]
            elif len(tickers_necesarios) == 1:
                precios_dict[ticker] = precios
        
        # Calcular DCA portfolio
        with st.spinner("🔄 Calculando estrategia DCA..."):
            resultados_portfolio = calcular_dca_portfolio_avanzado(precios_dict, config)
        
        # Calcular comparación lump sum si está habilitada
        if config.get('comparar_lump_sum'):
            inversion_total_final = resultados_portfolio['inversion_acumulada'].dropna().iloc[-1] if len(resultados_portfolio['inversion_acumulada'].dropna()) > 0 else 10000
            
            lump_sum_comp = calcular_lump_sum_comparison(
                precios_dict,
                config['allocations'],
                inversion_total_final,
                config['fecha_inicio']
            )
            resultados_portfolio['lump_sum_comparison'] = lump_sum_comp
        
        # Calcular métricas avanzadas
        metricas = calcular_metricas_avanzadas(
            resultados_portfolio['rentabilidad'],
            resultados_portfolio['valor_portfolio'],
            resultados_portfolio['inversion_acumulada']
        )
        
        # ====================================================================
        # VISUALIZACIÓN DE RESULTADOS
        # ====================================================================
        
        st.markdown("## 📊 Resultados del Análisis")
        
        # Métricas principales en cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Rentabilidad Total",
                f"{metricas.get('rentabilidad_total', 0):.2f}%",
                delta=f"${metricas.get('ganancia_neta', 0):,.0f}"
            )
        
        with col2:
            st.metric(
                "💰 Valor Final",
                f"${metricas.get('valor_final', 0):,.0f}"
            )
        
        with col3:
            st.metric(
                "📊 CAGR",
                f"{metricas.get('cagr', 0):.2f}%"
            )
        
        with col4:
            st.metric(
                "⚡ Sharpe Ratio",
                f"{metricas.get('sharpe_ratio', 0):.2f}"
            )
        
        # Métricas adicionales
        st.markdown("### 📈 Métricas Avanzadas")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Invertido", f"${metricas.get('inversion_final', 0):,.0f}")
        
        with col2:
            st.metric("📉 Max Drawdown", f"{metricas.get('max_drawdown', 0):.2f}%")
        
        with col3:
            st.metric("📊 Volatilidad", f"{metricas.get('volatilidad', 0):.2f}%")
        
        with col4:
            st.metric("✅ Días Positivos", f"{metricas.get('porcentaje_tiempo_positivo', 0):.1f}%")
        
        with col5:
            num_inversiones = len(config.get('fechas_inversion', []))
            st.metric("🔢 # Inversiones", f"{num_inversiones}")
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Evolución", "🥧 Composición", "📊 Análisis", "💾 Exportar"])
        
        with tab1:
            # Gráfico principal
            fig_main = crear_grafico_portfolio(resultados_portfolio, config, tema_oscuro, mostrar_grid)
            st.plotly_chart(fig_main, use_container_width=True)
            
            # Comparación DCA vs Lump Sum
            if config.get('comparar_lump_sum') and 'lump_sum_comparison' in resultados_portfolio:
                st.markdown("### 🔄 Comparación: DCA vs Lump Sum")
                
                lump_rent_final = resultados_portfolio['lump_sum_comparison'].dropna().iloc[-1] if len(resultados_portfolio['lump_sum_comparison'].dropna()) > 0 else 0
                dca_rent_final = metricas.get('rentabilidad_total', 0)
                diferencia = dca_rent_final - lump_rent_final
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("DCA", f"{dca_rent_final:.2f}%")
                
                with col2:
                    st.metric("Lump Sum", f"{lump_rent_final:.2f}%")
                
                with col3:
                    st.metric("Diferencia", f"{diferencia:+.2f}%", 
                             delta="DCA mejor" if diferencia > 0 else "Lump Sum mejor")
        
        with tab2:
            # Gráfico de composición
            fig_comp = crear_grafico_composicion(resultados_portfolio, config['allocations'])
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True)
            
            # Tabla detallada por activo
            st.markdown("### 📋 Detalle por Activo")
            
            detalles_tabla = []
            for ticker, datos in resultados_portfolio.get('detalles_por_ticker', {}).items():
                valor_final = datos['valor'].dropna().iloc[-1] if len(datos['valor'].dropna()) > 0 else 0
                acciones_final = datos['acciones'].dropna().iloc[-1] if len(datos['acciones'].dropna()) > 0 else 0
                
                # Calcular precio promedio de compra
                total_invertido_ticker = config['allocations'][ticker] / 100 * metricas.get('inversion_final', 0)
                precio_promedio = total_invertido_ticker / acciones_final if acciones_final > 0 else 0
                
                # Precio actual
                if ticker in precios_dict:
                    precio_actual = precios_dict[ticker].dropna().iloc[-1] if len(precios_dict[ticker].dropna()) > 0 else 0
                else:
                    precio_actual = 0
                
                rentabilidad_ticker = ((precio_actual / precio_promedio) - 1) * 100 if precio_promedio > 0 else 0
                
                emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                
                detalles_tabla.append({
                    'Activo': f"{emoji} {ticker}",
                    'Acciones': f"{acciones_final:.4f}",
                    'Valor Final': f"${valor_final:,.2f}",
                    'Precio Promedio': f"${precio_promedio:.2f}",
                    'Precio Actual': f"${precio_actual:.2f}",
                    'Rentabilidad': f"{rentabilidad_ticker:+.2f}%",
                    'Asignación': f"{config['allocations'][ticker]}%"
                })
            
            df_detalles = pd.DataFrame(detalles_tabla)
            st.dataframe(df_detalles, use_container_width=True, hide_index=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de drawdown
                fig_dd = crear_grafico_drawdown(resultados_portfolio)
                if fig_dd:
                    st.plotly_chart(fig_dd, use_container_width=True)
            
            with col2:
                # Resumen de métricas
                st.markdown("### 📊 Resumen de Métricas")
                
                st.info(f"""
                **Rendimiento:**
                - Rentabilidad Total: {metricas.get('rentabilidad_total', 0):.2f}%
                - CAGR: {metricas.get('cagr', 0):.2f}%
                - Ganancia Neta: ${metricas.get('ganancia_neta', 0):,.0f}
                
                **Riesgo:**
                - Volatilidad: {metricas.get('volatilidad', 0):.2f}%
                - Max Drawdown: {metricas.get('max_drawdown', 0):.2f}%
                - Sharpe Ratio: {metricas.get('sharpe_ratio', 0):.2f}
                
                **Consistencia:**
                - Tiempo en positivo: {metricas.get('porcentaje_tiempo_positivo', 0):.1f}%
                - Mejor año: {metricas.get('mejor_año', 0):+.2f}%
                - Peor año: {metricas.get('peor_año', 0):+.2f}%
                """)
            
            # Tabla de fechas de inversión
            st.markdown("### 📅 Calendario de Inversiones")
            
            fechas_inv = config.get('fechas_inversion', [])
            if fechas_inv:
                # Agrupar por año y mes
                fechas_por_periodo = {}
                for fecha in fechas_inv[:50]:  # Mostrar primeras 50
                    periodo = fecha.strftime('%Y-%m')
                    if periodo not in fechas_por_periodo:
                        fechas_por_periodo[periodo] = []
                    fechas_por_periodo[periodo].append(fecha.strftime('%d/%m/%Y'))
                
                for periodo, fechas in list(fechas_por_periodo.items())[:12]:
                    st.markdown(f"**{periodo}:** {', '.join(fechas)}")
                
                if len(fechas_inv) > 50:
                    st.info(f"... y {len(fechas_inv) - 50} fechas más")
        
        with tab4:
            st.markdown("### 💾 Exportar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Exportar a Excel
                excel_data = exportar_a_excel(resultados_portfolio, metricas)
                
                st.download_button(
                    label="📥 Descargar Excel Completo",
                    data=excel_data,
                    file_name=f"analisis_dca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Exportar configuración
                config_json = json.dumps({
                    'allocations': config['allocations'],
                    'inversion_por_periodo': config['inversion_total'],
                    'lump_sum': config['lump_sum'],
                    'aumentos_anuales': config['aumentos_anuales'],
                    'fecha_inicio': str(config['fecha_inicio']),
                    'fecha_fin': str(config['fecha_fin'])
                }, indent=2)
                
                st.download_button(
                    label="📋 Descargar Configuración (JSON)",
                    data=config_json,
                    file_name=f"config_dca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Preview de datos
            st.markdown("### 👀 Vista Previa de Datos")
            
            df_preview = pd.DataFrame({
                'Fecha': resultados_portfolio['rentabilidad'].index[-20:],
                'Rentabilidad (%)': resultados_portfolio['rentabilidad'].values[-20:],
                'Valor ($)': resultados_portfolio['valor_portfolio'].values[-20:],
                'Inversión ($)': resultados_portfolio['inversion_acumulada'].values[-20:]
            })
            
            st.dataframe(df_preview, use_container_width=True, hide_index=True)
    
    else:
        st.error("❌ No se pudieron descargar los datos. Verifica los tickers y el rango de fechas.")

else:
    # Pantalla de bienvenida
    st.markdown("""
    <div style="text-align: center; padding: 3rem; margin: 2rem 0;">
        <h2 style="color: #667eea;">🎯 Bienvenido al Analizador DCA Pro v3.0</h2>
        <p style="font-size: 1.2rem; color: #888; margin: 1rem 0;">
            Analiza estrategias de Dollar Cost Averaging con configuración avanzada
        </p>
        <div style="margin: 2rem 0;">
            <span style="font-size: 3rem;">📈💼🚀</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Características nuevas
    st.markdown("### ✨ Nuevas Características v3.0")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="portfolio-preset">
            <h4>🗓️ Scheduling Flexible</h4>
            <ul>
                <li>Inversión diaria, semanal, quincenal</li>
                <li>Días específicos del mes</li>
                <li>Períodos de pausa</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="portfolio-preset">
            <h4>💼 Portfolio Management</h4>
            <ul>
                <li>Asignaciones personalizadas</li>
                <li>Portfolios predefinidos</li>
                <li>Rebalanceo automático</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="portfolio-preset">
            <h4>📊 Analytics Avanzado</h4>
            <ul>
                <li>Sharpe Ratio & Volatilidad</li>
                <li>DCA vs Lump Sum</li>
                <li>Exportar a Excel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Ejemplos de portfolios
    st.markdown("### 🌟 Portfolios Predefinidos")
    
    cols = st.columns(len(PORTFOLIO_PRESETS))
    
    for i, (nombre, datos) in enumerate(PORTFOLIO_PRESETS.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border-radius: 10px; border: 2px solid #667eea; background: rgba(102, 126, 234, 0.05);">
                <h4>{nombre}</h4>
                <p style="font-size: 0.9rem; color: #888;">{datos['descripcion']}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>📊 <strong>BQuant-DCA Pro Visualizer v3.0</strong> | Advanced Portfolio Analysis</p>
    <p>💡 Desarrollado por @Gsnchez | Datos por Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
