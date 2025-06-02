import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import time
import json

# Configuración de la página
st.set_page_config(
    page_title="DCA Evolution Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejor apariencia
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
    .animation-controls {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con estilo
st.markdown('<h1 class="main-header">📈 DCA Evolution Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualiza la evolución día a día de tu estrategia Dollar Cost Averaging</p>', unsafe_allow_html=True)

# Diccionario de activos con colores vibrantes
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

# Función para descargar datos de yfinance
@st.cache_data(ttl=3600)
def descargar_datos_yfinance(tickers, fecha_inicio, fecha_fin):
    """Descarga datos históricos de Yahoo Finance"""
    try:
        with st.spinner(f"🔄 Descargando datos de {len(tickers)} activos..."):
            datos = yf.download(tickers, start=fecha_inicio, end=fecha_fin, progress=False, multi_level_index=False)
            if len(tickers) == 1:
                datos.columns = pd.MultiIndex.from_product([datos.columns, [tickers[0]]])
            
            precios = datos['Close'].fillna(method='ffill').fillna(method='bfill')
            return precios
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
        return None

# Función para calcular DCA correctamente
def calcular_dca_detallado(precios, inversion_mensual=100):
    """Calcula rentabilidad DCA día a día correctamente"""
    if precios is None or len(precios) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Limpiar datos de precios
    precios_clean = precios.dropna()
    if len(precios_clean) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Crear fechas de inversión mensual (primer día hábil de cada mes)
    fechas_inversion = []
    fecha_actual = precios_clean.index[0]
    
    while fecha_actual <= precios_clean.index[-1]:
        # Buscar el primer día del mes que tenga datos
        primer_dia_mes = fecha_actual.replace(day=1)
        
        # Encontrar el primer día con datos en este mes
        dias_mes = precios_clean.index[
            (precios_clean.index >= primer_dia_mes) & 
            (precios_clean.index < primer_dia_mes + pd.DateOffset(months=1))
        ]
        
        if len(dias_mes) > 0:
            fechas_inversion.append(dias_mes[0])
        
        # Siguiente mes
        fecha_actual = primer_dia_mes + pd.DateOffset(months=1)
    
    if len(fechas_inversion) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Inicializar series para resultados
    acciones_totales = 0.0
    inversion_total = 0.0
    
    # Series para almacenar resultados día a día
    rentabilidad_diaria = pd.Series(index=precios_clean.index, dtype=float)
    valor_portfolio_diario = pd.Series(index=precios_clean.index, dtype=float)
    inversion_acumulada_diaria = pd.Series(index=precios_clean.index, dtype=float)
    
    # Calcular DCA día a día
    for fecha in precios_clean.index:
        # Verificar si es día de inversión
        if fecha in fechas_inversion:
            precio_compra = precios_clean.loc[fecha]
            if precio_compra > 0:
                # Comprar acciones con la inversión mensual
                nuevas_acciones = inversion_mensual / precio_compra
                acciones_totales += nuevas_acciones
                inversion_total += inversion_mensual
        
        # Calcular valor del portfolio hoy
        precio_actual = precios_clean.loc[fecha]
        valor_portfolio_hoy = acciones_totales * precio_actual
        
        # Calcular rentabilidad acumulada
        if inversion_total > 0:
            rentabilidad_hoy = ((valor_portfolio_hoy / inversion_total) - 1) * 100
        else:
            rentabilidad_hoy = 0.0
        
        # Guardar valores
        rentabilidad_diaria.loc[fecha] = rentabilidad_hoy
        valor_portfolio_diario.loc[fecha] = valor_portfolio_hoy
        inversion_acumulada_diaria.loc[fecha] = inversion_total
    
    # Reindexar a toda la serie original (rellenar valores faltantes)
    rentabilidad_completa = rentabilidad_diaria.reindex(precios.index).fillna(method='ffill').fillna(0)
    valor_completo = valor_portfolio_diario.reindex(precios.index).fillna(method='ffill').fillna(0)
    inversion_completa = inversion_acumulada_diaria.reindex(precios.index).fillna(method='ffill').fillna(0)
    
    return rentabilidad_completa, valor_completo, inversion_completa

# Sidebar - Configuración elegante
st.sidebar.markdown("## ⚙️ Configuración de Análisis")

# Selector de fechas con diseño mejorado
st.sidebar.markdown("### 📅 Rango de Fechas")
col1, col2 = st.sidebar.columns(2)
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

# Inversión mensual con slider más visual
st.sidebar.markdown("### 💰 Inversión Mensual")
inversion_mensual = st.sidebar.select_slider(
    "Cantidad ($):",
    options=[50, 100, 200, 500, 1000, 2000, 5000],
    value=100,
    format_func=lambda x: f"${x:,}"
)

# Selección de activos con mejor UX
st.sidebar.markdown("### 🎯 Selección de Activos")

# Activos predefinidos con emojis
activos_con_emoji = {
    ticker: f"{info['emoji']} {ticker} - {info['nombre'][:20]}..." 
    for ticker, info in ACTIVOS_PREDEFINIDOS.items()
}

activos_seleccionados = st.sidebar.multiselect(
    "Activos a analizar:",
    options=list(ACTIVOS_PREDEFINIDOS.keys()),
    default=['SPY', 'QQQ', 'VTI'],
    format_func=lambda x: activos_con_emoji[x]
)

# Activos personalizados
with st.sidebar.expander("➕ Agregar activos personalizados"):
    activos_custom = st.text_input(
        "Tickers (separados por comas):",
        placeholder="TSLA, NVDA, AMZN"
    )
    
    if activos_custom:
        custom_list = [ticker.strip().upper() for ticker in activos_custom.split(',') if ticker.strip()]
        activos_seleccionados.extend(custom_list)

# Configuración de animación
st.sidebar.markdown("### 🎬 Controles de Animación")
velocidad_animacion = st.sidebar.select_slider(
    "Velocidad:",
    options=["🐌 Muy Lenta", "🚶 Lenta", "🏃 Normal", "🚀 Rápida", "⚡ Muy Rápida"],
    value="🏃 Normal"
)

# Mapear velocidades a delays
velocidad_map = {
    "🐌 Muy Lenta": 0.2,
    "🚶 Lenta": 0.1,
    "🏃 Normal": 0.05,
    "🚀 Rápida": 0.02,
    "⚡ Muy Rápida": 0.01
}

# Opciones de visualización
mostrar_marcadores_inversion = st.sidebar.checkbox("📍 Mostrar días de inversión", value=True)
mostrar_grid = st.sidebar.checkbox("📊 Mostrar grid", value=True)
tema_oscuro = st.sidebar.checkbox("🌙 Tema oscuro", value=True)

# Botón principal de análisis
if len(activos_seleccionados) > 0:
    if st.sidebar.button("🚀 Iniciar Análisis", type="primary"):
        st.session_state.analisis_iniciado = True
        st.session_state.activos_analizar = activos_seleccionados.copy()
        st.session_state.fecha_inicio_analisis = fecha_inicio
        st.session_state.fecha_fin_analisis = fecha_fin
        st.session_state.inversion_analisis = inversion_mensual

# Área principal
if 'analisis_iniciado' in st.session_state and st.session_state.analisis_iniciado:
    # Descargar datos
    precios = descargar_datos_yfinance(
        st.session_state.activos_analizar, 
        st.session_state.fecha_inicio_analisis, 
        st.session_state.fecha_fin_analisis
    )
    
    if precios is not None:
        st.success(f"✅ Datos descargados exitosamente para {len(st.session_state.activos_analizar)} activos")
        
        # Calcular rentabilidades para todos los activos
        rentabilidades = {}
        valores_portfolio = {}
        inversiones_acumuladas = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(st.session_state.activos_analizar):
            status_text.text(f"Calculando DCA para {ticker}...")
            
            if ticker in precios.columns:
                rent, valor, inversion = calcular_dca_detallado(
                    precios[ticker], 
                    st.session_state.inversion_analisis
                )
                
                if len(rent.dropna()) > 0:
                    rentabilidades[ticker] = rent
                    valores_portfolio[ticker] = valor
                    inversiones_acumuladas[ticker] = inversion
            
            progress_bar.progress((i + 1) / len(st.session_state.activos_analizar))
        
        progress_bar.empty()
        status_text.empty()
        
        if rentabilidades:
            # Crear el gráfico principal
            st.markdown("## 📈 Evolución de Rentabilidad DCA")
            
            # Controles de animación
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            
            with col1:
                if st.button("▶️ Reproducir", type="primary"):
                    st.session_state.reproducir_animacion = True
            
            with col2:
                if st.button("⏸️ Pausar"):
                    st.session_state.reproducir_animacion = False
            
            with col3:
                if st.button("📊 Ver Final"):
                    st.session_state.mostrar_final = True
            
            with col4:
                mostrar_valores = st.checkbox("💵 Mostrar valores en $", value=False)
            
            # Área del gráfico
            chart_container = st.empty()
            info_container = st.empty()
            
            # Obtener todas las fechas para la animación
            todas_fechas = sorted(set().union(*[rent.index for rent in rentabilidades.values()]))
            
            # Función para crear gráfico animado
            def crear_grafico_animado(hasta_fecha=None, mostrar_valores_abs=False):
                fig = go.Figure()
                
                # Determinar datos a mostrar
                if hasta_fecha:
                    # Filtrar datos hasta la fecha especificada
                    datos_filtrados = {}
                    for ticker in rentabilidades.keys():
                        datos_completos = rentabilidades[ticker].loc[rentabilidades[ticker].index <= hasta_fecha]
                        if len(datos_completos) > 0:
                            datos_filtrados[ticker] = datos_completos
                else:
                    datos_filtrados = rentabilidades
                
                # Añadir líneas para cada activo
                for ticker, datos_ticker in datos_filtrados.items():
                    if len(datos_ticker.dropna()) > 0:
                        # Configurar color y estilo
                        color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
                        emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                        
                        # Datos a mostrar (rentabilidad o valores absolutos)
                        if mostrar_valores_abs and ticker in valores_portfolio:
                            if hasta_fecha:
                                y_data = valores_portfolio[ticker].loc[valores_portfolio[ticker].index <= hasta_fecha]
                            else:
                                y_data = valores_portfolio[ticker]
                            y_title = "Valor Portfolio ($)"
                            hover_format = "$%{y:,.0f}"
                            yaxis_format = ".0f"
                        else:
                            y_data = datos_ticker
                            y_title = "Rentabilidad Acumulada (%)"
                            hover_format = "%{y:.1f}%"
                            yaxis_format = ".1f"
                        
                        # Limpiar datos para evitar valores extraños
                        y_data_clean = y_data.dropna()
                        if len(y_data_clean) == 0:
                            continue
                        
                        # Línea principal con interpolación suave
                        fig.add_trace(go.Scatter(
                            x=y_data_clean.index,
                            y=y_data_clean.values,
                            mode='lines',
                            name=f'{emoji} {ticker}',
                            line=dict(
                                color=color,
                                width=3,
                                smoothing=1.3  # Suavizado de líneas
                            ),
                            hovertemplate=f'<b>{ticker}</b><br>' +
                                        'Fecha: %{x|%d/%m/%Y}<br>' +
                                        f'Valor: {hover_format}<br>' +
                                        '<extra></extra>',
                            connectgaps=True
                        ))
                        
                        # Añadir marcadores en días de inversión si está habilitado
                        if mostrar_marcadores_inversion and hasta_fecha:
                            # Encontrar días de inversión (buscar saltos en inversión acumulada)
                            if ticker in inversiones_acumuladas:
                                inversiones_ticker = inversiones_acumuladas[ticker].loc[inversiones_acumuladas[ticker].index <= hasta_fecha]
                                # Detectar días donde aumentó la inversión
                                diff_inversiones = inversiones_ticker.diff()
                                dias_inversion_idx = diff_inversiones[diff_inversiones > 0].index
                                
                                if len(dias_inversion_idx) > 0:
                                    valores_inversion = y_data_clean.reindex(dias_inversion_idx).dropna()
                                    
                                    if len(valores_inversion) > 0:
                                        fig.add_trace(go.Scatter(
                                            x=valores_inversion.index,
                                            y=valores_inversion.values,
                                            mode='markers',
                                            name=f'{ticker} (Compras)',
                                            marker=dict(
                                                color=color,
                                                size=10,
                                                symbol='circle',
                                                line=dict(color='white', width=2)
                                            ),
                                            showlegend=False,
                                            hovertemplate=f'<b>{ticker} - Compra Mensual</b><br>' +
                                                        'Fecha: %{x|%d/%m/%Y}<br>' +
                                                        f'Valor: {hover_format}<br>' +
                                                        f'💰 ${st.session_state.inversion_analisis:,} invertidos<br>' +
                                                        '<extra></extra>'
                                        ))
                
                # Configurar layout mejorado
                template = "plotly_dark" if tema_oscuro else "plotly_white"
                
                # Configurar formato del eje Y basado en los datos
                if mostrar_valores_abs:
                    yaxis_config = dict(
                        tickformat="$,.0f",
                        showgrid=mostrar_grid,
                        gridcolor='rgba(128,128,128,0.2)'
                    )
                else:
                    yaxis_config = dict(
                        tickformat=".1f",
                        ticksuffix="%",
                        showgrid=mostrar_grid,
                        gridcolor='rgba(128,128,128,0.2)'
                    )
                
                fig.update_layout(
                    title=dict(
                        text=f"📊 Evolución DCA - Inversión Mensual: ${st.session_state.inversion_analisis:,}",
                        x=0.5,
                        font=dict(size=20, color='white' if tema_oscuro else 'black')
                    ),
                    xaxis_title="📅 Fecha",
                    yaxis_title=f"💰 {y_title}",
                    template=template,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(0,0,0,0.8)" if tema_oscuro else "rgba(255,255,255,0.8)"
                    ),
                    height=600,
                    showlegend=True,
                    xaxis=dict(
                        showgrid=mostrar_grid,
                        gridcolor='rgba(128,128,128,0.2)',
                        tickformat='%b %Y'
                    ),
                    yaxis=yaxis_config,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=60, r=60, t=80, b=60)
                )
                
                # Añadir anotación con fecha actual si es animación
                if hasta_fecha:
                    fig.add_annotation(
                        text=f"📅 {hasta_fecha.strftime('%d/%m/%Y')}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=16, color="yellow"),
                        bgcolor="rgba(0,0,0,0.7)",
                        bordercolor="yellow",
                        borderwidth=1,
                        borderpad=8
                    )
                
                return fig
            
            # Animación o gráfico estático
            if 'reproducir_animacion' in st.session_state and st.session_state.reproducir_animacion:
                st.session_state.reproducir_animacion = False  # Reset
                
                # Obtener fechas comunes a todos los activos
                fechas_comunes = None
                for ticker in rentabilidades.keys():
                    fechas_ticker = rentabilidades[ticker].dropna().index
                    if fechas_comunes is None:
                        fechas_comunes = fechas_ticker
                    else:
                        fechas_comunes = fechas_comunes.intersection(fechas_ticker)
                
                if len(fechas_comunes) == 0:
                    st.error("No hay fechas comunes entre los activos seleccionados")
                    return
                
                fechas_comunes = sorted(fechas_comunes)
                
                # Crear un sampling inteligente para la animación
                total_fechas = len(fechas_comunes)
                if total_fechas > 150:
                    # Si hay muchas fechas, hacer sampling
                    step = max(1, total_fechas // 150)
                    fechas_animacion = fechas_comunes[::step]
                    # Asegurar que incluimos la última fecha
                    if fechas_comunes[-1] not in fechas_animacion:
                        fechas_animacion.append(fechas_comunes[-1])
                else:
                    fechas_animacion = fechas_comunes
                
                st.info(f"🎬 Iniciando animación con {len(fechas_animacion)} frames...")
                progress_animacion = st.progress(0)
                
                for i, fecha_actual in enumerate(fechas_animacion):
                    # Crear y mostrar gráfico
                    fig = crear_grafico_animado(fecha_actual, mostrar_valores)
                    chart_container.plotly_chart(fig, use_container_width=True, key=f"anim_{i}")
                    
                    # Información en tiempo real mejorada
                    rentabilidades_actuales = {}
                    valores_actuales = {}
                    inversiones_actuales = {}
                    
                    for ticker in rentabilidades.keys():
                        # Buscar el valor más cercano a la fecha actual
                        ticker_data = rentabilidades[ticker].loc[rentabilidades[ticker].index <= fecha_actual]
                        if len(ticker_data) > 0:
                            rentabilidades_actuales[ticker] = ticker_data.iloc[-1]
                        
                        if ticker in valores_portfolio:
                            valor_data = valores_portfolio[ticker].loc[valores_portfolio[ticker].index <= fecha_actual]
                            if len(valor_data) > 0:
                                valores_actuales[ticker] = valor_data.iloc[-1]
                        
                        if ticker in inversiones_acumuladas:
                            inv_data = inversiones_acumuladas[ticker].loc[inversiones_acumuladas[ticker].index <= fecha_actual]
                            if len(inv_data) > 0:
                                inversiones_actuales[ticker] = inv_data.iloc[-1]
                    
                    # Mostrar métricas actuales en formato mejorado
                    if rentabilidades_actuales:
                        cols_info = st.columns(min(len(rentabilidades_actuales), 4))
                        
                        for j, ticker in enumerate(rentabilidades_actuales.keys()):
                            with cols_info[j % 4]:
                                emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                                color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
                                
                                rent_val = rentabilidades_actuales[ticker]
                                valor_val = valores_actuales.get(ticker, 0)
                                inv_val = inversiones_actuales.get(ticker, 0)
                                
                                # Determinar color de la métrica
                                color_delta = "normal"
                                if rent_val > 0:
                                    color_delta = "normal"
                                elif rent_val < 0:
                                    color_delta = "inverse"
                                
                                if mostrar_valores:
                                    st.metric(
                                        label=f"{emoji} {ticker}",
                                        value=f"${valor_val:,.0f}",
                                        delta=f"{rent_val:+.1f}%",
                                        delta_color=color_delta
                                    )
                                else:
                                    st.metric(
                                        label=f"{emoji} {ticker}",
                                        value=f"{rent_val:.1f}%",
                                        delta=f"${valor_val:,.0f}" if valor_val > 0 else None,
                                        delta_color=color_delta
                                    )
                    
                    # Actualizar progreso
                    progress_animacion.progress((i + 1) / len(fechas_animacion))
                    time.sleep(velocidad_map[velocidad_animacion])
                
                progress_animacion.empty()
                st.success("🎉 ¡Animación completada!")
                
            elif 'mostrar_final' in st.session_state and st.session_state.mostrar_final:
                st.session_state.mostrar_final = False  # Reset
                fig = crear_grafico_animado(mostrar_valores_abs=mostrar_valores)
                chart_container.plotly_chart(fig, use_container_width=True)
            
            # Métricas finales elegantes
            st.markdown("## 🏆 Resultados Finales")
            
            # Preparar datos para métricas
            metricas_finales = []
            
            for ticker in rentabilidades.keys():
                rentabilidad_final = rentabilidades[ticker].dropna()
                valor_portfolio_final = valores_portfolio[ticker].dropna() if ticker in valores_portfolio else pd.Series()
                inversion_total_final = inversiones_acumuladas[ticker].dropna() if ticker in inversiones_acumuladas else pd.Series()
                
                if len(rentabilidad_final) > 0:
                    rent_final_val = rentabilidad_final.iloc[-1]
                    valor_final_val = valor_portfolio_final.iloc[-1] if len(valor_portfolio_final) > 0 else 0
                    inv_final_val = inversion_total_final.iloc[-1] if len(inversion_total_final) > 0 else 0
                    
                    # Calcular ganancia/pérdida neta
                    ganancia_neta = valor_final_val - inv_final_val
                    
                    metricas_finales.append({
                        'ticker': ticker,
                        'rentabilidad': rent_final_val,
                        'valor_final': valor_final_val,
                        'inversion_total': inv_final_val,
                        'ganancia_neta': ganancia_neta
                    })
            
            # Ordenar por rentabilidad (mejor primero)
            metricas_finales.sort(key=lambda x: x['rentabilidad'], reverse=True)
            
            # Mostrar métricas en columnas
            if len(metricas_finales) <= 3:
                cols_metricas = st.columns(len(metricas_finales))
            else:
                # Si hay más de 3, hacer filas de 3
                for i in range(0, len(metricas_finales), 3):
                    cols_metricas = st.columns(min(3, len(metricas_finales) - i))
                    grupo_metricas = metricas_finales[i:i+3]
                    
                    for j, metrica in enumerate(grupo_metricas):
                        ticker = metrica['ticker']
                        with cols_metricas[j]:
                            emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                            color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
                            
                            # Determinar color del fondo basado en performance
                            if metrica['rentabilidad'] >= 20:
                                bg_intensity = "40"  # Verde fuerte
                                text_color = "#00C851"
                            elif metrica['rentabilidad'] >= 10:
                                bg_intensity = "30"  # Verde medio
                                text_color = "#00C851"
                            elif metrica['rentabilidad'] >= 0:
                                bg_intensity = "20"  # Verde suave
                                text_color = "#00C851"
                            elif metrica['rentabilidad'] >= -10:
                                bg_intensity = "20"  # Naranja suave
                                text_color = "#FF8800"
                            else:
                                bg_intensity = "30"  # Rojo
                                text_color = "#FF4444"
                            
                            # Card personalizada con métricas mejoradas
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, {color}{bg_intensity}, {color}10);
                                border-left: 4px solid {color};
                                padding: 1.2rem;
                                border-radius: 12px;
                                margin: 0.5rem 0;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            ">
                                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{emoji}</span>
                                    <h3 style="margin: 0; color: {color}; font-weight: bold;">{ticker}</h3>
                                </div>
                                
                                <div style="margin: 0.8rem 0;">
                                    <p style="font-size: 2rem; font-weight: bold; margin: 0; color: {text_color};">
                                        {metrica['rentabilidad']:+.1f}%
                                    </p>
                                </div>
                                
                                <div style="font-size: 0.9rem; opacity: 0.9; line-height: 1.4;">
                                    <p style="margin: 0.2rem 0;">
                                        💰 <strong>Valor Final:</strong> ${metrica['valor_final']:,.0f}
                                    </p>
                                    <p style="margin: 0.2rem 0;">
                                        📊 <strong>Total Invertido:</strong> ${metrica['inversion_total']:,.0f}
                                    </p>
                                    <p style="margin: 0.2rem 0;">
                                        {'🎉' if metrica['ganancia_neta'] >= 0 else '😔'} <strong>Ganancia:</strong> <span style="color: {text_color};">${metrica['ganancia_neta']:+,.0f}</span>
                                    </p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if i + 3 < len(metricas_finales):
                        st.markdown("")  # Espacio entre filas
            
            # Resumen general
            if len(metricas_finales) > 1:
                st.markdown("### 📈 Resumen del Portfolio")
                
                # Calcular métricas del portfolio conjunto
                valor_total_final = sum([m['valor_final'] for m in metricas_finales])
                inversion_total_final = sum([m['inversion_total'] for m in metricas_finales])
                ganancia_total = valor_total_final - inversion_total_final
                rentabilidad_portfolio = ((valor_total_final / inversion_total_final) - 1) * 100 if inversion_total_final > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Rentabilidad Total",
                        f"{rentabilidad_portfolio:.1f}%",
                        f"${ganancia_total:+,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "💰 Valor Final",
                        f"${valor_total_final:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        "📊 Total Invertido",
                        f"${inversion_total_final:,.0f}"
                    )
                
                with col4:
                    mejor_activo = max(metricas_finales, key=lambda x: x['rentabilidad'])
                    st.metric(
                        "🏆 Mejor Activo",
                        mejor_activo['ticker'],
                        f"{mejor_activo['rentabilidad']:+.1f}%"
                    )
            
        else:
            st.error("❌ No se pudieron calcular las rentabilidades para ningún activo.")
            st.info("💡 Verifica que los tickers sean válidos y que haya datos disponibles para el rango de fechas seleccionado.")

else:
    # Pantalla de bienvenida elegante
    st.markdown("""
    <div style="text-align: center; padding: 3rem; margin: 2rem 0;">
        <h2 style="color: #667eea;">🎯 ¿Listo para visualizar tu estrategia DCA?</h2>
        <p style="font-size: 1.2rem; color: #888; margin: 1rem 0;">
            Selecciona tus activos favoritos en el panel lateral y haz clic en "Iniciar Análisis" 
            para ver una increíble animación de cómo evoluciona tu inversión día a día.
        </p>
        <div style="margin: 2rem 0;">
            <span style="font-size: 3rem;">📈📊💰</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar algunos ejemplos de activos disponibles
    st.markdown("### 🌟 Algunos activos populares disponibles:")
    
    cols_ejemplos = st.columns(5)
    activos_ejemplo = list(ACTIVOS_PREDEFINIDOS.items())[:5]
    
    for i, (ticker, info) in enumerate(activos_ejemplo):
        with cols_ejemplos[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border-radius: 10px; border: 1px solid {info['color']};">
                <div style="font-size: 2rem;">{info['emoji']}</div>
                <div style="font-weight: bold; color: {info['color']};">{ticker}</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">{info['nombre'][:15]}...</div>
            </div>
            """, unsafe_allow_html=True)

# Footer elegante
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>📊 <strong>DCA Evolution Visualizer</strong> | Powered by Yahoo Finance</p>
    <p>💡 Visualiza el poder del Dollar Cost Averaging con datos reales</p>
</div>
""", unsafe_allow_html=True)
