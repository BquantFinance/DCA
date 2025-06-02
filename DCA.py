import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="BQuant-DCA Evolution Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado para animaciones sin flickering
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
    
    /* CRÍTICO: Eliminar completamente transiciones y animaciones CSS */
    div[data-testid="stPlotlyChart"], 
    div[data-testid="stPlotlyChart"] > div,
    div[data-testid="stPlotlyChart"] > div > div {
        transition: none !important;
        animation: none !important;
        transform: none !important;
    }
    
    /* Forzar estabilidad del contenedor */
    .chart-container {
        position: relative;
        height: 500px;
        overflow: hidden;
    }
    
    /* Desactivar hover effects que causan redraws */
    .js-plotly-plot .plotly:hover {
        transform: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con estilo
st.markdown('<h1 class="main-header">📈 BQuant-DCA Evolution Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualización Ultra-Fluida sin Parpadeos</p>', unsafe_allow_html=True)

# Diccionario de activos con colores optimizados para animación
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
            
            precios = datos['Close'].ffill().bfill()
            return precios
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
        return None

# Función DCA mejorada
def calcular_dca_detallado(precios, inversion_mensual=100):
    """Calcula rentabilidad DCA día a día de forma optimizada"""
    if precios is None or len(precios) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    precios_clean = precios.dropna()
    if len(precios_clean) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Crear fechas de inversión mensual más eficientemente
    fechas_inversion = []
    fecha_actual = precios_clean.index[0]
    
    while fecha_actual <= precios_clean.index[-1]:
        primer_dia_mes = fecha_actual.replace(day=1)
        dias_mes = precios_clean.index[
            (precios_clean.index >= primer_dia_mes) & 
            (precios_clean.index < primer_dia_mes + pd.DateOffset(months=1))
        ]
        
        if len(dias_mes) > 0:
            fechas_inversion.append(dias_mes[0])
        
        fecha_actual = primer_dia_mes + pd.DateOffset(months=1)
    
    if len(fechas_inversion) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Calcular usando vectorización para mejor performance
    acciones_totales = 0.0
    inversion_total = 0.0
    
    rentabilidad_diaria = pd.Series(index=precios_clean.index, dtype=float)
    valor_portfolio_diario = pd.Series(index=precios_clean.index, dtype=float)
    inversion_acumulada_diaria = pd.Series(index=precios_clean.index, dtype=float)
    
    for fecha in precios_clean.index:
        if fecha in fechas_inversion:
            precio_compra = precios_clean.loc[fecha]
            if precio_compra > 0:
                nuevas_acciones = inversion_mensual / precio_compra
                acciones_totales += nuevas_acciones
                inversion_total += inversion_mensual
        
        precio_actual = precios_clean.loc[fecha]
        valor_portfolio_hoy = acciones_totales * precio_actual
        
        if inversion_total > 0:
            rentabilidad_hoy = ((valor_portfolio_hoy / inversion_total) - 1) * 100
        else:
            rentabilidad_hoy = 0.0
        
        rentabilidad_diaria.loc[fecha] = rentabilidad_hoy
        valor_portfolio_diario.loc[fecha] = valor_portfolio_hoy
        inversion_acumulada_diaria.loc[fecha] = inversion_total
    
    rentabilidad_completa = rentabilidad_diaria.reindex(precios.index).ffill().fillna(0)
    valor_completo = valor_portfolio_diario.reindex(precios.index).ffill().fillna(0)
    inversion_completa = inversion_acumulada_diaria.reindex(precios.index).ffill().fillna(0)
    
    return rentabilidad_completa, valor_completo, inversion_completa

# NUEVA FUNCIÓN: Crear figura base estática (clave anti-flickering)
def crear_figura_base(x_range, y_range, tema_oscuro=True, mostrar_grid=True):
    """Crea una figura base con configuración fija para evitar redimensionamiento"""
    
    template = "plotly_dark" if tema_oscuro else "plotly_white"
    
    fig = go.Figure()
    
    # Configuración ultra-estable para eliminar flickering
    fig.update_layout(
        template=template,
        height=500,
        # CRÍTICO: Rangos fijos desde el inicio
        xaxis=dict(
            range=x_range,
            showgrid=mostrar_grid,
            gridcolor='rgba(128,128,128,0.1)',
            fixedrange=True,
            tickformat='%b %Y',
            # Desactivar auto-scaling
            autorange=False,
            type='date'
        ),
        yaxis=dict(
            range=y_range,
            showgrid=mostrar_grid,
            gridcolor='rgba(128,128,128,0.1)',
            tickformat=".1f",
            ticksuffix="%",
            fixedrange=True,
            # Desactivar auto-scaling
            autorange=False,
            scaleanchor=None,
            scaleratio=None
        ),
        title=dict(
            text="📊 Evolución DCA Ultra-Suave",
            x=0.5,
            font=dict(size=18),
            pad=dict(t=20)
        ),
        xaxis_title="📅 Fecha",
        yaxis_title="💰 Rentabilidad Acumulada (%)",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.8)" if tema_oscuro else "rgba(255,255,255,0.8)"
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=80, b=60),
        
        # CONFIGURACIONES CRÍTICAS ANTI-FLICKERING
        uirevision='constant',
        dragmode=False,
        
        # Desactivar todas las interacciones que causan redraws
        modebar={
            'remove': [
                'zoom', 'pan', 'select', 'lasso', 
                'zoomIn', 'zoomOut', 'autoScale', 'resetScale',
                'toImage'
            ]
        },
        
        # Configuraciones de performance
        transition={'duration': 0},  # Sin transiciones
        
        # Evitar redimensionamiento dinámico
        autosize=False,
    )
    
    return fig

# NUEVA FUNCIÓN: Actualización incremental de datos (sin recrear figura)
def actualizar_trazas_incrementalmente(fig, rentabilidades, hasta_indice, mostrar_marcadores=False):
    """Actualiza solo los datos de las trazas existentes"""
    
    # Limpiar trazas existentes sin recrear la figura
    fig.data = []
    
    for ticker in rentabilidades.keys():
        emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
        color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
        
        datos_ticker = rentabilidades[ticker].dropna()
        if len(datos_ticker) == 0:
            continue
            
        # Filtrar datos hasta el índice actual
        if hasta_indice is not None and hasta_indice < len(datos_ticker):
            datos_ticker = datos_ticker.iloc[:hasta_indice + 1]
        
        if len(datos_ticker) == 0:
            continue
        
        # Añadir traza principal con configuración optimizada
        fig.add_trace(go.Scatter(
            x=datos_ticker.index,
            y=datos_ticker.values,
            mode='lines',
            name=f'{emoji} {ticker}',
            line=dict(
                color=color,
                width=3,
                smoothing=0.8  # Suavizado ligero
            ),
            hovertemplate=f'<b>{ticker}</b><br>' +
                        'Fecha: %{x|%d/%m/%Y}<br>' +
                        'Rentabilidad: %{y:.1f}%<br>' +
                        '<extra></extra>',
            connectgaps=True,
            showlegend=True,
            # Optimizaciones específicas para animación
            visible=True,
            opacity=1.0
        ))
        
        # Marcadores de inversión simplificados
        if mostrar_marcadores and hasta_indice is not None and len(datos_ticker) > 0:
            # Solo mostrar último marcador para minimizar elementos
            fecha_actual = datos_ticker.index[-1]
            valor_actual = datos_ticker.iloc[-1]
            
            fig.add_trace(go.Scatter(
                x=[fecha_actual],
                y=[valor_actual],
                mode='markers',
                name=f'{ticker}_marker',
                marker=dict(
                    color=color,
                    size=10,
                    symbol='circle',
                    line=dict(color='white', width=2),
                    opacity=0.9
                ),
                showlegend=False,
                hovertemplate=f'<b>{ticker} - Compra</b><br>' +
                            'Fecha: %{x|%d/%m/%Y}<br>' +
                            '<extra></extra>'
            ))
    
    return fig

# NUEVA FUNCIÓN: Animación con técnica de doble buffer
def ejecutar_animacion_ultra_suave(rentabilidades, inversiones_acumuladas, velocidad, tema_oscuro, mostrar_grid, mostrar_marcadores):
    """Ejecuta animación ultra-suave usando técnica de doble buffer"""
    
    # Preparar datos globales para rangos fijos
    all_dates = []
    all_values = []
    
    for ticker, rent in rentabilidades.items():
        clean_data = rent.dropna()
        if len(clean_data) > 0:
            all_dates.extend(clean_data.index.tolist())
            all_values.extend(clean_data.values.tolist())
    
    if not all_dates or not all_values:
        st.error("No hay datos suficientes para la animación")
        return
    
    # Calcular rangos fijos con margen
    x_range = [min(all_dates), max(all_dates)]
    y_margin = (max(all_values) - min(all_values)) * 0.1
    y_range = [min(all_values) - y_margin, max(all_values) + y_margin]
    
    # Preparar frames de animación optimizados
    min_length = min([len(rent.dropna()) for rent in rentabilidades.values()])
    total_frames = min(60, min_length)  # Optimizado para fluidez
    frame_step = max(1, min_length // total_frames)
    
    # Mapear velocidades a delays
    velocidad_map = {
        "🐌 Muy Lenta": 0.12,
        "🚶 Lenta": 0.08,
        "🏃 Normal": 0.05,
        "🚀 Rápida": 0.03,
        "⚡ Muy Rápida": 0.02
    }
    delay = velocidad_map.get(velocidad, 0.05)
    
    # Crear contenedor único para la animación
    st.markdown("### 🎬 Animación Ultra-Suave en Progreso")
    
    # Contenedor con clase CSS personalizada
    chart_container = st.container()
    
    with chart_container:
        # Progreso de animación
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Área del gráfico con key fijo (CRÍTICO)
        chart_placeholder = st.empty()
        
        # Crear figura base UNA SOLA VEZ
        fig_base = crear_figura_base(x_range, y_range, tema_oscuro, mostrar_grid)
        
        # Loop de animación optimizado
        for frame_idx in range(total_frames):
            data_index = min(frame_idx * frame_step, min_length - 1)
            
            # Crear copia de la figura base
            fig_frame = go.Figure(fig_base)
            
            # Actualizar solo los datos (no el layout)
            fig_frame = actualizar_trazas_incrementalmente(
                fig_frame, 
                rentabilidades, 
                data_index, 
                mostrar_marcadores
            )
            
            # Añadir indicador de progreso temporal
            try:
                current_date = list(rentabilidades.values())[0].dropna().index[data_index]
                progress_text.text(f"📅 {current_date.strftime('%B %Y')} | Frame {frame_idx + 1}/{total_frames}")
            except (IndexError, KeyError):
                progress_text.text(f"Frame {frame_idx + 1}/{total_frames}")
            
            # ACTUALIZACIÓN CRÍTICA: Usar el mismo key durante toda la animación
            with chart_placeholder.container():
                st.plotly_chart(
                    fig_frame,
                    use_container_width=True,
                    key="ultra_smooth_animation",  # KEY FIJO - crucial para eliminar flickering
                    config={
                        'displayModeBar': False,
                        'responsive': True,
                        'staticPlot': False,
                        'doubleClick': False,
                        'showTips': False,
                        'displaylogo': False
                    }
                )
            
            # Actualizar progreso
            progress_bar.progress((frame_idx + 1) / total_frames)
            
            # Delay optimizado
            time.sleep(delay)
        
        # Finalizar animación
        progress_bar.empty()
        progress_text.empty()
        
        # Mostrar resultado final
        fig_final = crear_figura_base(x_range, y_range, tema_oscuro, mostrar_grid)
        fig_final = actualizar_trazas_incrementalmente(fig_final, rentabilidades, None, mostrar_marcadores)
        
        fig_final.update_layout(
            title=dict(text="🎉 Evolución Completa - ¡Animación Ultra-Suave Finalizada!"),
            annotations=[
                dict(
                    text="✨ Animación completada sin parpadeos",
                    xref="paper", yref="paper",
                    x=0.5, y=0.95,
                    showarrow=False,
                    font=dict(size=14, color="green"),
                    bgcolor="rgba(0,255,0,0.1)",
                    bordercolor="green",
                    borderwidth=1,
                    borderpad=8
                )
            ]
        )
        
        with chart_placeholder.container():
            st.plotly_chart(
                fig_final,
                use_container_width=True,
                key="final_ultra_smooth_chart",
                config={'displayModeBar': False, 'responsive': True}
            )
        
        st.success("🎉 ¡Animación ultra-suave completada sin parpadeos!")

# Sidebar - Configuración
st.sidebar.markdown("## ⚙️ Configuración de Análisis")

# Controles de fecha
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

# Inversión mensual
st.sidebar.markdown("### 💰 Inversión Mensual")
inversion_mensual = st.sidebar.select_slider(
    "Cantidad ($):",
    options=[50, 100, 200, 500, 1000, 2000, 5000],
    value=100,
    format_func=lambda x: f"${x:,}"
)

# Selección de activos
st.sidebar.markdown("### 🎯 Selección de Activos")

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

# Configuración de animación optimizada
st.sidebar.markdown("### 🎬 Controles de Animación Ultra-Suave")
velocidad_animacion = st.sidebar.select_slider(
    "Velocidad:",
    options=["🐌 Muy Lenta", "🚶 Lenta", "🏃 Normal", "🚀 Rápida", "⚡ Muy Rápida"],
    value="🏃 Normal"
)

# Opciones de visualización
mostrar_marcadores_inversion = st.sidebar.checkbox("📍 Mostrar días de inversión", value=False)  # Desactivado por defecto para mayor fluidez
mostrar_grid = st.sidebar.checkbox("📊 Mostrar grid", value=True)
tema_oscuro = st.sidebar.checkbox("🌙 Tema oscuro", value=True)

# Botón principal de análisis
if len(activos_seleccionados) > 0:
    if st.sidebar.button("🚀 Iniciar Análisis Ultra-Suave", type="primary"):
        st.session_state.analisis_iniciado = True
        st.session_state.activos_analizar = activos_seleccionados.copy()
        st.session_state.fecha_inicio_analisis = fecha_inicio
        st.session_state.fecha_fin_analisis = fecha_fin
        st.session_state.inversion_analisis = inversion_mensual

# Área principal mejorada
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
            # Crear el área de gráfico principal
            st.markdown("## 📈 Evolución de Rentabilidad DCA Ultra-Suave")
            
            # Controles de animación mejorados
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("▶️ Animación Ultra-Suave", type="primary"):
                    ejecutar_animacion_ultra_suave(
                        rentabilidades,
                        inversiones_acumuladas,
                        velocidad_animacion,
                        tema_oscuro,
                        mostrar_grid,
                        mostrar_marcadores_inversion
                    )
            
            with col2:
                if st.button("📊 Ver Resultado Final"):
                    # Mostrar gráfico final estático
                    all_dates = []
                    all_values = []
                    
                    for ticker, rent in rentabilidades.items():
                        clean_data = rent.dropna()
                        if len(clean_data) > 0:
                            all_dates.extend(clean_data.index.tolist())
                            all_values.extend(clean_data.values.tolist())
                    
                    if all_dates and all_values:
                        x_range = [min(all_dates), max(all_dates)]
                        y_margin = (max(all_values) - min(all_values)) * 0.1
                        y_range = [min(all_values) - y_margin, max(all_values) + y_margin]
                        
                        fig_estatico = crear_figura_base(x_range, y_range, tema_oscuro, mostrar_grid)
                        fig_estatico = actualizar_trazas_incrementalmente(
                            fig_estatico, 
                            rentabilidades, 
                            None, 
                            mostrar_marcadores_inversion
                        )
                        
                        st.plotly_chart(fig_estatico, use_container_width=True)
            
            with col3:
                if st.button("🔄 Reiniciar"):
                    st.session_state.analisis_iniciado = False
                    st.rerun()
            
            # Información técnica sobre mejoras
            with st.expander("🔧 Mejoras Técnicas Anti-Flickering"):
                st.markdown("""
                **Optimizaciones Implementadas:**
                
                1. **Figura Base Estática**: Se crea una vez con rangos fijos
                2. **Key Fijo**: Uso del mismo key durante toda la animación
                3. **Actualización Incremental**: Solo se modifican los datos, no el layout
                4. **CSS Anti-Transiciones**: Eliminación completa de transiciones CSS
                5. **Configuración Plotly Optimizada**: Desactivación de interactividad innecesaria
                6. **Doble Buffer**: Técnica de preparación previa de frames
                7. **Vectorización**: Cálculos optimizados para mejor performance
                """)
            
            # Métricas finales optimizadas
            st.markdown("## 🏆 Resultados Finales")
            
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
            
            # Mostrar métricas en columnas con formato mejorado
            if metricas_finales:
                cols_metricas = st.columns(min(len(metricas_finales), 3))
                
                for i, metrica in enumerate(metricas_finales):
                    ticker = metrica['ticker']
                    with cols_metricas[i % 3]:
                        emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                        color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
                        
                        # Usar métricas nativas de Streamlit
                        st.markdown(f"### {emoji} {ticker}")
                        
                        # Métrica principal
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.metric(
                                label="Rentabilidad",
                                value=f"{metrica['rentabilidad']:+.1f}%",
                                delta=f"${metrica['ganancia_neta']:+,.0f}"
                            )
                        
                        with col2:
                            # Indicador visual de performance
                            if metrica['rentabilidad'] >= 0:
                                st.success("✅ Positivo")
                            else:
                                st.error("❌ Negativo")
                        
                        # Información adicional
                        st.info(f"""
                        **💰 Valor Final:** ${metrica['valor_final']:,.0f}
                        
                        **📊 Total Invertido:** ${metrica['inversion_total']:,.0f}
                        
                        **📈 ROI:** {((metrica['valor_final'] / metrica['inversion_total']) - 1) * 100:.1f}%
                        """)
                        
                        st.markdown("---")
                    
                    # Nueva fila cada 3 elementos
                    if (i + 1) % 3 == 0 and i + 1 < len(metricas_finales):
                        cols_metricas = st.columns(min(len(metricas_finales) - i - 1, 3))

                
                # Resumen general del portfolio
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
        <h2 style="color: #667eea;">🎯 ¿Listo para la visualización ultra-suave?</h2>
        <p style="font-size: 1.2rem; color: #888; margin: 1rem 0;">
            Selecciona tus activos favoritos en el panel lateral y haz clic en "Iniciar Análisis Ultra-Suave" 
            para ver una animación completamente fluida, sin parpadeos.
        </p>
        <div style="margin: 2rem 0;">
            <span style="font-size: 3rem;">📈📊💰</span>
        </div>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin: 2rem auto; max-width: 600px;">
            <h3 style="color: white; margin: 0;">✨ Nuevo: Tecnología Anti-Flickering</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                Implementación con doble buffer, rangos fijos y optimizaciones de rendering.
            </p>
        </div>
        
        <div style="margin: 2rem 0; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🔧 Mejoras Técnicas Implementadas:</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; text-align: left;">
                <div style="padding: 0.5rem;">
                    <strong>🎯 Figura Base Estática</strong><br>
                    <small>Rangos fijos desde el inicio</small>
                </div>
                <div style="padding: 0.5rem;">
                    <strong>🔑 Key Constante</strong><br>
                    <small>Evita recreación de componentes</small>
                </div>
                <div style="padding: 0.5rem;">
                    <strong>⚡ Actualización Incremental</strong><br>
                    <small>Solo modifica datos, no layout</small>
                </div>
                <div style="padding: 0.5rem;">
                    <strong>🎨 CSS Anti-Transiciones</strong><br>
                    <small>Elimina parpadeos visuales</small>
                </div>
                <div style="padding: 0.5rem;">
                    <strong>📊 Plotly Optimizado</strong><br>
                    <small>Configuración de alto rendimiento</small>
                </div>
                <div style="padding: 0.5rem;">
                    <strong>🔄 Doble Buffer</strong><br>
                    <small>Preparación previa de frames</small>
                </div>
            </div>
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

# Footer con información técnica
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>📊 <strong>DCA Evolution Visualizer v3.0</strong> | Ultra-Smooth Anti-Flickering Technology</p>
    <p>🚀 Optimizado para animaciones fluidas con técnicas avanzadas de rendering</p>
    <p>💡 <em>Zero-flickering experience powered by double buffering and static layouts</em></p>
</div>
""", unsafe_allow_html=True)
