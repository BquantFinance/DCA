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
    page_title="BQuant-DCA Evolution Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejor apariencia y animaciones suaves
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
    /* Critical: Prevent chart container from resizing */
    div[data-testid="stPlotlyChart"] > div {
        transition: none !important;
        animation: none !important;
    }
    /* Smooth fade transitions for info updates */
    .info-container {
        transition: opacity 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con estilo
st.markdown('<h1 class="main-header">📈 BQuant-DCA Evolution Visualizer</h1>', unsafe_allow_html=True)
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
            
            precios = datos['Close'].ffill().bfill()
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
    rentabilidad_completa = rentabilidad_diaria.reindex(precios.index).ffill().fillna(0)
    valor_completo = valor_portfolio_diario.reindex(precios.index).ffill().fillna(0)
    inversion_completa = inversion_acumulada_diaria.reindex(precios.index).ffill().fillna(0)
    
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

# Mapear velocidades a delays - Velocidades más optimizadas
velocidad_map = {
    "🐌 Muy Lenta": 0.08,
    "🚶 Lenta": 0.05,
    "🏃 Normal": 0.03,
    "🚀 Rápida": 0.02,
    "⚡ Muy Rápida": 0.01
}

# Opciones de visualización
mostrar_marcadores_inversion = st.sidebar.checkbox("📍 Mostrar días de inversión", value=True)
mostrar_grid = st.sidebar.checkbox("📊 Mostrar grid", value=True)
tema_oscuro = st.sidebar.checkbox("🌙 Tema oscuro", value=True)

# Nueva opción: Suavizado de curvas
suavizado_curvas = st.sidebar.checkbox("🌊 Suavizado de curvas", value=True)

# Botón principal de análisis
if len(activos_seleccionados) > 0:
    if st.sidebar.button("🚀 Iniciar Análisis", type="primary"):
        st.session_state.analisis_iniciado = True
        st.session_state.activos_analizar = activos_seleccionados.copy()
        st.session_state.fecha_inicio_analisis = fecha_inicio
        st.session_state.fecha_fin_analisis = fecha_fin
        st.session_state.inversion_analisis = inversion_mensual

# Función optimizada para crear gráfico sin flickering
def crear_grafico_ultra_suave(rentabilidades, inversiones_acumuladas, hasta_indice=None, tema_oscuro=True, mostrar_grid=True, suavizado=True):
    """Crea gráfico optimizado para animación sin flickering"""
    
    # Configurar template
    template = "plotly_dark" if tema_oscuro else "plotly_white"
    
    # Crear figura base una sola vez
    fig = go.Figure()
    
    # Configurar layout fijo para evitar redimensionamiento
    fig.update_layout(
        template=template,
        height=500,  # Altura fija
        width=None,  # Ancho responsivo pero sin cambios abruptos
        title=dict(
            text=f"📊 Evolución DCA Ultra-Suave - ${st.session_state.inversion_analisis:,}/mes",
            x=0.5,
            font=dict(size=18, color='white' if tema_oscuro else 'black'),
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
        xaxis=dict(
            showgrid=mostrar_grid,
            gridcolor='rgba(128,128,128,0.1)',
            fixedrange=True,  # Evitar zoom automático
            tickformat='%b %Y'
        ),
        yaxis=dict(
            showgrid=mostrar_grid,
            gridcolor='rgba(128,128,128,0.1)',
            tickformat=".1f",
            ticksuffix="%",
            fixedrange=True,  # Evitar zoom automático
            autorange=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=80, b=60),
        # Configuraciones críticas para evitar flickering
        uirevision='constant',  # Mantener estado UI constante
        dragmode=False,  # Deshabilitar interactividad que cause redraws
        # Optimizaciones de rendimiento
        modebar={'remove': ['zoom', 'pan', 'select', 'lasso', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']},
    )
    
    # Preparar datos según índice actual
    for ticker in rentabilidades.keys():
        emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
        color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
        
        # Obtener datos hasta el punto actual
        datos_ticker = rentabilidades[ticker].dropna()
        if len(datos_ticker) == 0:
            continue
            
        # Filtrar datos según progreso de animación
        if hasta_indice is not None:
            if hasta_indice < len(datos_ticker):
                datos_ticker = datos_ticker.iloc[:hasta_indice + 1]
        
        if len(datos_ticker) == 0:
            continue
        
        # Línea principal optimizada
        line_config = dict(
            color=color,
            width=3
        )
        
        # Aplicar suavizado solo si está habilitado y hay suficientes puntos
        if suavizado and len(datos_ticker) > 5:
            line_config['smoothing'] = 1.0  # Suavizado conservador
        
        fig.add_trace(go.Scatter(
            x=datos_ticker.index,
            y=datos_ticker.values,
            mode='lines',
            name=f'{emoji} {ticker}',
            line=line_config,
            hovertemplate=f'<b>{ticker}</b><br>' +
                        'Fecha: %{x|%d/%m/%Y}<br>' +
                        'Rentabilidad: %{y:.1f}%<br>' +
                        '<extra></extra>',
            connectgaps=True,
            # Optimización: usar less traces
            showlegend=True
        ))
        
        # Marcadores de inversión optimizados (solo si es necesario)
        if mostrar_marcadores_inversion and ticker in inversiones_acumuladas and hasta_indice is not None:
            inversiones_ticker = inversiones_acumuladas[ticker].dropna()
            if len(inversiones_ticker) > 0:
                # Detectar días de inversión de forma más eficiente
                diff_inversiones = inversiones_ticker.diff()
                dias_inversion = diff_inversiones[diff_inversiones > 0]
                
                # Filtrar según progreso
                if hasta_indice < len(datos_ticker):
                    fecha_limite = datos_ticker.index[hasta_indice]
                    dias_inversion = dias_inversion[dias_inversion.index <= fecha_limite]
                
                if len(dias_inversion) > 0:
                    # Solo últimos 3 marcadores para evitar saturación
                    dias_recientes = dias_inversion.tail(3)
                    valores_marcadores = datos_ticker.reindex(dias_recientes.index).dropna()
                    
                    if len(valores_marcadores) > 0:
                        fig.add_trace(go.Scatter(
                            x=valores_marcadores.index,
                            y=valores_marcadores.values,
                            mode='markers',
                            name=f'{ticker}_markers',
                            marker=dict(
                                color=color,
                                size=8,
                                symbol='circle',
                                line=dict(color='white', width=1.5),
                                opacity=0.8
                            ),
                            showlegend=False,
                            hovertemplate=f'<b>{ticker} - Compra</b><br>' +
                                        'Fecha: %{x|%d/%m/%Y}<br>' +
                                        f'💰 ${st.session_state.inversion_analisis:,}<br>' +
                                        '<extra></extra>'
                        ))
    
    return fig

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
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("▶️ Reproducir Ultra-Suave", type="primary"):
                    st.session_state.reproducir_animacion = True
            
            with col2:
                if st.button("⏸️ Pausar"):
                    st.session_state.reproducir_animacion = False
            
            with col3:
                if st.button("📊 Ver Final"):
                    st.session_state.mostrar_final = True
            
            # Crear contenedor fijo para el gráfico
            chart_placeholder = st.empty()
            
            # Animación ultra-suave sin flickering
            if 'reproducir_animacion' in st.session_state and st.session_state.reproducir_animacion:
                st.session_state.reproducir_animacion = False  # Reset
                
                # Encontrar la longitud común mínima
                min_length = min([len(rent.dropna()) for rent in rentabilidades.values()])
                
                if min_length == 0:
                    st.error("No hay datos suficientes para la animación")
                else:
                    st.info("🎬 Iniciando animación ultra-suave sin flickering...")
                    
                    # Preparar frames de animación optimizados
                    total_frames = min(100, min_length)  # Máximo 100 frames para fluidez
                    frame_step = max(1, min_length // total_frames)
                    
                    # Crear una sola vez el contenedor del gráfico
                    with chart_placeholder.container():
                        chart_container = st.empty()
                        progress_container = st.empty()
                        
                        # Barra de progreso
                        progress_bar = progress_container.progress(0)
                        
                        # Pre-configurar el layout para evitar redimensionamiento
                        fig_base = crear_grafico_ultra_suave(
                            rentabilidades, 
                            inversiones_acumuladas, 
                            hasta_indice=min_length-1,  # Gráfico completo para establecer escalas
                            tema_oscuro=tema_oscuro,
                            mostrar_grid=mostrar_grid,
                            suavizado=suavizado_curvas
                        )
                        
                        # Configurar rango fijo para evitar reescalado
                        all_dates = []
                        all_values = []
                        for rent in rentabilidades.values():
                            clean_data = rent.dropna()
                            if len(clean_data) > 0:
                                all_dates.extend(clean_data.index.tolist())
                                all_values.extend(clean_data.values.tolist())
                        
                        if all_dates and all_values:
                            fig_base.update_layout(
                                xaxis=dict(
                                    range=[min(all_dates), max(all_dates)],
                                    fixedrange=True
                                ),
                                yaxis=dict(
                                    range=[min(all_values) * 1.1, max(all_values) * 1.1],
                                    fixedrange=True
                                )
                            )
                        
                        # Animación frame por frame
                        for frame_idx in range(total_frames):
                            # Calcular índice de datos para este frame
                            data_index = min(frame_idx * frame_step, min_length - 1)
                            
                            # Crear figura para este frame
                            fig_frame = crear_grafico_ultra_suave(
                                rentabilidades,
                                inversiones_acumuladas,
                                hasta_indice=data_index,
                                tema_oscuro=tema_oscuro,
                                mostrar_grid=mostrar_grid,
                                suavizado=suavizado_curvas
                            )
                            
                            # Mantener rangos fijos (crítico para evitar flickering)
                            if all_dates and all_values:
                                fig_frame.update_layout(
                                    xaxis=dict(
                                        range=[min(all_dates), max(all_dates)],
                                        fixedrange=True
                                    ),
                                    yaxis=dict(
                                        range=[min(all_values) * 1.1, max(all_values) * 1.1],
                                        fixedrange=True
                                    )
                                )
                            
                            # Indicador de progreso en el gráfico
                            if len(all_dates) > data_index:
                                current_date = list(rentabilidades.values())[0].dropna().index[data_index]
                                fig_frame.add_annotation(
                                    text=f"📅 {current_date.strftime('%b %Y')}",
                                    xref="paper", yref="paper",
                                    x=0.02, y=0.02,
                                    showarrow=False,
                                    font=dict(size=12, color="rgba(255,255,255,0.8)"),
                                    bgcolor="rgba(0,0,0,0.3)",
                                    borderwidth=0,
                                    borderpad=5
                                )
                            
                            # Actualizar gráfico SIN CAMBIAR EL CONTENEDOR
                            # La clave está en usar el MISMO key para cada frame
                            chart_container.plotly_chart(
                                fig_frame,
                                use_container_width=True,
                                config={
                                    'displayModeBar': False,
                                    'staticPlot': False,  # Permite hover pero no interacciones que causen redraws
                                    'responsive': True
                                },
                                key="smooth_animation_chart"  # MISMO KEY SIEMPRE
                            )
                            
                            # Actualizar progreso
                            progress_bar.progress((frame_idx + 1) / total_frames)
                            
                            # Delay optimizado
                            time.sleep(velocidad_map[velocidad_animacion])
                        
                        # Limpiar barra de progreso
                        progress_container.empty()
                        
                        # Mostrar resultado final
                        fig_final = crear_grafico_ultra_suave(
                            rentabilidades,
                            inversiones_acumuladas,
                            hasta_indice=None,  # Gráfico completo
                            tema_oscuro=tema_oscuro,
                            mostrar_grid=mostrar_grid,
                            suavizado=suavizado_curvas
                        )
                        
                        fig_final.update_layout(
                            title=dict(text="🎉 Evolución Completa - Animación Ultra-Suave Finalizada")
                        )
                        
                        chart_container.plotly_chart(
                            fig_final,
                            use_container_width=True,
                            key="final_chart"
                        )
                        
                        st.success("🎉 ¡Animación ultra-suave completada sin flickering!")
            
            elif 'mostrar_final' in st.session_state and st.session_state.mostrar_final:
                st.session_state.mostrar_final = False  # Reset
                
                # Mostrar gráfico final estático
                fig_estatico = crear_grafico_ultra_suave(
                    rentabilidades,
                    inversiones_acumuladas,
                    hasta_indice=None,
                    tema_oscuro=tema_oscuro,
                    mostrar_grid=mostrar_grid,
                    suavizado=suavizado_curvas
                )
                
                chart_placeholder.plotly_chart(fig_estatico, use_container_width=True)
            
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
            
            # Mostrar métricas en columnas con formato mejorado
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
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin: 2rem auto; max-width: 500px;">
            <h3 style="color: white; margin: 0;">✨ Nuevo: Animación Ultra-Suave</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                Sin flickering ni parpadeos. Experiencia de visualización completamente fluida.
            </p>
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
    <p>📊 <strong>DCA Evolution Visualizer v2.0</strong> | Ultra-Smooth Animation</p>
    <p>💡 Visualiza el poder del Dollar Cost Averaging sin flickering</p>
</div>
""", unsafe_allow_html=True)
