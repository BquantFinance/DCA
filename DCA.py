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
            
            precios = datos['Adj Close'].fillna(method='ffill').fillna(method='bfill')
            return precios
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
        return None

# Función para calcular DCA con más detalle
def calcular_dca_detallado(precios, inversion_mensual=100):
    """Calcula rentabilidad DCA día a día con detalles"""
    if precios is None or len(precios) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Crear fechas de inversión (primer día de cada mes)
    fechas_inversion = pd.date_range(
        start=precios.index[0].replace(day=1),
        end=precios.index[-1],
        freq='MS'
    )
    
    # Filtrar fechas que están en nuestros datos
    fechas_inversion = [f for f in fechas_inversion if f in precios.index]
    
    if len(fechas_inversion) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Calcular inversión acumulada y acciones compradas
    acciones_totales = pd.Series(0.0, index=precios.index)
    inversion_acumulada = pd.Series(0.0, index=precios.index)
    
    acciones_compradas_mes = 0
    inversion_total = 0
    
    for fecha in precios.index:
        if fecha in fechas_inversion:
            # Día de inversión
            precio_compra = precios.loc[fecha]
            if not pd.isna(precio_compra) and precio_compra > 0:
                acciones_compradas_mes = inversion_mensual / precio_compra
                inversion_total += inversion_mensual
        
        acciones_totales.loc[fecha] = acciones_totales.loc[:fecha].iloc[:-1].sum() + acciones_compradas_mes
        inversion_acumulada.loc[fecha] = inversion_total
        
        acciones_compradas_mes = 0  # Reset para próxima fecha
    
    # Calcular valor del portfolio día a día
    valor_portfolio = acciones_totales * precios
    
    # Calcular rentabilidad porcentual
    rentabilidad = ((valor_portfolio / inversion_acumulada) - 1) * 100
    rentabilidad = rentabilidad.fillna(0)
    
    return rentabilidad, valor_portfolio, inversion_acumulada

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
                    fechas_mostrar = [f for f in todas_fechas if f <= hasta_fecha]
                else:
                    fechas_mostrar = todas_fechas
                
                # Añadir líneas para cada activo
                for ticker in rentabilidades.keys():
                    datos_ticker = rentabilidades[ticker].loc[fechas_mostrar].dropna()
                    
                    if len(datos_ticker) > 0:
                        # Configurar color y estilo
                        color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
                        emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                        
                        # Datos a mostrar (rentabilidad o valores absolutos)
                        if mostrar_valores_abs and ticker in valores_portfolio:
                            y_data = valores_portfolio[ticker].loc[fechas_mostrar].dropna()
                            y_title = "Valor Portfolio ($)"
                            hover_format = "$%{y:,.0f}"
                        else:
                            y_data = datos_ticker
                            y_title = "Rentabilidad (%)"
                            hover_format = "%{y:.1f}%"
                        
                        # Línea principal
                        fig.add_trace(go.Scatter(
                            x=y_data.index,
                            y=y_data,
                            mode='lines',
                            name=f'{emoji} {ticker}',
                            line=dict(
                                color=color,
                                width=3,
                                shape='spline'  # Líneas más suaves
                            ),
                            hovertemplate=f'<b>{ticker}</b><br>' +
                                        'Fecha: %{x}<br>' +
                                        f'Valor: {hover_format}<br>' +
                                        '<extra></extra>',
                            fill=None
                        ))
                        
                        # Añadir marcadores en días de inversión si está habilitado
                        if mostrar_marcadores_inversion and hasta_fecha:
                            # Encontrar días de inversión (primeros del mes)
                            dias_inversion = [f for f in fechas_mostrar 
                                            if f.day == 1 and f in y_data.index]
                            
                            if dias_inversion:
                                valores_inversion = y_data.loc[dias_inversion]
                                
                                fig.add_trace(go.Scatter(
                                    x=valores_inversion.index,
                                    y=valores_inversion,
                                    mode='markers',
                                    name=f'{ticker} (Inversiones)',
                                    marker=dict(
                                        color=color,
                                        size=8,
                                        symbol='circle',
                                        line=dict(color='white', width=2)
                                    ),
                                    showlegend=False,
                                    hovertemplate=f'<b>{ticker} - Día de Inversión</b><br>' +
                                                'Fecha: %{x}<br>' +
                                                f'Valor: {hover_format}<br>' +
                                                '<extra></extra>'
                                ))
                
                # Configurar layout
                template = "plotly_dark" if tema_oscuro else "plotly_white"
                
                fig.update_layout(
                    title=dict(
                        text=f"📈 Evolución DCA - Inversión Mensual: ${st.session_state.inversion_analisis:,}",
                        x=0.5,
                        font=dict(size=24, color='white' if tema_oscuro else 'black')
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
                    height=650,
                    showlegend=True,
                    xaxis=dict(showgrid=mostrar_grid),
                    yaxis=dict(showgrid=mostrar_grid),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                # Añadir anotación con fecha actual si es animación
                if hasta_fecha:
                    fig.add_annotation(
                        text=f"📅 {hasta_fecha.strftime('%d/%m/%Y')}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=18, color="yellow"),
                        bgcolor="rgba(0,0,0,0.7)",
                        bordercolor="yellow",
                        borderwidth=2,
                        borderpad=10
                    )
                
                return fig
            
            # Animación o gráfico estático
            if 'reproducir_animacion' in st.session_state and st.session_state.reproducir_animacion:
                st.session_state.reproducir_animacion = False  # Reset
                
                # Preparar fechas para animación (saltar fechas para mejor performance)
                fechas_animacion = todas_fechas[::max(1, len(todas_fechas) // 200)]  # Máximo 200 frames
                
                progress_animacion = st.progress(0)
                
                for i, fecha_actual in enumerate(fechas_animacion):
                    fig = crear_grafico_animado(fecha_actual, mostrar_valores)
                    chart_container.plotly_chart(fig, use_container_width=True, key=f"anim_{i}")
                    
                    # Información en tiempo real
                    rentabilidades_actuales = {}
                    for ticker in rentabilidades.keys():
                        if fecha_actual in rentabilidades[ticker].index:
                            rentabilidades_actuales[ticker] = rentabilidades[ticker].loc[fecha_actual]
                    
                    if rentabilidades_actuales:
                        # Mostrar métricas actuales
                        cols_info = st.columns(len(rentabilidades_actuales))
                        info_text = ""
                        
                        for j, (ticker, valor) in enumerate(rentabilidades_actuales.items()):
                            emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                            if mostrar_valores and ticker in valores_portfolio:
                                valor_mostrar = valores_portfolio[ticker].loc[fecha_actual]
                                info_text += f"{emoji} **{ticker}**: ${valor_mostrar:,.0f} | "
                            else:
                                info_text += f"{emoji} **{ticker}**: {valor:.1f}% | "
                        
                        info_container.markdown(f"**Valores actuales:** {info_text[:-3]}")
                    
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
            
            cols_metricas = st.columns(len(rentabilidades))
            
            for i, (ticker, rentabilidad) in enumerate(rentabilidades.items()):
                with cols_metricas[i]:
                    valor_final = rentabilidad.dropna().iloc[-1] if len(rentabilidad.dropna()) > 0 else 0
                    valor_portfolio_final = valores_portfolio[ticker].dropna().iloc[-1] if ticker in valores_portfolio and len(valores_portfolio[ticker].dropna()) > 0 else 0
                    inversion_total_final = inversiones_acumuladas[ticker].dropna().iloc[-1] if ticker in inversiones_acumuladas and len(inversiones_acumuladas[ticker].dropna()) > 0 else 0
                    
                    emoji = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('emoji', '📈')
                    color = ACTIVOS_PREDEFINIDOS.get(ticker, {}).get('color', '#1f77b4')
                    
                    # Card personalizada con métricas
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {color}20, {color}10);
                        border-left: 4px solid {color};
                        padding: 1rem;
                        border-radius: 10px;
                        margin: 0.5rem 0;
                    ">
                        <h3 style="margin: 0; color: {color};">{emoji} {ticker}</h3>
                        <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                            {valor_final:+.1f}%
                        </p>
                        <p style="margin: 0; opacity: 0.8;">
                            💰 ${valor_portfolio_final:,.0f}<br>
                            📊 Invertido: ${inversion_total_final:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
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
