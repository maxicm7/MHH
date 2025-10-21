import streamlit as st
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime, timedelta
import traceback 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Optimizador de Portafolios",
    page_icon="⚖️",
    layout="wide"
)

# --- CACHING DE DATOS ---
# Usamos @st.cache_data para que la descarga de datos de Yahoo Finance
# solo se ejecute una vez por cada conjunto de tickers y fechas,
# haciendo la aplicación mucho más rápida.
# --- REEMPLAZA TU FUNCIÓN get_stock_data CON ESTA ---

@st.cache_data
def get_stock_data(tickers_string, start_date, end_date):
    """
    Descarga los precios de cierre ajustados para una lista de tickers.
    Devuelve un DataFrame de precios y una lista de tickers que no se pudieron encontrar.
    (Versión Robusta y a Prueba de Errores)
    """
    tickers = [ticker.strip().upper() for ticker in tickers_string.split(',') if ticker.strip()]
    if not tickers:
        return pd.DataFrame(), []

    try:
        # 1. Intenta descargar todos los tickers a la vez
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # 2. VERIFICACIÓN CRÍTICA: Si el DataFrame está vacío, falló toda la descarga.
        if data.empty:
            st.error(f"Fallo total en la descarga. Yahoo Finance no devolvió datos para ninguno de los tickers en el rango de fechas. (Tickers intentados: {', '.join(tickers)})")
            return pd.DataFrame(), tickers

        # 3. Extraer la columna 'Adj Close' de forma segura.
        # Si pides múltiples tickers, yfinance devuelve un MultiIndex.
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close']
        # Si pides un solo ticker, devuelve un DataFrame simple.
        else:
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']]
                # Renombrar la columna para que tenga el nombre del ticker
                prices.columns = [tickers[0]]
            else:
                # Caso muy raro donde ni siquiera viene 'Adj Close' para un solo ticker
                st.error(f"El DataFrame para el ticker '{tickers[0]}' no contiene la columna 'Adj Close'. Columnas disponibles: {', '.join(data.columns)}")
                return pd.DataFrame(), tickers

        # Si después de extraer 'Adj Close' nos queda una Serie (caso de un solo ticker exitoso
        # entre varios fallidos), la convertimos a DataFrame.
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=prices.name)
            
        # 4. Identificar qué tickers específicos fallaron (columnas llenas de NaN)
        # y cuáles tuvieron éxito.
        failed_tickers = prices.columns[prices.isnull().all()].tolist()
        successful_tickers = prices.columns.drop(failed_tickers).tolist()

        if not successful_tickers:
            # Si todos fallaron, aunque el DataFrame inicial no estuviera vacío
            return pd.DataFrame(), tickers
            
        # 5. Devolver un DataFrame limpio solo con los tickers exitosos
        final_prices = prices[successful_tickers]
        
        # Rellenar huecos ocasionales con el último valor conocido
        if final_prices.isnull().values.any():
            final_prices = final_prices.ffill().bfill()
        
        return final_prices.dropna(), failed_tickers

    except Exception as e:
        # Captura cualquier otro error inesperado durante el proceso
        st.error(f"Ocurrió un error inesperado al descargar o procesar los datos: {e}")
        # Muestra el traceback completo en la consola para depuración
        traceback.print_exc()
        return pd.DataFrame(), tickers
# --- TÍTULO Y DESCRIPCIÓN ---
st.title("⚖️ Optimizador de Portafolios Financieros")
st.markdown("""
Esta herramienta utiliza la Teoría Moderna de Portafolios (Markowitz) para encontrar la asignación óptima de activos
en una cartera. Introduce los tickers de las acciones que te interesan y selecciona tu objetivo de optimización.
""")

# --- INPUTS DEL USUARIO EN LA BARRA LATERAL ---
st.sidebar.header("Configuración del Portafolio")

# Entrada de tickers
tickers_input = st.sidebar.text_area("Ingresa los Tickers de las Acciones (separados por comas)", "AAPL, MSFT, GOOG, AMZN, NVDA, TSLA, JPM, JNJ, V, META")

# Rango de fechas para el análisis histórico
today = datetime.today().date()
three_years_ago = today - timedelta(days=3*365)
start_date = st.sidebar.date_input("Fecha de Inicio", three_years_ago)
end_date = st.sidebar.date_input("Fecha de Fin", today)

# Objetivo de optimización
optimization_objective = st.sidebar.selectbox(
    "Selecciona tu Objetivo de Optimización",
    ("Maximizar Ratio de Sharpe (mejor retorno ajustado al riesgo)", 
     "Minimizar Volatilidad (menor riesgo)", 
     "Lograr un Retorno Específico")
)

# Input condicional para retorno específico
target_return = None
if "Lograr un Retorno Específico" in optimization_objective:
    target_return = st.sidebar.slider(
        "Retorno Anual Deseado (%)", 
        min_value=1.0, 
        max_value=100.0, 
        value=20.0, 
        step=1.0
    ) / 100.0

# Botón para ejecutar el análisis
run_button = st.sidebar.button("🚀 Optimizar Portafolio", type="primary")


# --- LÓGICA PRINCIPAL Y VISUALIZACIÓN DE RESULTADOS ---

if run_button:
    if not tickers_input:
        st.warning("Por favor, ingresa al menos un ticker.")
    elif start_date >= end_date:
        st.error("La Fecha de Inicio debe ser anterior a la Fecha de Fin.")
    else:
        with st.spinner("Descargando datos históricos y optimizando..."):
            
            # 1. Obtener los datos de precios
            prices, failed_tickers = get_stock_data(tickers_input, start_date, end_date)
            
            if failed_tickers:
                st.warning(f"No se pudieron encontrar datos para los siguientes tickers: {', '.join(failed_tickers)}")
            
            if prices.empty or len(prices.columns) < 2:
                st.error("La optimización requiere al menos dos activos con datos históricos válidos. Por favor, revisa los tickers y el rango de fechas.")
            else:
                st.subheader("Activos Incluidos en la Optimización")
                st.write(f"Se utilizarán los siguientes {len(prices.columns)} activos:", ", ".join(prices.columns))

                try:
                    # 2. Calcular retornos esperados y la matriz de covarianza
                    mu = expected_returns.mean_historical_return(prices)
                    S = risk_models.sample_cov(prices)
                    
                    # 3. Crear el objeto de optimización
                    ef = EfficientFrontier(mu, S)
                    
                    # 4. Aplicar el objetivo de optimización
                    if "Maximizar Ratio de Sharpe" in optimization_objective:
                        ef.max_sharpe()
                    elif "Minimizar Volatilidad" in optimization_objective:
                        ef.min_volatility()
                    elif "Lograr un Retorno Específico" in optimization_objective and target_return is not None:
                        ef.efficient_return(target_return)
                        
                    # 5. Obtener los pesos limpios (elimina ponderaciones muy pequeñas)
                    cleaned_weights = ef.clean_weights()
                    
                    # 6. Mostrar los resultados
                    st.subheader("📊 Portafolio Óptimo")
                    
                    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Ponderación'])
                    weights_df = weights_df[weights_df['Ponderación'] > 0] # Mostrar solo los que tienen peso
                    weights_df.index.name = 'Ticker'
                    
                    # Gráfico de Torta
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.pie(weights_df['Ponderación'], labels=weights_df.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal') # Asegura que el gráfico sea un círculo
                    st.pyplot(fig)
                    
                    st.write("Ponderaciones Detalladas:")
                    st.dataframe(weights_df.style.format({'Ponderación': '{:.2%}'}))
                    
                    st.subheader("📈 Rendimiento Esperado del Portafolio Óptimo")
                    
                    expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
                    
                    # El resultado de ef.portfolio_performance() se imprime en consola, así que lo capturamos
                    # para mostrarlo en Streamlit
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retorno Anual Esperado", f"{expected_return:.2%}")
                    col2.metric("Volatilidad Anual", f"{annual_volatility:.2%}")
                    col3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")

                except Exception as e:
                    st.error(f"Ocurrió un error durante la optimización: {e}")
                    st.info("Esto puede suceder si el retorno objetivo es inalcanzable (demasiado alto o bajo) para los activos seleccionados. Intenta con un objetivo de retorno diferente o 'Maximizar Ratio de Sharpe'.")
