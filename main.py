import streamlit as st
import pandas as pd
import re
import requests
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Analizador de Opciones", layout="wide")

st.title("📊 Analizador de Estrategias de Opciones")

def parse_option_symbol(symbol):
    match = re.match(r'^([A-Z]+)([CV])(\d+(?:\.\d+)?)[.\-]?([A-Z]{2})?$', symbol)
    
    if match:
        ticker = match.group(1)
        option_type = 'Call' if match.group(2) == 'C' else 'Put'
        strike = float(match.group(3))
        month_code = match.group(4) if match.group(4) else ''
        
        return {
            'underlying_ticker': ticker,
            'option_type': option_type,
            'strike': strike,
            'month_code': month_code,
            'full_symbol': symbol
        }
    return None

month_codes = {
    'EN': 'Enero', 'FB': 'Febrero', 'FE': 'Febrero', 'MR': 'Marzo', 'AB': 'Abril',
    'MY': 'Mayo', 'JN': 'Junio', 'JL': 'Julio', 'AG': 'Agosto',
    'ST': 'Septiembre', 'OC': 'Octubre', 'NV': 'Noviembre', 'DI': 'Diciembre'
}

manual_mapping = {
    'GFG': 'GGAL',
    'YPF': 'YPFD',
}

@st.cache_data(ttl=300)
def load_data():
    url_options = 'https://data912.com/live/arg_options'
    url_stocks = 'https://data912.com/live/arg_stocks'
    
    options_data = requests.get(url_options).json()
    stocks_data = requests.get(url_stocks).json()
    
    df_options = pd.DataFrame(options_data)
    df_stocks = pd.DataFrame(stocks_data)
    
    parsed_options = []
    
    for idx, row in df_options.iterrows():
        parsed = parse_option_symbol(row['symbol'])
        if parsed:
            parsed.update({
                'px_bid': row['px_bid'],
                'px_ask': row['px_ask'],
                'volume': row['v'],
                'last_price': row['c'],
                'pct_change': row['pct_change']
            })
            parsed_options.append(parsed)
    
    df_options_parsed = pd.DataFrame(parsed_options)
    
    ticker_mapping = {}
    
    for option_ticker in df_options_parsed['underlying_ticker'].unique():
        if option_ticker in manual_mapping:
            ticker_mapping[option_ticker] = manual_mapping[option_ticker]
        elif option_ticker in df_stocks['symbol'].values:
            ticker_mapping[option_ticker] = option_ticker
        else:
            matches = df_stocks[df_stocks['symbol'].str.startswith(option_ticker[:3])]
            if not matches.empty:
                ticker_mapping[option_ticker] = matches.iloc[0]['symbol']
    
    df_merged = df_options_parsed.merge(
        df_stocks[['symbol', 'c', 'px_bid', 'px_ask']].rename(columns={
            'symbol': 'stock_symbol',
            'c': 'stock_price',
            'px_bid': 'stock_bid',
            'px_ask': 'stock_ask'
        }),
        left_on=df_options_parsed['underlying_ticker'].map(ticker_mapping),
        right_on='stock_symbol',
        how='left'
    )
    
    df_merged['month'] = df_merged['month_code'].map(month_codes)
    
    df_merged['moneyness'] = df_merged.apply(
        lambda x: 'ITM' if (x['option_type'] == 'Call' and x['stock_price'] > x['strike']) or 
                           (x['option_type'] == 'Put' and x['stock_price'] < x['strike'])
                  else 'ATM' if abs(x['stock_price'] - x['strike']) / x['stock_price'] < 0.05
                  else 'OTM',
        axis=1
    )
    
    df_merged['intrinsic_value'] = df_merged.apply(
        lambda x: max(0, x['stock_price'] - x['strike']) if x['option_type'] == 'Call'
                  else max(0, x['strike'] - x['stock_price']),
        axis=1
    )
    
    return df_merged, ticker_mapping

def find_strategies(df_merged):
    strategy_opportunities = []
    
    for stock_symbol in df_merged['stock_symbol'].unique():
        if pd.isna(stock_symbol):
            continue
        
        for month in df_merged['month_code'].unique():
            if pd.isna(month):
                continue
            
            subset = df_merged[
                (df_merged['stock_symbol'] == stock_symbol) & 
                (df_merged['month_code'] == month)
            ]
            
            if len(subset) < 2:
                continue
            
            stock_price = subset['stock_price'].iloc[0]
            calls = subset[subset['option_type'] == 'Call'].copy()
            puts = subset[subset['option_type'] == 'Put'].copy()
            
            if calls.empty or puts.empty:
                continue
            
            common_strikes = set(calls['strike']) & set(puts['strike'])
            
            for strike in common_strikes:
                call = calls[calls['strike'] == strike].iloc[0]
                put = puts[puts['strike'] == strike].iloc[0]
                
                if call['last_price'] > 0 and put['last_price'] > 0:
                    total_cost = call['last_price'] + put['last_price']
                    
                    strategy_opportunities.append({
                        'Tipo': 'Straddle',
                        'Stock': stock_symbol,
                        'Precio Actual': stock_price,
                        'Strike Call': strike,
                        'Strike Put': strike,
                        'Vencimiento': subset.iloc[0]['month'],
                        'Precio Call': call['last_price'],
                        'Precio Put': put['last_price'],
                        'Costo Total': total_cost,
                        'BE Superior': strike + total_cost,
                        'BE Inferior': strike - total_cost,
                        'Call Symbol': call['full_symbol'],
                        'Put Symbol': put['full_symbol'],
                        'month_code': month
                    })
            
            for _, call in calls.iterrows():
                for _, put in puts.iterrows():
                    if call['strike'] >= put['strike'] and call['last_price'] > 0 and put['last_price'] > 0:
                        if call['strike'] == put['strike']:
                            continue
                        
                        total_cost = call['last_price'] + put['last_price']
                        breakeven_up = call['strike'] + total_cost
                        breakeven_down = put['strike'] - total_cost
                        
                        strike_diff = (call['strike'] - put['strike']) / stock_price * 100
                        if strike_diff <= 20:
                            strategy_opportunities.append({
                                'Tipo': 'Strangle',
                                'Stock': stock_symbol,
                                'Precio Actual': stock_price,
                                'Strike Call': call['strike'],
                                'Strike Put': put['strike'],
                                'Vencimiento': subset.iloc[0]['month'],
                                'Precio Call': call['last_price'],
                                'Precio Put': put['last_price'],
                                'Costo Total': total_cost,
                                'BE Superior': breakeven_up,
                                'BE Inferior': breakeven_down,
                                'Call Symbol': call['full_symbol'],
                                'Put Symbol': put['full_symbol'],
                                'month_code': month
                            })
    
    df_strategies = pd.DataFrame(strategy_opportunities)
    
    if not df_strategies.empty:
        df_strategies['% BE Superior'] = ((df_strategies['BE Superior'] / df_strategies['Precio Actual']) - 1) * 100
        df_strategies['% BE Inferior'] = ((df_strategies['BE Inferior'] / df_strategies['Precio Actual']) - 1) * 100
        df_strategies['% Mov. Mínimo'] = df_strategies.apply(
            lambda x: min(abs(x['% BE Superior']), abs(x['% BE Inferior'])), 
            axis=1
        )
    
    return df_strategies

def plot_strategy_with_shading(strategy_data):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    stock_price = strategy_data['Precio Actual']
    call_strike = strategy_data['Strike Call']
    put_strike = strategy_data['Strike Put']
    total_cost = strategy_data['Costo Total']
    
    be_up = strategy_data['BE Superior']
    be_down = strategy_data['BE Inferior']
    
    min_price = min(be_down, stock_price)
    max_price = max(be_up, stock_price)
    price_range = max_price - min_price
    margin = price_range * 0.3
    
    prices = np.linspace(min_price - margin, max_price + margin, 200)
    
    pnl = []
    for price in prices:
        call_value = max(0, price - call_strike)
        put_value = max(0, put_strike - price)
        pnl.append(call_value + put_value - total_cost)
    
    pnl = np.array(pnl)
    
    ax.fill_between(prices, pnl, 0, where=(pnl >= 0), color='lightgreen', alpha=0.6, label='Zona de ganancia')
    ax.fill_between(prices, pnl, 0, where=(pnl < 0), color='lightcoral', alpha=0.6, label='Zona de pérdida')
    
    ax.plot(prices, pnl, linewidth=2.5, color='black', label='P&L combinado')
    
    call_pnl = np.maximum(prices - call_strike, 0) - strategy_data['Precio Call']
    put_pnl = np.maximum(put_strike - prices, 0) - strategy_data['Precio Put']
    
    ax.plot(prices, call_pnl, linewidth=1.5, color='skyblue', alpha=0.7, 
            label=f"Long Call ST: {call_strike}")
    ax.plot(prices, put_pnl, linewidth=1.5, color='orange', alpha=0.7, 
            label=f"Long Put ST: {put_strike}")
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.axvline(x=stock_price, color='red', linestyle='--', linewidth=2, 
               label=f'Precio spot: {stock_price:.2f}')
    
    ax.axvline(x=be_up, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=be_down, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    
    strategy_type = strategy_data['Tipo']
    pct_up = strategy_data['% BE Superior']
    pct_down = strategy_data['% BE Inferior']
    
    ax.set_title(
        f"{strategy_type} - {strategy_data['Stock']}\n"
        f"Call ${call_strike:.0f} + Put ${put_strike:.0f} | Costo: ${total_cost:.2f}\n"
        f"Necesita {pct_up:+.1f}% (↑) o {pct_down:+.1f}% (↓)",
        fontsize=14, fontweight='bold', pad=15
    )
    
    ax.set_xlabel('Precio del Subyacente', fontsize=12)
    ax.set_ylabel('Ganancia/Pérdida', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    y_min = min(pnl) * 1.2 if min(pnl) < 0 else -total_cost * 0.2
    y_max = max(pnl) * 1.2 if max(pnl) > 0 else total_cost * 0.2
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig

st.sidebar.header("⚙️ Configuración")

if st.sidebar.button("🔄 Actualizar Datos"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Cargando datos de opciones..."):
    try:
        df_merged, ticker_mapping = load_data()
        st.sidebar.success(f"✅ Datos cargados: {len(df_merged)} opciones")
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()

tab1, tab2, tab3 = st.tabs(["📈 Estrategias", "📊 Datos Completos", "ℹ️ Info"])

with tab1:
    st.header("Estrategias Long Straddle y Strangle")
    
    with st.spinner("Buscando oportunidades..."):
        df_strategies = find_strategies(df_merged)
    
    if not df_strategies.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy_types = ['Todos'] + sorted(df_strategies['Tipo'].unique().tolist())
            selected_type = st.selectbox("Tipo de Estrategia", strategy_types)
        
        with col2:
            sort_options = {
                'Costo Total': 'Costo Total',
                'Movimiento Mínimo': '% Mov. Mínimo',
                'Stock': 'Stock'
            }
            sort_by = st.selectbox("Ordenar por", list(sort_options.keys()))
        
        with col3:
            stocks = ['Todos'] + sorted(df_strategies['Stock'].unique().tolist())
            selected_stock_filter = st.selectbox("Filtrar por Stock", stocks)
        
        filtered_strategies = df_strategies.copy()
        
        if selected_type != 'Todos':
            filtered_strategies = filtered_strategies[filtered_strategies['Tipo'] == selected_type]
        
        if selected_stock_filter != 'Todos':
            filtered_strategies = filtered_strategies[filtered_strategies['Stock'] == selected_stock_filter]
        
        filtered_strategies = filtered_strategies.sort_values(sort_options[sort_by])
        
        st.subheader(f"🎯 {len(filtered_strategies)} Oportunidades Encontradas")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Estrategias", len(filtered_strategies))
        with metric_cols[1]:
            st.metric("Costo Promedio", f"${filtered_strategies['Costo Total'].mean():.2f}")
        with metric_cols[2]:
            st.metric("Mov. Min. Promedio", f"{filtered_strategies['% Mov. Mínimo'].mean():.2f}%")
        with metric_cols[3]:
            straddles = len(filtered_strategies[filtered_strategies['Tipo']=='Straddle'])
            st.metric("Straddles/Strangles", f"{straddles}/{len(filtered_strategies)-straddles}")
        
        display_cols = ['Tipo', 'Stock', 'Precio Actual', 'Strike Call', 'Strike Put', 
                       'Vencimiento', 'Costo Total', 'BE Superior', 'BE Inferior', 
                       '% BE Superior', '% BE Inferior', '% Mov. Mínimo']
        
        st.dataframe(
            filtered_strategies[display_cols].style.format({
                'Precio Actual': '${:.2f}',
                'Strike Call': '${:.0f}',
                'Strike Put': '${:.0f}',
                'Costo Total': '${:.2f}',
                'BE Superior': '${:.2f}',
                'BE Inferior': '${:.2f}',
                '% BE Superior': '{:.2f}%',
                '% BE Inferior': '{:.2f}%',
                '% Mov. Mínimo': '{:.2f}%'
            }).background_gradient(subset=['Costo Total'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=400
        )
        
        st.subheader("📊 Visualizar Estrategia")
        
        original_indices = filtered_strategies.index.tolist()
        
        strategy_options = [
            f"#{idx} - {row['Tipo']} - {row['Stock']} - Call ${row['Strike Call']:.0f}/Put ${row['Strike Put']:.0f} - "
            f"{row['Vencimiento']} (${row['Costo Total']:.2f})"
            for idx, row in filtered_strategies.iterrows()
        ]
        
        selected_idx = st.selectbox(
            "Selecciona una estrategia para graficar:",
            options=range(len(strategy_options)),
            format_func=lambda x: strategy_options[x]
        )
        
        if st.button("🎨 Graficar Estrategia", type="primary"):
            selected_strategy = filtered_strategies.iloc[selected_idx]
            
            fig = plot_strategy_with_shading(selected_strategy)
            st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### 📋 Detalles de la Estrategia")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Tipo", selected_strategy['Tipo'])
            with col2:
                st.metric("Stock", selected_strategy['Stock'])
            with col3:
                st.metric("Precio Actual", f"${selected_strategy['Precio Actual']:.2f}")
            with col4:
                st.metric("Vencimiento", selected_strategy['Vencimiento'])
            with col5:
                st.metric("Costo Total", f"${selected_strategy['Costo Total']:.2f}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Strike Call", f"${selected_strategy['Strike Call']:.0f}")
            with col2:
                st.metric("Strike Put", f"${selected_strategy['Strike Put']:.0f}")
            with col3:
                st.metric("Precio Call", f"${selected_strategy['Precio Call']:.2f}")
            with col4:
                st.metric("Precio Put", f"${selected_strategy['Precio Put']:.2f}")
            with col5:
                st.metric("Mov. Mínimo", f"{selected_strategy['% Mov. Mínimo']:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BE Superior", f"${selected_strategy['BE Superior']:.2f}")
            with col2:
                st.metric("% BE Superior", f"{selected_strategy['% BE Superior']:+.2f}%")
            with col3:
                st.metric("BE Inferior", f"${selected_strategy['BE Inferior']:.2f}")
            with col4:
                st.metric("% BE Inferior", f"{selected_strategy['% BE Inferior']:+.2f}%")
        
    else:
        st.warning("No se encontraron oportunidades de estrategias en los datos actuales.")

with tab2:
    st.header("Datos Completos de Opciones")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stocks = ['Todos'] + sorted(df_merged['stock_symbol'].dropna().unique().tolist())
        selected_stock = st.selectbox("Stock", stocks, key='data_stock')
    
    with col2:
        option_types = ['Todos', 'Call', 'Put']
        selected_type = st.selectbox("Tipo", option_types, key='data_type')
    
    with col3:
        moneyness = ['Todos', 'ITM', 'ATM', 'OTM']
        selected_moneyness = st.selectbox("Moneyness", moneyness, key='data_moneyness')
    
    filtered_df = df_merged.copy()
    
    if selected_stock != 'Todos':
        filtered_df = filtered_df[filtered_df['stock_symbol'] == selected_stock]
    
    if selected_type != 'Todos':
        filtered_df = filtered_df[filtered_df['option_type'] == selected_type]
    
    if selected_moneyness != 'Todos':
        filtered_df = filtered_df[filtered_df['moneyness'] == selected_moneyness]
    
    st.dataframe(
        filtered_df[['full_symbol', 'stock_symbol', 'stock_price', 'option_type', 
                    'strike', 'last_price', 'moneyness', 'intrinsic_value', 'month']],
        use_container_width=True,
        height=500
    )

with tab3:
    st.header("Información del Sistema")
    
    st.write("""
    ### 📖 Acerca de esta aplicación
    
    Analiza opciones del mercado argentino y encuentra oportunidades de:
    - **Long Straddle**: Compra de Call y Put al mismo strike
    - **Long Strangle**: Compra de Call y Put a strikes diferentes
    
    **Características:**
    - Datos en tiempo real
    - Identificación automática de estrategias
    - Visualización con sombreado de ganancia/pérdida
    - Cálculo de breakevens y análisis de sensibilidad
    """)
    
    st.divider()
    
    st.subheader("🔄 Mapeo de Tickers")
    
    mapping_data = []
    for opt_ticker, stock_ticker in ticker_mapping.items():
        mapping_data.append({'Ticker Opción': opt_ticker, 'Ticker Acción': stock_ticker})
    
    st.dataframe(pd.DataFrame(mapping_data), use_container_width=True)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Opciones", len(df_merged))
    with col2:
        st.metric("Stocks Únicos", df_merged['stock_symbol'].nunique())
    with col3:
        if not df_strategies.empty:
            st.metric("Estrategias Encontradas", len(df_strategies))

st.sidebar.divider()
st.sidebar.caption("💡 Los datos se actualizan cada 5 minutos automáticamente")
st.sidebar.caption("📊 El sombreado verde indica ganancia, rojo indica pérdida")

st.divider()
st.caption("📊 Los datos provienen de https://data912.com/ - No puedo mencionar al dueño, porque 🟦🟨")