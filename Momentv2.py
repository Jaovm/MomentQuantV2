import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --- Configuração da Página ---
st.set_page_config(
    page_title="Painel Quantitativo de Ações BR", 
    layout="wide",
    page_icon="📈"
)

# --- CSS Customizado ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0083B8;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Definição do Universo ---
DEFAULT_TICKERS = [
    'ITUB3.SA', 'TOTS3.SA', 'MDIA3.SA', 'TAEE3.SA', 'BBSE3.SA', 'WEGE3.SA', 
    'PSSA3.SA', 'EGIE3.SA', 'B3SA3.SA', 'VIVT3.SA', 'AGRO3.SA', 'PRIO3.SA', 
    'BBAS3.SA', 'BPAC11.SA', 'SBSP3.SA', 'SAPR4.SA', 'CMIG3.SA', 'UNIP6.SA', 'FRAS3.SA'
]
BENCHMARK = "^BVSP"

# --- 2. Funções de Coleta e Cálculo ---

@st.cache_data(ttl=3600)
def get_stock_data(tickers, benchmark, period="5y"):
    tickers_list = list(set(tickers + [benchmark]))
    data = yf.download(tickers_list, period=period, group_by='ticker', auto_adjust=True, threads=True)
    close_prices = pd.DataFrame()
    for t in tickers:
        if t in data.columns.levels[0]:
            close_prices[t] = data[t]['Close']
    if benchmark in data.columns.levels[0]:
         close_prices['BENCHMARK'] = data[benchmark]['Close']
    return close_prices.dropna(how='all')

@st.cache_data(ttl=3600)
def get_fundamentals(tickers):
    metrics = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, t in enumerate(tickers):
        status_text.text(f"Baixando fundamentos: {t}")
        try:
            stock = yf.Ticker(t)
            info = stock.info
            get_v = lambda k: info.get(k, np.nan) if info.get(k) is not None else np.nan
            metrics.append({
                'Ticker': t,
                'ROE': get_v('returnOnEquity'),
                'Margem': get_v('profitMargins'),
                'Div_Ebitda': get_v('debtToEquity'),
                'Cresc_Lucro': get_v('earningsGrowth'),
                'PL': get_v('trailingPE'),
                'PVP': get_v('priceToBook'),
                'EV_Ebitda': get_v('enterpriseToEbitda'),
                'DY': get_v('dividendYield')
            })
        except: metrics.append({'Ticker': t})
        progress_bar.progress((i + 1) / len(tickers))
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(metrics).set_index('Ticker')

def calculate_residual_momentum(prices_df):
    momentum_scores = {}
    window, skip = 252, 21 # 12 meses, exclui o último
    if len(prices_df) < window + skip or 'BENCHMARK' not in prices_df.columns: return pd.Series()
    recent_prices = prices_df.iloc[-(window+skip):-skip]
    for col in prices_df.columns:
        if col == 'BENCHMARK': continue
        y = recent_prices[col].pct_change().dropna()
        x = recent_prices['BENCHMARK'].pct_change().dropna()
        common = y.index.intersection(x.index)
        if len(common) < 100: continue
        slope, intercept, _, _, _ = stats.linregress(x.loc[common], y.loc[common])
        resid = y.loc[common] - (intercept + slope * x.loc[common])
        momentum_scores[col] = resid.mean() / resid.std() if resid.std() != 0 else 0
    return pd.Series(momentum_scores, name='Resid_Mom')

# --- 3. Sidebar e Controles ---
st.sidebar.header("⚙️ Configurações")
selected_tickers = st.sidebar.multiselect("Universo de Ações", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
period_input = st.sidebar.selectbox("Histórico para Backtest", ["1y", "3y", "5y"], index=1)

st.sidebar.subheader("💡 Sugestão de Aporte")
# --- NOVO CONTROLE DE QUANTIDADE ---
n_aportes = st.sidebar.slider("Quantidade de ações na sugestão", min_value=1, max_value=10, value=5)

if st.sidebar.button("🚀 Gerar Painel de Decisão"):
    prices = get_stock_data(selected_tickers, BENCHMARK)
    df_funds = get_fundamentals(selected_tickers)
    
    # Cálculos de Rankings
    df_funds = df_funds.fillna(0)
    q_rank = (df_funds['ROE'].rank(pct=True) + df_funds['Margem'].rank(pct=True) - df_funds['Div_Ebitda'].rank(pct=True))
    v_rank = (df_funds['DY'].rank(pct=True) + (1 - df_funds['PL'].rank(pct=True)) + (1 - df_funds['PVP'].rank(pct=True)))
    mom_resid = calculate_residual_momentum(prices)
    
    master_df = df_funds.copy()
    master_df['Quality_Rank'] = (q_rank.rank(pct=True) * 100).round(1)
    master_df['Value_Rank'] = (v_rank.rank(pct=True) * 100).round(1)
    master_df['Mom_Rank'] = (mom_resid.rank(pct=True) * 100).fillna(0).round(1)
    master_df['Score_Geral'] = ((master_df['Quality_Rank'] + master_df['Value_Rank'] + master_df['Mom_Rank']) / 3).round(1)

    # --- Dashboard Tabs ---
    st.title("📊 Estratégia Quantitativa de Aportes")
    tab1, tab2 = st.tabs(["🏆 Sugestão de Aporte & Rankings", "📈 Performance Histórica"])

    with tab1:
        # SEÇÃO DE SUGESTÃO DINÂMICA
        st.header(f"💰 Sugestão de Aporte para o Mês (Top {n_aportes} Ativos)")
        suggestion_df = master_df.sort_values(by='Score_Geral', ascending=False).head(n_aportes).copy()
        
        # Cálculo de Pesos (Smart Weighting)
        total_score = suggestion_df['Score_Geral'].sum()
        suggestion_df['Peso_Sug_Pct'] = (suggestion_df['Score_Geral'] / total_score * 100).round(1)
        
        col_top, col_table = st.columns([1, 2])
        with col_top:
            top_1 = suggestion_df.iloc[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3>⭐ Principal Ativo: {top_1.name}</h3>
                <h1 style='color:#0083B8'>{top_1['Peso_Sug_Pct']}%</h1>
                <p><b>Alocação sugerida do capital novo.</b></p>
                <hr>
                <p>Qualidade: {top_1['Quality_Rank']}<br>Valor: {top_1['Value_Rank']}<br>Momentum: {top_1['Mom_Rank']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_table:
            st.dataframe(
                suggestion_df[['Peso_Sug_Pct', 'Score_Geral', 'DY', 'PL']]
                .style.background_gradient(subset=['Peso_Sug_Pct'], cmap='Blues')
                .format({'Peso_Sug_Pct': '{:.1f}%', 'DY': '{:.2%}', 'PL': '{:.1f}x'}),
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📋 Ranking Geral do Universo Selecionado")
        st.dataframe(master_df[['Score_Geral', 'Quality_Rank', 'Value_Rank', 'Mom_Rank']].sort_values('Score_Geral', ascending=False), use_container_width=True)

    with tab2:
        # Backtest Simplificado
        lookback_days = {"1y": 252, "3y": 756, "5y": 1260}[period_input]
        bt_prices = prices.iloc[-lookback_days:]
        bt_rets = bt_prices.pct_change().dropna()
        
        # Equity Curve da Sugestão vs Benchmark
        portfolio_rets = bt_rets[suggestion_df.index].mean(axis=1)
        bench_rets = bt_rets['BENCHMARK']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_rets.index, y=(1+portfolio_rets).cumprod(), name=f"Top {n_aportes} Selecionados"))
        fig.add_trace(go.Scatter(x=bench_rets.index, y=(1+bench_rets).cumprod(), name="Ibovespa (Benchmark)", line=dict(dash='dash')))
        fig.update_layout(title="Comparativo de Retorno Acumulado", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Configure os ativos e a quantidade de sugestões na barra lateral e clique em 'Gerar Painel'.")
