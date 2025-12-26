import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime

# --- Configuração da Página ---
st.set_page_config(
    page_title="Painel Quantitativo de Ações BR", 
    layout="wide",
    page_icon="📈"
)

# --- CSS Customizado para visual ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Definição do Universo e Inputs ---
DEFAULT_TICKERS = [
    'ITUB3.SA', 'TOTS3.SA', 'MDIA3.SA', 'TAEE3.SA', 'BBSE3.SA', 'WEGE3.SA', 
    'PSSA3.SA', 'EGIE3.SA', 'B3SA3.SA', 'VIVT3.SA', 'AGRO3.SA', 'PRIO3.SA', 
    'BBAS3.SA', 'BPAC11.SA', 'SBSP3.SA', 'SAPR4.SA', 'CMIG3.SA', 'UNIP6.SA', 'FRAS3.SA'
]
BENCHMARK = "^BVSP" # Ibovespa

# --- 2. Funções de Coleta e Cálculo ---

@st.cache_data(ttl=3600) # Cache de 1 hora
def get_stock_data(tickers, benchmark, period="5y"):
    """Coleta dados de preço ajustado."""
    with st.spinner('Baixando histórico de preços...'):
        tickers_list = tickers + [benchmark]
        # Download otimizado
        data = yf.download(tickers_list, period=period, group_by='ticker', auto_adjust=True, threads=True)
    
    close_prices = pd.DataFrame()
    for t in tickers:
        try:
            # Verifica se o ticker existe no dataframe baixado
            if t in data.columns.levels[0]:
                close_prices[t] = data[t]['Close']
        except KeyError:
            continue
    
    # Tratamento para o Benchmark
    if benchmark in data.columns.levels[0]:
         close_prices['BENCHMARK'] = data[benchmark]['Close']
            
    return close_prices.dropna(how='all')

@st.cache_data(ttl=3600)
def get_fundamentals(tickers):
    """Coleta indicadores fundamentalistas (Snapshot Atual)."""
    metrics = []
    
    # Barra de progresso visual
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers)
    for i, t in enumerate(tickers):
        status_text.text(f"Baixando fundamentos: {t} ({i+1}/{total})")
        try:
            stock = yf.Ticker(t)
            info = stock.info
            
            def get_val(key):
                val = info.get(key, np.nan)
                return val if val is not None else np.nan

            metrics.append({
                'Ticker': t,
                'Nome': info.get('shortName', t),
                'Setor': info.get('sector', 'Outros'),
                'ROE': get_val('returnOnEquity'),
                'Margem Líquida': get_val('profitMargins'),
                'Dívida/EBITDA': get_val('debtToEquity'), # Proxy
                'Crescimento Lucro': get_val('earningsGrowth'),
                'P/L': get_val('trailingPE'),
                'P/VP': get_val('priceToBook'),
                'EV/EBITDA': get_val('enterpriseToEbitda'),
                'Dividend Yield': get_val('dividendYield')
            })
        except:
            metrics.append({'Ticker': t})
        
        progress_bar.progress((i + 1) / total)
            
    status_text.empty()
    progress_bar.empty()
    
    df = pd.DataFrame(metrics)
    return df.set_index('Ticker')

def calculate_residual_momentum(prices_df, lookback_months=12, exclude_recent_months=1):
    """Calcula o Momentum Residual (Alpha ajustado)."""
    momentum_scores = {}
    
    window = lookback_months * 21
    skip = exclude_recent_months * 21
    
    if len(prices_df) < window + skip:
        return pd.Series()

    recent_prices = prices_df.iloc[:-skip]
    
    for col in prices_df.columns:
        if col == 'BENCHMARK': continue
        
        asset_ret = recent_prices[col].pct_change().tail(window).dropna()
        if 'BENCHMARK' not in recent_prices.columns:
            continue
            
        bench_ret = recent_prices['BENCHMARK'].pct_change().tail(window).dropna()
        
        common_idx = asset_ret.index.intersection(bench_ret.index)
        if len(common_idx) < window * 0.8:
            momentum_scores[col] = np.nan
            continue
            
        y = asset_ret.loc[common_idx]
        x = bench_ret.loc[common_idx]
        
        try:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            expected_ret = intercept + slope * x
            residuals = y - expected_ret
            score = residuals.mean() / residuals.std() if residuals.std() != 0 else 0
            momentum_scores[col] = score
        except:
            momentum_scores[col] = np.nan
        
    return pd.Series(momentum_scores, name='Residual Momentum')

# --- 3. Interface do Streamlit ---

st.sidebar.header("⚙️ Configurações")
selected_tickers = st.sidebar.multiselect("Universo de Ações", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
period_input = st.sidebar.selectbox("Período de Análise", ["1y", "3y", "5y"], index=2)

st.sidebar.markdown("---")
st.sidebar.info("**Nota:** Dados obtidos via Yahoo Finance. Podem haver atrasos ou lacunas em fundamentos específicos de Small Caps.")

if st.sidebar.button("🚀 Gerar Painel de Decisão"):
    if not selected_tickers:
        st.error("Por favor, selecione pelo menos um ativo.")
    else:
        # 1. Coleta de Dados
        prices = get_stock_data(selected_tickers, BENCHMARK, period="5y")
        df_funds = get_fundamentals(selected_tickers)
        
        # 2. Cálculos e Tratamento
        df_funds['ROE'] = df_funds['ROE'].fillna(0) * 100
        df_funds['Margem Líquida'] = df_funds['Margem Líquida'].fillna(0) * 100
        df_funds['Dividend Yield'] = df_funds['Dividend Yield'].fillna(0) * 100
        
        # Inverter métricas onde "menor é melhor" para o ranking
        df_calc = df_funds.copy()
        
        # Quality Score
        q_rank = (df_calc['ROE'].rank(pct=True) + 
                  df_calc['Margem Líquida'].rank(pct=True) + 
                  df_calc['Crescimento Lucro'].rank(pct=True) - 
                  df_calc['Dívida/EBITDA'].fillna(100).rank(pct=True)) 
        
        # Value Score (Invertendo P/L, EV/EBITDA, P/VP)
        v_rank = (df_calc['Dividend Yield'].rank(pct=True) + 
                  (1 - df_calc['P/L'].fillna(100).rank(pct=True)) + 
                  (1 - df_calc['P/VP'].fillna(100).rank(pct=True)) + 
                  (1 - df_calc['EV/EBITDA'].fillna(100).rank(pct=True)))
        
        # Momentum Residual
        mom_resid = calculate_residual_momentum(prices, lookback_months=12, exclude_recent_months=1)
        
        # Consolidar
        master_df = df_funds.copy()
        master_df['Residual Momentum'] = mom_resid
        
        # Normalização Final (0 a 100)
        master_df['Quality Rank'] = (q_rank.rank(pct=True) * 100).fillna(0).round(1)
        master_df['Value Rank'] = (v_rank.rank(pct=True) * 100).fillna(0).round(1)
        master_df['Momentum Rank'] = (master_df['Residual Momentum'].rank(pct=True) * 100).fillna(0).round(1)
        
        # Score Geral (Média Ponderada Igualitária)
        master_df['Score Geral'] = ((master_df['Quality Rank'] + 
                                     master_df['Value Rank'] + 
                                     master_df['Momentum Rank']) / 3).round(1)

        # --- Dashboard ---
        st.title(f"📊 Painel de Decisão Quantitativa")
        st.markdown(f"**Universo:** {len(selected_tickers)} Ativos | **Benchmark:** Ibovespa")
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["🏆 Rankings e Sugestão de Aporte", "📈 Backtest de Estratégias", "🧠 Correlações e Insights"])
        
        # === TAB 1: RANKINGS E SUGESTÃO ===
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Picks (Score Global)")
                st.caption("Média de Qualidade, Valor e Momentum")
                top_picks = master_df.sort_values(by='Score Geral', ascending=False).head(10)
                st.dataframe(
                    top_picks[['Score Geral', 'Quality Rank', 'Value Rank', 'Momentum Rank']]
                    .style.background_gradient(cmap='Blues'), 
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Indicadores Fundamentais")
                st.caption("Visão bruta dos dados coletados")
                st.dataframe(master_df[['P/L', 'ROE', 'Dividend Yield', 'Residual Momentum']].style.format("{:.2f}"), use_container_width=True)
            
            # KPI Cards
            st.markdown("##### Líderes por Fator")
            c1, c2, c3 = st.columns(3)
            try:
                top_q = master_df.sort_values(by='Quality Rank', ascending=False).index[0]
                top_v = master_df.sort_values(by='Value Rank', ascending=False).index[0]
                top_m = master_df.sort_values(by='Momentum Rank', ascending=False).index[0]
                c1.info(f"💎 **Top Quality:** {top_q}")
                c2.success(f"💰 **Top Value:** {top_v}")
                c3.warning(f"🚀 **Top Momentum:** {top_m}")
            except:
                pass

            # --- SEÇÃO DE SUGESTÃO DE APORTE ---
            st.markdown("---")
            st.header("💰 Sugestão de Aporte Inteligente (Mês Atual)")
            st.markdown("""
            Abaixo apresentamos uma carteira sugerida para novos aportes, baseada no **Smart Beta**. 
            O peso é calculado proporcionalmente ao *Score Geral* dos Top 5 ativos.
            """)

            # Lógica
            top_n = 5
            suggestion_df = master_df.sort_values(by='Score Geral', ascending=False).head(top_n).copy()
            
            if not suggestion_df.empty:
                total_score = suggestion_df['Score Geral'].sum()
                suggestion_df['Peso Ideal (%)'] = (suggestion_df['Score Geral'] / total_score * 100).round(1)
                
                # Identificar Driver
                def identify_driver(row):
                    factors = {
                        '💎 Quality': row['Quality Rank'], 
                        '💰 Value': row['Value Rank'], 
                        '🚀 Momentum': row['Momentum Rank']
                    }
                    return max(factors, key=factors.get)

                suggestion_df['Fator Dominante'] = suggestion_df.apply(identify_driver, axis=1)
                
                col_sug_1, col_sug_2 = st.columns([1, 2])
                
                with col_sug_1:
                    best_asset = suggestion_df.iloc[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⭐ Destaque do Mês: {best_asset.name}</h3>
                        <h1>{best_asset['Peso Ideal (%)']}% <span style='font-size:16px'>de alocação</span></h1>
                        <p><b>Score Geral:</b> {best_asset['Score Geral']}</p>
                        <p><b>Fator Dominante:</b> {best_asset['Fator Dominante']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sug_2:
                    st.subheader("Carteira de Aporte Sugerida")
                    st.dataframe(
                        suggestion_df[['Peso Ideal (%)', 'Fator Dominante', 'Score Geral', 'P/L', 'Dividend Yield']]
                        .style.background_gradient(subset=['Peso Ideal (%)'], cmap='Greens')
                        .format("{:.1f}", subset=['Score Geral', 'Peso Ideal (%)', 'Dividend Yield'])
                        .format("{:.1f}x", subset=['P/L']),
                        use_container_width=True
                    )
            else:
                st.warning("Dados insuficientes para gerar sugestão.")

        # === TAB 2: BACKTEST ===
        with tab2:
            st.markdown(f"#### Performance Histórica Simulada ({period_input})")
            
            days_map = {"1y": 252, "3y": 252*3, "5y": 252*5}
            lookback = days_map.get(period_input, 252)
            
            if len(prices) > 20:
                bt_prices = prices.iloc[-lookback:].copy()
                bt_ret = bt_prices.pct_change().dropna()
                
                # Definição das Carteiras Teóricas (Viés de Look-ahead para análise de fator atual)
                top_q_idx = master_df.sort_values('Quality Rank', ascending=False).head(int(len(master_df)*0.3)).index
                top_v_idx = master_df.sort_values('Value Rank', ascending=False).head(int(len(master_df)*0.3)).index
                top_m_idx = master_df.sort_values('Momentum Rank', ascending=False).head(int(len(master_df)*0.4)).index
                top_combo_idx = master_df.sort_values('Score Geral', ascending=False).head(5).index
                
                # Filtrar apenas colunas que existem no bt_ret
                def get_valid_cols(indices, df_ret):
                    return [c for c in indices if c in df_ret.columns]

                strategies = {}
                if 'BENCHMARK' in bt_ret.columns:
                    strategies['Ibovespa'] = (1 + bt_ret['BENCHMARK']).cumprod()
                
                cols_q = get_valid_cols(top_q_idx, bt_ret)
                cols_v = get_valid_cols(top_v_idx, bt_ret)
                cols_m = get_valid_cols(top_m_idx, bt_ret)
                cols_combo = get_valid_cols(top_combo_idx, bt_ret)
                
                if cols_q: strategies['Top Quality (30%)'] = (1 + bt_ret[cols_q].mean(axis=1)).cumprod()
                if cols_v: strategies['Top Value (30%)'] = (1 + bt_ret[cols_v].mean(axis=1)).cumprod()
                if cols_m: strategies['Top Momentum (40%)'] = (1 + bt_ret[cols_m].mean(axis=1)).cumprod()
                if cols_combo: strategies['⭐ Top 5 (Sugestão)'] = (1 + bt_ret[cols_combo].mean(axis=1)).cumprod()
                
                df_equity = pd.DataFrame(strategies)
                df_equity = df_equity / df_equity.iloc[0] * 100 # Base 100
                
                fig_bt = px.line(df_equity, title="Curva de Retorno Acumulado (Base 100)")
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Métricas de Risco/Retorno
                metrics_bt = []
                for name, series in strategies.items():
                    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
                    vol = series.pct_change().std() * (252**0.5)
                    sharpe = (total_ret) / vol if vol > 0 else 0
                    
                    roll_max = series.cummax()
                    drawdown = (series - roll_max) / roll_max
                    max_dd = drawdown.min()
                    
                    metrics_bt.append({
                        'Estratégia': name,
                        'Retorno Total': f"{total_ret*100:.1f}%",
                        'Volatilidade (aa)': f"{vol*100:.1f}%",
                        'Sharpe': f"{sharpe:.2f}",
                        'Max Drawdown': f"{max_dd*100:.1f}%"
                    })
                
                st.table(pd.DataFrame(metrics_bt).set_index('Estratégia'))
            else:
                st.error("Dados históricos insuficientes para backtest.")

        # === TAB 3: INSIGHTS ===
        with tab3:
            col_hm1, col_hm2 = st.columns([2, 1])
            
            with col_hm1:
                st.subheader("Mapa de Correlação (Últimos 12m)")
                if len(prices) > 252:
                    recent_ret = prices.iloc[-252:].pct_change().dropna()
                    corr = recent_ret.drop(columns=['BENCHMARK'], errors='ignore').corr()
                    fig_corr = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            with col_hm2:
                st.subheader("Matriz: Valor vs Qualidade")
                st.caption("Eixo X: Value Rank | Eixo Y: Quality Rank | Tamanho: Momentum")
                fig_scat = px.scatter(
                    master_df, 
                    x='Value Rank', y='Quality Rank', 
                    hover_name=master_df.index, 
                    color='Score Geral', 
                    size='Momentum Rank', 
                    size_max=40,
                    color_continuous_scale='Viridis'
                )
                fig_scat.add_hline(y=50, line_dash="dash", line_color="gray")
                fig_scat.add_vline(x=50, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_scat, use_container_width=True)

else:
    st.info("👋 Bem-vindo! Clique no botão na barra lateral para iniciar a análise.")
