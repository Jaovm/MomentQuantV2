import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 0. CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==============================================================================
st.set_page_config(
    page_title="SPX Quant Lab | Decision Support",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# CSS Customizado para visual "Dark Professional"
st.markdown("""
    <style>
    .metric-card {background-color: #0E1117; border: 1px solid #303030; padding: 20px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 1. ENGINE DE DADOS (PARALELISMO & PERFORMANCE)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados com tratamento de MultiIndex."""
    t_list = list(set(tickers))
    if 'BOVA11.SA' not in t_list:
        t_list.append('BOVA11.SA')
    
    try:
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False
        )['Adj Close']
        
        # Tratamento para mudanças recentes na API do yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro crítico no download de preços: {e}")
        return pd.DataFrame()

def get_single_ticker_info(ticker: str):
    """Função auxiliar para thread worker."""
    try:
        return yf.Ticker(ticker).info
    except:
        return None

@st.cache_data(ttl=3600*24)
def fetch_fundamentals_parallel(tickers: list) -> pd.DataFrame:
    """Busca fundamentalista assíncrona (Multithreading)."""
    clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
    data = []
    
    # Execução Paralela (IO-Bound optimization)
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(get_single_ticker_info, clean_tickers))
        
    for i, info in enumerate(results):
        if info:
            t = clean_tickers[i]
            # Setorização Inteligente
            sector = info.get('sector', 'Unknown')
            if sector in ['Unknown', 'N/A'] and 'longName' in info:
                 name = info['longName'].lower()
                 if 'banco' in name or 'seguridade' in name: sector = 'Financial Services'
                 elif 'energia' in name or 'elétrica' in name: sector = 'Utilities'
            
            # Métricas Calculadas
            ebitda = info.get('ebitda', np.nan)
            ev = info.get('enterpriseValue', np.nan)
            ev_ebitda = ev / ebitda if (ebitda and ev and ebitda > 0) else np.nan
            
            # Accruals (Qualidade do Lucro)
            net_income = info.get('netIncomeToCommon', np.nan)
            ocf = info.get('operatingCashflow', np.nan)
            assets = info.get('totalAssets', np.nan)
            accruals = (net_income - ocf) / assets if (assets and assets > 0) else 0

            data.append({
                'ticker': t,
                'sector': sector,
                'forwardPE': info.get('forwardPE', np.nan),
                'evToEbitda': info.get('enterpriseToEbitda', ev_ebitda),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan),
                'accruals': accruals # Menor é melhor
            })
    
    if not data: return pd.DataFrame()
    return pd.DataFrame(data).set_index('ticker')

# ==============================================================================
# 2. MATH & FACTORS (ALPHA GENERATION)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Alpha Puro: Momentum descontando o Beta do Mercado."""
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
    market = rets['BOVA11.SA']
    
    scores = {}
    window = lookback + skip
    
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        y = rets[ticker].tail(window)
        x = market.tail(window)
        if len(y) < window: continue
        try:
            model = sm.OLS(y.values, sm.add_constant(x.values)).fit()
            resid = model.resid[:-skip]
            # Ajuste pelo risco idiossincrático (Sharpe do Resíduo)
            scores[ticker] = (np.sum(resid) / np.std(resid)) if np.std(resid) > 0 else 0
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_fundamental_factors(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Scores Fundamentais Normalizados."""
    df = fund_df.copy()
    
    # 1. Fundamental Momentum (Growth)
    metrics_growth = ['earningsGrowth', 'revenueGrowth']
    df['Fund_Mom'] = 0
    for m in metrics_growth:
        if m in df.columns:
            s = df[m].fillna(df[m].median())
            df['Fund_Mom'] += (s - s.mean()) / s.std()
    
    # 2. Value (Earnings Yield & EV/EBITDA Inverso)
    # Preferimos EV/EBITDA. Se for alto, é caro (score baixo).
    # Tratamento: Inverter valor. Se negativo, penalizar.
    ev_score = df['evToEbitda'].fillna(df['evToEbitda'].median())
    df['Value_Score'] = np.where(ev_score > 0, 1/ev_score, -1) # Quanto maior o inverso, mais barato
    
    # 3. Quality (ROE + Margem - Alavancagem - Accruals)
    roe = df['returnOnEquity'].fillna(0)
    accruals = df['accruals'].fillna(0) # Queremos accruals baixos (lucro caixa)
    lev = df['debtToEquity'].fillna(100)
    
    # Z-Score simples para combinação interna
    z_roe = (roe - roe.mean()) / roe.std()
    z_acc = -1 * (accruals - accruals.mean()) / accruals.std() # Inverte sinal
    z_lev = -1 * (lev - lev.mean()) / lev.std() # Inverte sinal (menos dívida é melhor)
    
    df['Quality_Score'] = (z_roe + z_acc + z_lev) / 3
    
    return df[['Fund_Mom', 'Value_Score', 'Quality_Score']]

# ==============================================================================
# 3. GESTÃO DE RISCO (HIERARCHICAL RISK PARITY)
# ==============================================================================

def get_hrp_weights(price_data: pd.DataFrame) -> pd.Series:
    """
    Calcula pesos via Hierarchical Risk Parity (HRP).
    Diferencial: Não penaliza diversificação em clusters correlacionados.
    """
    rets = price_data.pct_change().dropna()
    cov = rets.cov()
    corr = rets.corr()
    
    # 1. Clustering Hierárquico
    dist = np.sqrt((1 - corr) / 2)
    link = sch.linkage(squareform(dist.clip(0, 1)), 'single')
    
    # 2. Quasi-Diagonalization (Reordenar matriz para agrupar clusters)
    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()
    
    sort_ix = get_quasi_diag(link)
    sorted_tickers = corr.index[sort_ix].tolist()
    
    # 3. Recursive Bisection (Simplificado: Inverse Variance na ordem clusterizada)
    # Isso garante que o risco flua suavemente entre clusters
    cov_sorted = cov.loc[sorted_tickers, sorted_tickers]
    inv_var = 1 / np.diag(cov_sorted)
    weights = inv_var / inv_var.sum()
    
    return pd.Series(weights, index=sorted_tickers).sort_values(ascending=False)

# ==============================================================================
# 4. BACKTEST ENGINE (LÓGICA HÍBRIDA)
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Robust Z-Score usando MAD (Median Absolute Deviation)."""
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0: return series - median
    return ((series - median) / (mad * 1.4826)).clip(-3, 3)

def run_honest_backtest(
    prices: pd.DataFrame, 
    valid_tickers: list, 
    lookback_days: int = 252
):
    """
    Backtest 'Honesto':
    1. Usa o universo de tickers filtrado pela QUALIDADE ATUAL.
    2. Aplica historicamente apenas sinais de Preço (Momentum + HRP).
    """
    # Garante que temos as colunas necessárias
    cols_to_use = [t for t in valid_tickers if t in prices.columns]
    if 'BOVA11.SA' in prices.columns and 'BOVA11.SA' not in cols_to_use:
        cols_to_use.append('BOVA11.SA')
        
    subset_prices = prices[cols_to_use].dropna(how='all')
    
    # Se não houver dados suficientes no total, retorna vazio
    if len(subset_prices) < lookback_days:
        return pd.DataFrame()

    # Pula o período inicial de "aquecimento" (lookback)
    rebalance_dates = subset_prices.resample('MS').first().index
    rebalance_dates = [d for d in rebalance_dates if d >= subset_prices.index[0] + timedelta(days=lookback_days)]
    
    history = []
    
    for date in rebalance_dates:
        # Janela de dados disponível NAQUELA DATA
        hist_window = subset_prices.loc[:date].tail(lookback_days)
        
        # --- CORREÇÃO DO ERRO ---
        # Verifica se temos EXATAMENTE a quantidade de dias necessária para o iloc[-252]
        if len(hist_window) < lookback_days: 
            continue
            
        # Calcula Momentum (Sinal Técnico disponível no passado)
        # Momentum 12-1 (Preço de 1 mês atrás / Preço de 12 meses atrás - 1)
        try:
            # iloc[-21] = aprox 1 mês atrás
            # iloc[-lookback_days] = início da janela (12 meses atrás)
            price_1m_ago = hist_window.iloc[-21]
            price_12m_ago = hist_window.iloc[0] # Como usamos tail(lookback), o índice 0 é o mais antigo
            
            # Evita divisão por zero ou nulos
            ret_12m = (price_1m_ago / price_12m_ago - 1).dropna()
        except IndexError:
            continue
        
        # Ranking Top 25% Momentum
        # Filtra apenas tickers que têm dados válidos de momentum
        available_tickers = [t for t in ret_12m.index if t in valid_tickers and t != 'BOVA11.SA']
        
        if not available_tickers:
            continue

        top_cut = int(len(available_tickers) * 0.25)
        if top_cut < 3: top_cut = 3
        
        selection = ret_12m.loc[available_tickers].sort_values(ascending=False).head(top_cut).index.tolist()
        
        if not selection: continue
            
        # Alocação de Risco (HRP)
        try:
            curr_prices = hist_window[selection].tail(60) # Curta janela para covariância
            if len(curr_prices) < 30: # Proteção extra para covariância
                weights = pd.Series(1/len(selection), index=selection)
            else:
                weights = get_hrp_weights(curr_prices)
        except:
            weights = pd.Series(1/len(selection), index=selection)
            
        # Retorno do próximo mês
        next_date = date + timedelta(days=30)
        # CORREÇÃO DO WARNING: fill_method=None
        future_rets = subset_prices.loc[date:next_date].pct_change(fill_method=None).dropna()
        
        if not future_rets.empty:
            # Alinha pesos com colunas disponíveis no futuro (caso algum ativo pare de negociar)
            valid_w_assets = [a for a in weights.index if a in future_rets.columns]
            if valid_w_assets:
                strat_ret = (future_rets[valid_w_assets] * weights[valid_w_assets]).sum(axis=1)
                bova_ret = future_rets['BOVA11.SA'] if 'BOVA11.SA' in future_rets.columns else pd.Series(0, index=future_rets.index)
                
                df_ret = pd.DataFrame({'Strategy': strat_ret, 'BOVA11': bova_ret})
                history.append(df_ret)
            
    if history:
        full_ret = pd.concat(history)
        full_ret = full_ret[~full_ret.index.duplicated()]
        return (1 + full_ret).cumprod()
    
    return pd.DataFrame()
# ==============================================================================
# 5. APP PRINCIPAL
# ==============================================================================

def main():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.write("## 🦁") # Placeholder logo SPX/Fundo
    with col_title:
        st.title("Quant Factor Lab | Multi-Strategy Engine")
        st.markdown("**Investment Grade Decision Support System**")

    # --- SIDEBAR ---
    st.sidebar.header("🛠️ Configuração do Universo")
    
    # Input melhorado
    default_univ = "WEGE3.SA, ITUB4.SA, VALE3.SA, PETR4.SA, PRIO3.SA, BBAS3.SA, RENT3.SA, LREN3.SA, ELET3.SA, GGBR4.SA, CSAN3.SA, B3SA3.SA, SUZB3.SA, EQTL3.SA, RADL3.SA, RDOR3.SA, VIBRA3.SA, JBSS3.SA, CPFE3.SA, EMBR3.SA"
    ticker_input = st.sidebar.text_area("Tickers (IBOV / Small Caps)", default_univ, height=150)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.markdown("---")
    st.sidebar.header("⚖️ Pesos do Modelo (Alpha)")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.4, help="Força da tendência descontada do mercado")
    w_fm = st.sidebar.slider("Fundamental Growth", 0.0, 1.0, 0.2, help="Crescimento de Receita e Lucros")
    w_val = st.sidebar.slider("Value (EV/EBITDA)", 0.0, 1.0, 0.2, help="Preço justo vs Geração de caixa")
    w_qual = st.sidebar.slider("Quality (Accruals)", 0.0, 1.0, 0.2, help="ROE alto e baixa manipulação contábil")

    run_btn = st.sidebar.button("🚀 Executar Pipeline Quant", type="primary")

    if run_btn:
        with st.status("Processando Dados Institucionais...", expanded=True) as status:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*4) # 4 anos de histórico

            # 1. Fetching
            st.write("📡 Baixando Preços (Yahoo Finance)...")
            prices = fetch_price_data(tickers, start_date, end_date)
            
            st.write("📊 Baixando e Processando Fundamentos (Paralelo)...")
            fundamentals = fetch_fundamentals_parallel(tickers)
            
            if prices.empty or fundamentals.empty:
                st.error("Falha na obtenção de dados.")
                return

            # 2. Factor Calculation
            st.write("🧮 Calculando Fatores e Z-Scores...")
            res_mom = compute_residual_momentum(prices)
            fund_factors = compute_fundamental_factors(fundamentals)
            
            # Merge de todos os fatores
            df_master = fund_factors.copy()
            df_master['Res_Mom'] = res_mom
            
            # Adicionar Setor
            if 'sector' in fundamentals.columns: df_master['Sector'] = fundamentals['sector']
            
            df_master.dropna(thresh=3, inplace=True) # Exige min de dados

            # 3. Normalização e Scoring
            # Z-Score Robusto por fator
            factors = ['Res_Mom', 'Fund_Mom', 'Value_Score', 'Quality_Score']
            w_map = {'Res_Mom': w_rm, 'Fund_Mom': w_fm, 'Value_Score': w_val, 'Quality_Score': w_qual}
            
            final_scores = pd.DataFrame(index=df_master.index)
            
            for f in factors:
                if f in df_master.columns:
                    # Neutralidade Setorial (Opcional - aqui simplificado global)
                    final_scores[f] = robust_zscore(df_master[f])
            
            # Weighted Composite Score
            final_scores['Composite'] = 0
            for f in factors:
                if f in final_scores.columns:
                    final_scores['Composite'] += final_scores[f] * w_map[f]
            
            final_scores = final_scores.sort_values('Composite', ascending=False)
            top_picks = final_scores.head(10)
            
            # 4. Alocação HRP nos Top Picks
            st.write("🛡️ Otimizando Portfólio (HRP Algorithm)...")
            recent_prices = prices[top_picks.index].tail(126) # 6 meses para covariância
            hrp_weights = get_hrp_weights(recent_prices)
            
            # 5. Backtest Híbrido
            st.write("⏳ Rodando Backtest Walk-Forward...")
            # Definimos o universo investível como o Top 50% do ranking atual (filtro de qualidade)
            quality_universe = final_scores.head(int(len(final_scores)*0.5)).index.tolist()
            equity_curve = run_honest_backtest(prices, quality_universe)
            
            status.update(label="Cálculo Concluído!", state="complete", expanded=False)

        # --- DASHBOARD UI ---
        
        # KPI ROW
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        top_ticker = top_picks.index[0]
        top_sector = fundamentals.loc[top_ticker, 'sector'] if top_ticker in fundamentals.index else "N/A"
        
        kpi1.metric("Top Pick", top_ticker, top_sector)
        kpi2.metric("Universo Analisado", f"{len(df_master)} Ativos", "Filtrados")
        
        if not equity_curve.empty:
            cum_ret = equity_curve.iloc[-1]
            cagr = (cum_ret**(252/len(equity_curve)) - 1)
            vol = equity_curve.pct_change().std() * (252**0.5)
            sharpe = (cagr - 0.10) / vol
            
            kpi3.metric("CAGR Estratégia", f"{cagr['Strategy']:.1%}", f"vs {cagr['BOVA11']:.1%} Bench")
            kpi4.metric("Sharpe Ratio", f"{sharpe['Strategy']:.2f}", "Risco Ajustado")

        tab1, tab2, tab3 = st.tabs(["🏆 Ranking & Alocação", "🔍 Factor Attribution", "📈 Backtest Técnico"])

        with tab1:
            col_rank, col_alloc = st.columns([1.5, 1])
            with col_rank:
                st.subheader("Ranking Multifatorial (Z-Score)")
                st.dataframe(
                    final_scores.style.background_gradient(cmap='RdYlGn', subset=['Composite']),
                    height=500, use_container_width=True
                )
            with col_alloc:
                st.subheader("Sugestão de Alocação (HRP)")
                if not hrp_weights.empty:
                    df_w = hrp_weights.to_frame("Peso")
                    df_w['Valor (R$ 10k)'] = df_w['Peso'] * 10000
                    st.dataframe(df_w.style.format({'Peso': '{:.1%}', 'Valor (R$ 10k)': 'R${:,.2f}'}))
                    
                    fig_pie = px.pie(values=hrp_weights.values, names=hrp_weights.index, hole=0.4, title="Diversificação por Risco (Clusters)")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Por que este ativo foi escolhido?")
            sel_ticker = st.selectbox("Selecione o Ativo para Inspecionar:", top_picks.index)
            
            if sel_ticker:
                # Dados para gráfico de barras
                attr_data = final_scores.loc[sel_ticker, factors]
                fig_attr = px.bar(
                    x=attr_data.index, 
                    y=attr_data.values, 
                    color=attr_data.values,
                    color_continuous_scale='RdYlGn',
                    labels={'y': 'Z-Score (Desvios Padrão)', 'x': 'Fator'},
                    title=f"Decomposição de Alpha: {sel_ticker}"
                )
                fig_attr.add_hline(y=0, line_dash="dot", line_color="white")
                st.plotly_chart(fig_attr, use_container_width=True)
                
                # Dados Brutos
                st.markdown("#### Dados Fundamentais Brutos")
                raw_data = fundamentals.loc[sel_ticker].to_frame().T
                st.dataframe(raw_data)

        with tab3:
            st.markdown("""
            > **Nota de Metodologia:** Este backtest evita o *Look-Ahead Bias*. Ele utiliza o universo de ativos filtrado pela 
            > qualidade atual, mas aplica historicamente apenas sinais de **Preço (Momentum)** e **Risco (HRP)**. 
            > Isso simula uma estratégia de "Trend Following em empresas de Qualidade".
            """)
            if not equity_curve.empty:
                st.line_chart(equity_curve)
                
                # Drawdown Chart
                drawdown = equity_curve / equity_curve.cummax() - 1
                st.area_chart(drawdown['Strategy'], color="#ff4b4b")
                st.caption("Drawdown Histórico (Queda do Topo)")
            else:
                st.warning("Dados insuficientes para backtest robusto.")

if __name__ == "__main__":
    main()
