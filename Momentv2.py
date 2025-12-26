import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING (Busca de Dados)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados, garantindo o benchmark BOVA11.SA."""
    t_list = list(tickers)
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
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """Busca snapshots fundamentais atuais com métricas expandidas para Value Composto."""
    data = []
    clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
    
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            sector = info.get('sector', 'Unknown')
            
            # Fallback básico para setor
            if sector in ['Unknown', 'N/A'] and 'longName' in info:
                 if 'Banco' in info['longName'] or 'Financeira' in info['longName']:
                     sector = 'Financial Services'
            
            # Coleta de métricas para Value Composto
            data.append({
                'ticker': t,
                'sector': sector,
                'forwardPE': info.get('forwardPE', np.nan),
                'trailingPE': info.get('trailingPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'freeCashflow': info.get('freeCashflow', np.nan),
                'marketCap': info.get('marketCap', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan)
            })
        except:
            pass
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data).set_index('ticker')
    # Calcular Cash Flow Yield se possível
    if 'freeCashflow' in df.columns and 'marketCap' in df.columns:
        df['cashFlowYield'] = df['freeCashflow'] / df['marketCap']
    
    return df

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=36, skip=1) -> pd.Series:
    """
    Calcula Residual Momentum Clássico (Blitz et al.):
    1. Regressão de 36 meses (lookback) vs Mercado.
    2. Rankear pelo t-stat do intercepto ou retornos residuais.
    3. Adiciona Volatility Scaling (Barroso-Santa-Clara).
    """
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns or len(rets) < lookback: 
        return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    vols = {}
    
    # Janela de regressão (ex: 36 meses)
    # Skip de 1 mês para evitar reversão de curto prazo
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        
        # Dados para regressão (lookback meses, pulando os últimos 'skip' meses)
        y = rets[ticker].tail(lookback + skip).iloc[:-skip]
        x = market.tail(lookback + skip).iloc[:-skip]
        
        if len(y) < lookback: continue
            
        try:
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            
            # Usamos o t-stat do intercepto (Alpha) como score de momentum residual
            # Isso penaliza alphas com alta incerteza
            scores[ticker] = model.tvalues[0]
            
            # Volatility Scaling: Inverso da volatilidade dos resíduos recentes (6 meses)
            recent_resid = model.resid[-6:]
            vol = np.std(recent_resid)
            vols[ticker] = 1.0 / vol if vol > 0 else 0
            
        except:
            scores[ticker] = 0
            vols[ticker] = 0
            
    s_scores = pd.Series(scores)
    s_vols = pd.Series(vols)
    
    # Escalonamento: Score * Vol_Inv
    final_res_mom = s_scores * s_vols
    return final_res_mom.rename('Residual_Momentum')

def compute_composite_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """
    Score de Valor Composto: Média de múltiplos normalizados.
    Inclui: E/P (Forward & Trailing), B/P, EBITDA/EV, CashFlow Yield.
    """
    df = fund_df.copy()
    # Converter colunas para numérico para evitar TypeErrors
    for col in df.columns:
        if col != 'sector':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    val_metrics = pd.DataFrame(index=df.index)
    
    # Invertemos os múltiplos para que "maior é melhor" (Barato)
    if 'forwardPE' in df.columns: val_metrics['EP_Forward'] = np.where(df['forwardPE'] > 0, 1/df['forwardPE'], np.nan)
    if 'trailingPE' in df.columns: val_metrics['EP_Trailing'] = np.where(df['trailingPE'] > 0, 1/df['trailingPE'], np.nan)
    if 'priceToBook' in df.columns: val_metrics['BP'] = np.where(df['priceToBook'] > 0, 1/df['priceToBook'], np.nan)
    if 'enterpriseToEbitda' in df.columns: val_metrics['EBITDA_EV'] = np.where(df['enterpriseToEbitda'] > 0, 1/df['enterpriseToEbitda'], np.nan)
    if 'cashFlowYield' in df.columns: val_metrics['CF_Yield'] = df['cashFlowYield']
    
    # Normalizar cada métrica antes de tirar a média
    for col in val_metrics.columns:
        val_metrics[col] = robust_zscore(val_metrics[col])
        
    return val_metrics.mean(axis=1).rename("Value_Score")

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    """Z-Score combinado de crescimento de Receita e Lucro."""
    metrics = ['earningsGrowth', 'revenueGrowth']
    temp_df = pd.DataFrame(index=fund_df.index)
    for m in metrics:
        if m in fund_df.columns:
            s = fund_df[m].fillna(fund_df[m].median())
            temp_df[m] = robust_zscore(s)
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: scores['DE_Inv'] = -1 * fund_df['debtToEquity']
    
    for col in scores.columns:
        scores[col] = robust_zscore(scores[col])
        
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto usando Mediana e MAD."""
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isnull().all(): return series
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: 
        # Fallback para desvio padrão se MAD for zero
        std = series.std()
        if std == 0 or np.isnan(std): return series.fillna(0) * 0
        return (series - median) / std
    z = (series - median) / (mad * 1.4826) 
    return z.clip(-3, 3) 

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calcula score final ponderado."""
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
            
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: FACTOR TIMING & PROJEÇÃO
# ==============================================================================

def analyze_factor_timing(all_prices: pd.DataFrame, all_fundamentals: pd.DataFrame, lookback_years=3):
    """
    Gera gráfico de performance histórica por fator.
    Simula o retorno de cada fator isoladamente nos últimos anos.
    """
    end_date = all_prices.index[-1]
    start_date = end_date - timedelta(days=lookback_years*365)
    
    # Resample mensal para agilizar
    monthly_prices = all_prices.loc[start_date:end_date].resample('ME').last()
    monthly_rets = monthly_prices.pct_change().dropna()
    
    factor_performances = {}
    
    # Fatores para testar
    factors = {
        'Residual_Momentum': lambda p: compute_residual_momentum(p),
        'Value': lambda p: compute_composite_value_score(all_fundamentals),
        'Quality': lambda p: compute_quality_score(all_fundamentals),
        'Fund_Momentum': lambda p: compute_fundamental_momentum(all_fundamentals)
    }
    
    # Simplificação: Calculamos os scores uma vez (atuais) e vemos como esses ativos performaram
    # Para um Factor Timing real, precisaríamos de dados fundamentais históricos (Point-in-Time),
    # mas como o yfinance só dá o snapshot atual, simularemos a performance recente dos top picks atuais de cada fator.
    
    for f_name, f_func in factors.items():
        try:
            if f_name == 'Residual_Momentum':
                scores = f_func(all_prices.loc[:end_date])
            else:
                scores = f_func(None)
            
            top_tickers = scores.sort_values(ascending=False).head(10).index.tolist()
            valid_tickers = [t for t in top_tickers if t in monthly_rets.columns]
            
            if valid_tickers:
                # Retorno médio equiponderado dos top picks deste fator
                factor_rets = monthly_rets[valid_tickers].mean(axis=1)
                factor_performances[f_name] = (1 + factor_rets).cumprod()
        except:
            continue
            
    if 'BOVA11.SA' in monthly_rets.columns:
        factor_performances['Benchmark'] = (1 + monthly_rets['BOVA11.SA']).cumprod()
        
    return pd.DataFrame(factor_performances)

def project_dca_future(current_balance: float, monthly_deposit: float, years: int, expected_return: float):
    """
    Projeção de patrimônio futuro.
    """
    months = years * 12
    monthly_ret = (1 + expected_return) ** (1/12) - 1
    
    balances = []
    balance = current_balance
    for m in range(months + 1):
        balances.append(balance)
        balance = balance * (1 + monthly_ret) + monthly_deposit
        
    return pd.Series(balances)

# (Restante do código de backtest e interface será adaptado na Fase 4)

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio e ordena do maior para o menor peso."""
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    if vol_target is not None:
        recent_rets = prices[selected].pct_change().tail(63)
        vols = recent_rets.std() * (252**0.5)
        vols[vols == 0] = 1e-6 
        raw_weights_inv = 1 / vols
        weights = raw_weights_inv / raw_weights_inv.sum() 
    else:
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights.sort_values(ascending=False)

def run_dynamic_backtest(
    all_prices: pd.DataFrame, 
    all_fundamentals: pd.DataFrame, 
    weights_config: dict, 
    top_n: int, 
    use_vol_target: bool,
    use_sector_neutrality: bool,
    start_date_backtest: datetime
):
    """Executa um backtest mês a mês (Walk-Forward)."""
    end_date = all_prices.index[-1]
    subset_prices = all_prices.loc[start_date_backtest - timedelta(days=400):end_date]
    rebalance_dates = subset_prices.loc[start_date_backtest:end_date].resample('MS').first().index.tolist()
    
    if not rebalance_dates:
        return pd.DataFrame()

    strategy_daily_rets = []
    benchmark_daily_rets = []

    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
        prices_historical = subset_prices.loc[:rebal_date]
        mom_window = prices_historical.tail(400) 
        risk_window = prices_historical.tail(90)
        
        # Cálculo dos fatores com as novas melhorias
        res_mom = compute_residual_momentum(mom_window)
        fund_mom = compute_fundamental_momentum(all_fundamentals)
        val_score = compute_composite_value_score(all_fundamentals)
        qual_score = compute_quality_score(all_fundamentals)
        
        df_period = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_period['Res_Mom'] = res_mom
        df_period['Fund_Mom'] = fund_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        
        if 'sector' in all_fundamentals.columns: 
             df_period['Sector'] = all_fundamentals['sector']
        
        df_period.dropna(thresh=2, inplace=True)
        norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
        w_keys = {}
        
        if use_sector_neutrality and 'Sector' in df_period.columns and df_period['Sector'].nunique() > 1:
            for c in norm_cols:
                if c in df_period.columns:
                    new_col = f"{c}_Z"
                    df_period[new_col] = df_period.groupby('Sector')[c].transform(
                         lambda x: robust_zscore(x) if len(x) > 1 else x - x.median()
                    )
                    w_keys[new_col] = weights_config.get(c, 0.0)
        else:
            for c in norm_cols:
                if c in df_period.columns:
                    new_col = f"{c}_Z"
                    df_period[new_col] = robust_zscore(df_period[c])
                    w_keys[new_col] = weights_config.get(c, 0.0)
                    
        ranked_period = build_composite_score(df_period, w_keys)
        current_weights = construct_portfolio(ranked_period, risk_window, top_n, 0.15 if use_vol_target else None)
        
        market_period = subset_prices.loc[rebal_date:next_date].iloc[1:] 
        period_pct = market_period.pct_change().dropna()
        if period_pct.empty: continue
            
        valid_tickers = [t for t in current_weights.index if t in period_pct.columns]
        if valid_tickers:
            strat_rets = (period_pct[valid_tickers] * current_weights[valid_tickers]).sum(axis=1)
            strategy_daily_rets.append(strat_rets)
            if 'BOVA11.SA' in period_pct.columns:
                benchmark_daily_rets.append(period_pct['BOVA11.SA'])

    if not strategy_daily_rets: return pd.DataFrame()
    
    final_strat = pd.concat(strategy_daily_rets)
    final_bench = pd.concat(benchmark_daily_rets)
    
    df_backtest = pd.DataFrame({
        'Strategy': (1 + final_strat).cumprod(),
        'BOVA11.SA': (1 + final_bench).cumprod()
    })
    return df_backtest

def run_dca_simulation(backtest_df: pd.DataFrame, monthly_investment: float):
    """Simula evolução patrimonial com aportes mensais."""
    if backtest_df.empty: return pd.DataFrame(), pd.DataFrame(), {}
    
    # Pegar datas de aporte (primeiro dia útil de cada mês)
    dca_dates = backtest_df.resample('MS').first().index
    
    strategy_returns = backtest_df['Strategy'].pct_change().fillna(0)
    benchmark_returns = backtest_df['BOVA11.SA'].pct_change().fillna(0)
    
    strat_balance = 0
    bench_balance = 0
    
    strat_history = []
    bench_history = []
    transactions = []
    
    for date in backtest_df.index:
        # Aplicar retorno do dia
        strat_balance *= (1 + strategy_returns.loc[date])
        bench_balance *= (1 + benchmark_returns.loc[date])
        
        # Se for dia de aporte
        if date in dca_dates:
            strat_balance += monthly_investment
            bench_balance += monthly_investment
            transactions.append({'Date': date, 'Amount': monthly_investment})
            
        strat_history.append(strat_balance)
        bench_history.append(bench_balance)
        
    dca_curve = pd.DataFrame({
        'Strategy_DCA': strat_history,
        'BOVA11.SA_DCA': bench_history
    }, index=backtest_df.index)
    
    return dca_curve, pd.DataFrame(transactions), {}

# ==============================================================================
# MÓDULO 5: INTERFACE (Streamlit)
# ==============================================================================

def main():
    # Custom CSS para melhorar o visual
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7f9;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚀 Quant Factor Lab Pro v2")
    st.markdown("### Sistema Quantitativo de Seleção de Ativos")
    st.info("Esta ferramenta utiliza modelos de **Residual Momentum** (Blitz et al.) e **Value Composto** para identificar as melhores oportunidades no mercado brasileiro.")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2620/2620582.png", width=100)
        st.header("⚙️ Painel de Controle")
        
        universe_option = st.selectbox("Universo de Ativos", ["Ibovespa (Top 50)", "Personalizado"])
        if universe_option == "Ibovespa (Top 50)":
            tickers = [
                'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 'BBAS3.SA', 'B3SA3.SA', 'WEGE3.SA',
                'RENT3.SA', 'SUZB3.SA', 'GGBR4.SA', 'ITSA4.SA', 'JBSS3.SA', 'RDOR3.SA', 'RAIL3.SA', 'EQTL3.SA',
                'CSAN3.SA', 'VIVT3.SA', 'LREN3.SA', 'PRIO3.SA', 'RADL3.SA', 'HAPV3.SA', 'BPAC11.SA', 'ELET3.SA',
                'VBBR3.SA', 'SBSP3.SA', 'CCRO3.SA', 'CMIG4.SA', 'HYPE3.SA', 'CPLE6.SA', 'UGPA3.SA', 'SANB11.SA',
                'EGIE3.SA', 'TOTS3.SA', 'TRPL4.SA', 'CSNA3.SA', 'ENEV3.SA', 'GOAU4.SA', 'CYRE3.SA', 'BRFS3.SA',
                'ALOS3.SA', 'MULT3.SA', 'CRFB3.SA', 'TIMS3.SA', 'EMBR3.SA', 'CPFE3.SA', 'MRVE3.SA', 'CIEL3.SA',
                'BRKM5.SA', 'AZUL4.SA'
            ]
        else:
            tickers_input = st.text_area("Tickers (separados por vírgula)", "VALE3.SA, PETR4.SA, ITUB4.SA")
            tickers = [t.strip() for t in tickers_input.split(",")]

        st.subheader("Pesos dos Fatores")
        w_res_mom = st.slider("Residual Momentum", 0.0, 1.0, 0.4)
        w_val = st.slider("Value Composto", 0.0, 1.0, 0.3)
        w_qual = st.slider("Quality", 0.0, 1.0, 0.2)
        w_fund_mom = st.slider("Fundamental Momentum", 0.0, 1.0, 0.1)
        
        st.subheader("Parâmetros DCA")
        dca_amount = st.number_input("Aporte Mensal (R$)", 100, 100000, 1000)
        dca_years_proj = st.slider("Anos para Projeção", 1, 30, 10)
        
        st.subheader("Backtest")
        top_n = st.number_input("Top N Ativos", 3, 20, 5)
        use_vol_target = st.checkbox("Volatility Scaling (Portfólio)", True)
        use_sector_neutrality = st.checkbox("Neutralidade Setorial", True)

    if st.button("🚀 Executar Análise"):
        with st.status("Processando dados...", expanded=True) as status:
            st.write("Buscando preços...")
            prices = fetch_price_data(tickers, "2018-01-01", datetime.now().strftime("%Y-%m-%d"))
            
            st.write("Buscando fundamentos...")
            fundamentals = fetch_fundamentals(tickers)
            
            st.write("Calculando fatores...")
            res_mom = compute_residual_momentum(prices)
            val_score = compute_composite_value_score(fundamentals)
            qual_score = compute_quality_score(fundamentals)
            fund_mom = compute_fundamental_momentum(fundamentals)
            
            # Master DF
            final_df = pd.DataFrame(index=fundamentals.index)
            final_df['Res_Mom'] = res_mom.reindex(fundamentals.index)
            final_df['Value'] = val_score.reindex(fundamentals.index)
            final_df['Quality'] = qual_score.reindex(fundamentals.index)
            final_df['Fund_Mom'] = fund_mom.reindex(fundamentals.index)
            
            # Correção segura para a coluna 'sector'
            if 'sector' in fundamentals.columns:
                final_df['Sector'] = fundamentals['sector']
            else:
                final_df['Sector'] = 'Unknown'
            
            # Normalização
            cols_to_norm = ['Res_Mom', 'Value', 'Quality', 'Fund_Mom']
            for c in cols_to_norm:
                final_df[f"{c}_Z"] = robust_zscore(final_df[c])
            
            weights_config = {
                'Res_Mom_Z': w_res_mom,
                'Value_Z': w_val,
                'Quality_Z': w_qual,
                'Fund_Mom_Z': w_fund_mom
            }
            
            ranked_df = build_composite_score(final_df, weights_config)
            
            st.write("Executando Backtest...")
            backtest_1yr = run_dynamic_backtest(prices, fundamentals, {k.replace('_Z', ''): v for k, v in weights_config.items()}, top_n, use_vol_target, use_sector_neutrality, datetime.now() - timedelta(days=365))
            
            st.write("Simulando DCA...")
            dca_curve, dca_trans, _ = run_dca_simulation(backtest_1yr, dca_amount)
            
            st.write("Analisando Factor Timing...")
            timing_df = analyze_factor_timing(prices, fundamentals)
            
            status.update(label="Análise Concluída!", state="complete", expanded=False)

        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Principal", "📈 Factor Timing", "🔮 Projeção DCA", "🔍 Detalhes Técnicos"])
        
        with tab1:
            # Métricas de Resumo
            m1, m2, m3, m4 = st.columns(4)
            if not backtest_1yr.empty:
                ret_12m = backtest_1yr['Strategy'].iloc[-1] - 1
                bench_12m = backtest_1yr['BOVA11.SA'].iloc[-1] - 1
                m1.metric("Retorno Estratégia (12m)", f"{ret_12m:.2%}", delta=f"{(ret_12m - bench_12m):.2%} vs Bench")
                
                daily_rets = backtest_1yr['Strategy'].pct_change().dropna()
                vol_ann = daily_rets.std() * (252**0.5)
                m2.metric("Volatilidade Anualizada", f"{vol_ann:.2%}")
                
                sharpe = (ret_12m - 0.10) / vol_ann if vol_ann > 0 else 0
                m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                m4.metric("Ativos Selecionados", f"{top_n}")

            st.markdown("---")
            
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("🏆 Top Picks (Ranking de Fatores)")
                # Formatação amigável da tabela
                display_ranked = ranked_df[['Composite_Score', 'Sector', 'Res_Mom', 'Value', 'Quality']].head(top_n).copy()
                display_ranked.columns = ['Score Final', 'Setor', 'Momentum Residual', 'Valor Composto', 'Qualidade']
                st.dataframe(display_ranked.style.background_gradient(cmap='RdYlGn', subset=['Score Final']), width='stretch', height=400)
                
                if not backtest_1yr.empty:
                    st.subheader("📈 Curva de Equidade (Últimos 12 Meses)")
                    fig_perf = px.line(backtest_1yr, labels={'value': 'Retorno Acumulado', 'Date': 'Data'}, color_discrete_map={'Strategy': '#00CC96', 'BOVA11.SA': '#EF553B'})
                    fig_perf.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_perf, width='stretch')
            
            with col2:
                st.subheader("💼 Alocação da Carteira")
                weights = construct_portfolio(ranked_df, prices.tail(90), top_n, 0.15 if use_vol_target else None)
                
                # Tabela de Rebalanceamento
                latest_prices = prices.iloc[-1]
                rebal_df = pd.DataFrame(index=weights.index)
                rebal_df['Peso'] = weights.values
                rebal_df['Preço'] = latest_prices.reindex(weights.index).values
                rebal_df['Alocação (R$)'] = rebal_df['Peso'] * dca_amount
                # Garantir que os preços sejam numéricos e tratar divisões por zero ou NaN
                rebal_df['Preço'] = pd.to_numeric(rebal_df['Preço'], errors='coerce')
                rebal_df['Qtd. Sugerida'] = (rebal_df['Alocação (R$)'] / rebal_df['Preço'].replace(0, np.nan))
                rebal_df['Qtd. Sugerida'] = pd.to_numeric(rebal_df['Qtd. Sugerida'], errors='coerce').fillna(0).round(0)
                
                st.table(rebal_df[['Peso', 'Preço', 'Qtd. Sugerida']].style.format({'Peso': '{:.1%}', 'Preço': 'R$ {:.2f}', 'Qtd. Sugerida': '{:.0f}'}))
                
                fig_pie = px.pie(values=weights.values, names=weights.index, hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(showlegend=True, legend=dict(orientation="h"))
                st.plotly_chart(fig_pie, width='stretch')

        with tab2:
            st.subheader("Factor Timing: Performance Histórica por Fator")
            st.markdown("Este gráfico mostra como os ativos que hoje são 'Top Picks' de cada fator performaram nos últimos 3 anos.")
            if not timing_df.empty:
                st.plotly_chart(px.line(timing_df, title="Evolução dos Fatores (Base 1.0)"), width='stretch')
            else:
                st.warning("Dados insuficientes para análise de timing.")

        with tab3:
            st.subheader("Projeção de Patrimônio Futuro")
            avg_annual_ret = (backtest_1yr['Strategy'].iloc[-1] ** (1)) - 1 if not backtest_1yr.empty else 0.15
            
            c1, c2, c3 = st.columns(3)
            exp_ret = c1.slider("Retorno Esperado Anual (%)", 0.0, 40.0, float(avg_annual_ret * 100)) / 100
            proj_years = c2.number_input("Anos de Projeção", 1, 50, dca_years_proj)
            init_bal = c3.number_input("Saldo Inicial (R$)", 0, 1000000, 0)
            
            projection = project_dca_future(init_bal, dca_amount, proj_years, exp_ret)
            
            fig_proj = px.area(projection, title=f"Projeção em {proj_years} anos (R$ {projection.iloc[-1]:,.2f})")
            fig_proj.update_layout(showlegend=False, yaxis_title="Patrimônio (R$)", xaxis_title="Meses")
            st.plotly_chart(fig_proj, width='stretch')
            
            st.info(f"Com um aporte de R$ {dca_amount:,.2f} e retorno de {exp_ret:.2%}, seu patrimônio estimado será de R$ {projection.iloc[-1]:,.2f}")

        with tab4:
            st.subheader("Dados Brutos e Correlação")
            corr = final_df[['Res_Mom', 'Value', 'Quality', 'Fund_Mom']].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), width='stretch')
            st.dataframe(fundamentals)

if __name__ == "__main__":
    main()
