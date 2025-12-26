import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v3 (Monte Carlo & Risk)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING & VALIDAÇÃO
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados, garantindo benchmarks."""
    t_list = list(tickers)
    # Garante benchmarks
    for bench in ['BOVA11.SA', 'SMAL11.SA']:
        if bench not in t_list:
            t_list.append(bench)
    
    try:
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False,
            threads=True
        )['Adj Close']
        
        # Tratamento para download de único ticker vs múltiplos
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Validação de Tickers
        downloaded = set(data.columns)
        requested = set(tickers)
        missing = requested - downloaded
        
        # Remove colunas vazias (tickers que retornaram NaN total)
        data = data.dropna(axis=1, how='all')
        
        if missing:
            st.warning(f"⚠️ Tickers não encontrados ou sem dados: {', '.join(missing)}")
            
        return data
    except Exception as e:
        st.error(f"Erro crítico ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """Busca snapshots fundamentais com tratamento de erro granular."""
    data = []
    clean_tickers = [t for t in tickers if t not in ['BOVA11.SA', 'SMAL11.SA']]
    failed_tickers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        status_text.text(f"Baixando fundamentos: {t}")
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            
            # Validação mínima: se não tem setor nem preço, provavelmente falhou
            if 'sector' not in info and 'regularMarketPrice' not in info:
                raise ValueError("Dados vazios")

            sector = info.get('sector', 'Unknown')
            if sector in ['Unknown', 'N/A'] and 'longName' in info:
                 if 'Banco' in info['longName'] or 'Financeira' in info['longName']:
                     sector = 'Financial Services'
            
            # Dados para Value Composite
            mkt_cap = info.get('marketCap', np.nan)
            op_cashflow = info.get('operatingCashflow', np.nan)
            ocf_yield = (op_cashflow / mkt_cap) if (mkt_cap and op_cashflow and mkt_cap > 0) else np.nan

            data.append({
                'ticker': t,
                'sector': sector,
                'currentPrice': info.get('currentPrice', np.nan),
                # Value Metrics
                'forwardPE': info.get('forwardPE', np.nan),
                'trailingPE': info.get('trailingPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'ocfYield': ocf_yield,
                # Quality/Growth Metrics
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan)
            })
        except Exception as e:
            failed_tickers.append(t)
            # Adiciona placeholder para não quebrar o código de backtest, mas penaliza
            data.append({'ticker': t, 'sector': 'Unknown'}) 
            
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    status_text.empty()
    
    if failed_tickers:
        with st.expander(f"⚠️ Aviso: Falha em {len(failed_tickers)} tickers (Dados Fundamentais)", expanded=False):
            st.write(f"Não foi possível obter dados detalhados para: {', '.join(failed_tickers)}. Eles terão Score neutro/zerado.")

    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).set_index('ticker')

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES
# ==============================================================================

def compute_residual_momentum_enhanced(price_df: pd.DataFrame, lookback=36, skip=1) -> pd.Series:
    """Residual Momentum Clássico (Blitz) com Volatility Scaling."""
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    formation_window = 12 
    
    for ticker in rets.columns:
        if ticker in ['BOVA11.SA', 'SMAL11.SA']: continue
        
        y_full = rets[ticker].tail(lookback + skip)
        x_full = market.tail(lookback + skip)
        
        if len(y_full) < 12: continue
            
        try:
            X = sm.add_constant(x_full.values)
            model = sm.OLS(y_full.values, X).fit()
            residuals = pd.Series(model.resid, index=y_full.index)
            resid_12m = residuals.iloc[-(formation_window + skip) : -skip]
            
            if len(resid_12m) == 0:
                scores[ticker] = 0
                continue

            raw_momentum = resid_12m.sum()
            resid_vol = residuals.std()
            
            if resid_vol == 0:
                scores[ticker] = 0
            else:
                scores[ticker] = raw_momentum / resid_vol # Vol Scaling
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_value_robust(fund_df: pd.DataFrame) -> pd.Series:
    """Composite Value Score."""
    scores = pd.DataFrame(index=fund_df.index)
    
    def invert_metric(series):
        # Transforma métricas onde 'menor é melhor'
        valid = np.where(series > 0, 1/series, np.nan) 
        return pd.Series(valid, index=series.index)

    if 'forwardPE' in fund_df: scores['EP_Fwd'] = invert_metric(fund_df['forwardPE'])
    if 'trailingPE' in fund_df: scores['EP_Trail'] = invert_metric(fund_df['trailingPE'])
    if 'priceToBook' in fund_df: scores['BP'] = invert_metric(fund_df['priceToBook'])
    if 'enterpriseToEbitda' in fund_df: scores['EBITDA_Yld'] = invert_metric(fund_df['enterpriseToEbitda'])
    if 'ocfYield' in fund_df: scores['OCF_Yld'] = fund_df['ocfYield']

    for col in scores.columns:
        filled = scores[col].fillna(scores[col].min()) 
        scores[col] = (filled - filled.mean()) / filled.std()

    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: scores['DE_Inv'] = -1 * fund_df['debtToEquity']
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: MATEMÁTICA E MÉTRICAS
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: return series - median 
    z = (series - median) / (mad * 1.4826) 
    return z.clip(-3, 3) 

def calculate_advanced_metrics(prices_series: pd.Series, risk_free_rate_annual: float = 0.10):
    """Calcula Sharpe, Sortino, Calmar e Ulcer Index Corretamente."""
    if prices_series.empty or len(prices_series) < 2:
        return {}
    
    # Retornos Diários
    daily_rets = prices_series.pct_change().dropna()
    
    # 1. Retorno Total & CAGR
    total_ret = (prices_series.iloc[-1] / prices_series.iloc[0]) - 1
    days = (prices_series.index[-1] - prices_series.index[0]).days
    cagr = (1 + total_ret)**(365/days) - 1 if days > 0 else 0
    
    # 2. Volatilidade Anualizada
    vol_ann = daily_rets.std() * np.sqrt(252)
    
    # 3. Sharpe Ratio (Excesso de retorno sobre RF)
    rf_daily = (1 + risk_free_rate_annual)**(1/252) - 1
    excess_rets = daily_rets - rf_daily
    sharpe = (excess_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    
    # 4. Sortino Ratio (Penaliza apenas vol negativa)
    downside_rets = excess_rets[excess_rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (excess_rets.mean() * 252) / downside_std if downside_std > 0 else 0
    
    # 5. Drawdown & Calmar Ratio
    cum_rets = (1 + daily_rets).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # 6. Ulcer Index (RMS do Drawdown - mede profundidade e duração da dor)
    ulcer_index = np.sqrt((drawdown**2).mean())
    
    return {
        'Retorno Total': total_ret,
        'CAGR': cagr,
        'Volatilidade': vol_ann,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_dd,
        'Ulcer Index': ulcer_index
    }

def calculate_market_breadth(prices: pd.DataFrame, benchmark='BOVA11.SA'):
    """Calcula % de ações acima da média móvel de 200 dias."""
    sma200 = prices.rolling(window=200).mean().iloc[-1]
    curr_price = prices.iloc[-1]
    
    # Remove o benchmark da conta
    valid_tickers = [t for t in prices.columns if t != benchmark]
    if not valid_tickers: return 0, 0
    
    above_sma = (curr_price[valid_tickers] > sma200[valid_tickers]).sum()
    total = len(valid_tickers)
    
    breadth_pct = above_sma / total if total > 0 else 0
    return breadth_pct, total

# ==============================================================================
# MÓDULO 4: SIMULAÇÃO MONTE CARLO
# ==============================================================================

def run_monte_carlo(
    initial_balance: float, 
    monthly_contrib: float, 
    mu_annual: float, 
    sigma_annual: float, 
    years: int, 
    simulations: int = 1000
):
    """Projeção Probabilística de Patrimônio."""
    months = years * 12
    dt = 1/12
    
    # Matriz de Retornos Aleatórios (Normal Distribution)
    # mu - 0.5*sigma^2 é a correção geométrica para Log-Normal
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    shock = sigma_annual * np.sqrt(dt) * np.random.normal(0, 1, (months, simulations))
    
    # Retornos mensais
    monthly_returns = np.exp(drift + shock) - 1
    
    # Simulação Loopada (necessária por causa do aporte mensal constante)
    portfolio_paths = np.zeros((months + 1, simulations))
    portfolio_paths[0] = initial_balance
    
    for t in range(1, months + 1):
        # Valor anterior * (1 + retorno) + aporte
        portfolio_paths[t] = portfolio_paths[t-1] * (1 + monthly_returns[t-1]) + monthly_contrib
        
    # Percentis
    percentiles = np.percentile(portfolio_paths, [5, 50, 95], axis=1)
    
    dates = [datetime.now() + timedelta(days=30*i) for i in range(months + 1)]
    
    df_mc = pd.DataFrame({
        'Pessimista (5%)': percentiles[0],
        'Base (50%)': percentiles[1],
        'Otimista (95%)': percentiles[2]
    }, index=dates)
    
    return df_mc

# ==============================================================================
# MÓDULO 5: BACKTEST & ENGINE
# ==============================================================================
# (Funções auxiliares de backtest mantidas similares, mas chamando as novas métricas)

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
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

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
    return df.sort_values('Composite_Score', ascending=False)

def run_dynamic_backtest(all_prices, all_fundamentals, weights_config, top_n, use_vol_target, use_sector_neutrality, start_date_backtest):
    end_date = all_prices.index[-1]
    subset_prices = all_prices.loc[start_date_backtest - timedelta(days=500):end_date]
    rebalance_dates = subset_prices.loc[start_date_backtest:end_date].resample('MS').first().index.tolist()
    
    if not rebalance_dates: return pd.DataFrame(), pd.DataFrame()

    strategy_rets, bench_rets, smal_rets = [], [], []
    factor_tracking = {k: [] for k in ['Res_Mom', 'Value', 'Quality']}

    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
        
        prices_historical = subset_prices.loc[:rebal_date]
        mom_window = prices_historical.tail(500) 
        risk_window = prices_historical.tail(90)
        
        res_mom = compute_residual_momentum_enhanced(mom_window)
        val_score = compute_value_robust(all_fundamentals)
        qual_score = compute_quality_score(all_fundamentals)
        
        df_period = pd.DataFrame(index=all_prices.columns.drop(['BOVA11.SA', 'SMAL11.SA'], errors='ignore'))
        df_period['Res_Mom'] = res_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        if 'sector' in all_fundamentals.columns: df_period['Sector'] = all_fundamentals['sector']
        df_period.dropna(thresh=2, inplace=True)
        
        norm_cols = ['Res_Mom', 'Value', 'Quality']
        w_keys = {}
        
        if use_sector_neutrality and 'Sector' in df_period.columns and df_period['Sector'].nunique() > 1:
            for c in norm_cols:
                if c in df_period.columns:
                    new_col = f"{c}_Z"
                    df_period[new_col] = df_period.groupby('Sector')[c].transform(lambda x: robust_zscore(x) if len(x) > 1 else x - x.median())
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
            strat_ret = period_pct[valid_tickers].dot(current_weights[valid_tickers])
        else:
            strat_ret = pd.Series(0.0, index=period_pct.index)
        
        strategy_rets.append(strat_ret)
        bench_rets.append(period_pct['BOVA11.SA'] if 'BOVA11.SA' in period_pct.columns else pd.Series(0.0, index=period_pct.index))
        smal_rets.append(period_pct['SMAL11.SA'] if 'SMAL11.SA' in period_pct.columns else pd.Series(0.0, index=period_pct.index))

        # Factor Tracking (Puro)
        for factor in ['Res_Mom', 'Value', 'Quality']:
            col_z = f"{factor}_Z"
            if col_z in df_period.columns:
                top_factor = df_period.sort_values(col_z, ascending=False).head(top_n).index
                valid_f = [t for t in top_factor if t in period_pct.columns]
                if valid_f:
                    f_ret = period_pct[valid_f].mean(axis=1) 
                    factor_tracking[factor].append(f_ret)

    if strategy_rets:
        full_strategy = pd.concat(strategy_rets)
        full_benchmark = pd.concat(bench_rets)
        full_smal = pd.concat(smal_rets)
        
        # Alinha índices
        common_idx = full_strategy.index.intersection(full_benchmark.index)
        
        cumulative = pd.DataFrame({
            'Strategy': (1 + full_strategy.loc[common_idx]).cumprod(),
            'BOVA11': (1 + full_benchmark.loc[common_idx]).cumprod(),
            'SMAL11': (1 + full_smal.loc[common_idx]).cumprod()
        })
        
        factor_cum = pd.DataFrame(index=cumulative.index)
        for f, rets_list in factor_tracking.items():
            if rets_list:
                s = pd.concat(rets_list)
                s = s[~s.index.duplicated(keep='first')].reindex(cumulative.index).fillna(0)
                factor_cum[f] = (1 + s).cumprod()
        
        return cumulative.dropna(), factor_cum.dropna()
    return pd.DataFrame(), pd.DataFrame()

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Pro v3 (Monte Carlo & Risk)")
    st.markdown("Otimização Multifator com **Value Robusto**, **Momentum Residual** e **Análise de Risco Avançada**.")

    st.sidebar.header("1. Universo e Dados")
    default_univ = "ITUB3.SA, TOTS3.SA, MDIA3.SA, TAEE3.SA, BBSE3.SA, WEGE3.SA, PSSA3.SA, EGIE3.SA, B3SA3.SA, VIVT3.SA, AGRO3.SA, PRIO3.SA, BBAS3.SA, BPAC11.SA, SBSP3.SA, SAPR4.SA, CMIG3.SA, UNIP6.SA, FRAS3.SA, VALE3.SA, PETR4.SA, RENT3.SA, LREN3.SA"
    ticker_input = st.sidebar.text_area("Tickers (Separados por vírgula)", default_univ, height=100)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.50)
    w_val = st.sidebar.slider("Robust Value", 0.0, 1.0, 0.30)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Construção de Portfólio")
    top_n = st.sidebar.number_input("Número de Ativos (Top N)", 1, 20, 8)
    use_vol_target = st.sidebar.checkbox("Usar Risk Parity?", True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("4. Simulação & Monte Carlo")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 50000, 2000)
    dca_years = st.sidebar.slider("Anos de Backtest Histórico", 1, 5, 3)
    mc_years = st.sidebar.slider("Anos de Projeção (MC)", 5, 30, 10)
    
    run_btn = st.sidebar.button("🚀 Rodar Análise Completa", type="primary")

    if run_btn:
        if not tickers:
            st.error("Insira pelo menos um ticker.")
            return

        with st.status("Processando Dados...", expanded=True) as status:
            end_date = datetime.now()
            start_date_total = end_date - timedelta(days=365 * (dca_years + 3)) 
            start_date_dca = end_date - timedelta(days=365 * dca_years)

            # 1. Dados
            prices = fetch_price_data(tickers, start_date_total, end_date)
            fundamentals = fetch_fundamentals(tickers) 
            
            if prices.empty or fundamentals.empty:
                st.error("Falha fatal na obtenção de dados. Verifique os tickers.")
                status.update(label="Erro!", state="error")
                return
            
            # 2. Score Atual
            res_mom = compute_residual_momentum_enhanced(prices)
            val_score = compute_value_robust(fundamentals)
            qual_score = compute_quality_score(fundamentals)

            df_master = pd.DataFrame(index=tickers)
            df_master['Res_Mom'] = res_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            
            # --- CORREÇÃO APLICADA AQUI ---
            if 'sector' in fundamentals.columns: 
                df_master['Sector'] = fundamentals['sector']
            
            if 'currentPrice' in fundamentals.columns:
                df_master['currentPrice'] = fundamentals['currentPrice']
            # ------------------------------

            df_master.dropna(thresh=2, inplace=True)
            
            # Z-Score e Pesos
            weights_map = {'Res_Mom': w_rm, 'Value': w_val, 'Quality': w_qual}
            norm_cols = ['Res_Mom', 'Value', 'Quality']
            weights_keys = {}
            for c in norm_cols:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    weights_keys[new_col] = weights_map.get(c, 0.0)

            final_df = build_composite_score(df_master, weights_keys)
            
            # 3. Backtests
            backtest_dynamic, factor_timing_df = run_dynamic_backtest(
                prices, fundamentals, weights_map, top_n, use_vol_target, True, start_date_dca
            )
            
            status.update(label="Concluído!", state="complete", expanded=False)

        # ==============================================================================
        # TABS DE RESULTADOS
        # ==============================================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Ranking & Regime", "📈 Performance & Risco", "🔮 Projeção Monte Carlo", "📊 Factor Timing", "🔍 Dados Brutos"
        ])

        with tab1:
            # Indicadores de Regime
            st.subheader("Bússola de Mercado")
            breadth, total_tickers = calculate_market_breadth(prices, 'BOVA11.SA')
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Breadth (>MM200)", f"{breadth:.1%}", f"{total_tickers} ativos")
            
            # Fator Vencedor (6 Meses)
            if not factor_timing_df.empty:
                last_6m = factor_timing_df.iloc[-1] / factor_timing_df.iloc[-126] - 1
                winner = last_6m.idxmax()
                c2.metric("Fator Dominante (6m)", winner, f"{last_6m.max():.2%} ret")
            
            current_top = final_df.head(top_n)
            st.markdown("---")
            st.subheader("Carteira Sugerida Hoje")
            
            # Tratamento defensivo para exibição
            cols_to_show = ['Composite_Score']
            if 'Sector' in current_top.columns: cols_to_show.append('Sector')
            if 'currentPrice' in current_top.columns: cols_to_show.append('currentPrice')
            cols_to_show += list(weights_keys.keys())
            
            st.dataframe(current_top[cols_to_show].style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']))

        with tab2:
            st.subheader(f"Análise de Risco ({dca_years} Anos)")
            if not backtest_dynamic.empty:
                # Calcula métricas para Estratégia e Benchmarks
                m_strat = calculate_advanced_metrics(backtest_dynamic['Strategy'])
                m_bova = calculate_advanced_metrics(backtest_dynamic['BOVA11'])
                m_smal = calculate_advanced_metrics(backtest_dynamic['SMAL11'])
                
                # Tabela Comparativa
                metrics_df = pd.DataFrame([m_strat, m_bova, m_smal], index=['Estratégia', 'BOVA11', 'SMAL11'])
                
                # Formatação
                fmt_dict = {
                    'Retorno Total': '{:.2%}', 'CAGR': '{:.2%}', 'Volatilidade': '{:.2%}',
                    'Max Drawdown': '{:.2%}', 'Sharpe': '{:.2f}', 'Sortino': '{:.2f}',
                    'Calmar': '{:.2f}', 'Ulcer Index': '{:.4f}'
                }
                st.dataframe(metrics_df.style.format(fmt_dict).background_gradient(cmap='RdYlGn', axis=0))
                
                st.plotly_chart(px.line(backtest_dynamic, title="Curva de Retorno Acumulado", 
                                        color_discrete_map={'Strategy': '#00CC96', 'BOVA11': '#EF553B', 'SMAL11': '#636EFA'}), 
                                use_container_width=True)
                
                # Drawdown Chart
                drawdown_df = backtest_dynamic / backtest_dynamic.cummax() - 1
                st.plotly_chart(px.area(drawdown_df, title="Underwater Plot (Drawdown)", 
                                        color_discrete_map={'Strategy': '#00CC96', 'BOVA11': '#EF553B'}), 
                                use_container_width=True)

        with tab3:
            st.subheader("Simulação Monte Carlo (Probabilística)")
            st.markdown(f"Projeção de **{mc_years} anos** com aporte mensal de **R$ {dca_amount:,.2f}**.")
            st.info("Baseado na média (mu) e volatilidade (sigma) históricas da Estratégia.")

            if not backtest_dynamic.empty:
                # Parâmetros Históricos da Estratégia
                daily_rets = backtest_dynamic['Strategy'].pct_change().dropna()
                mu_hist = daily_rets.mean() * 252
                sigma_hist = daily_rets.std() * np.sqrt(252)
                
                # Inputs Ajustáveis
                c1, c2 = st.columns(2)
                adj_mu = c1.slider("Retorno Esperado Anual (%)", 0.0, 40.0, mu_hist*100) / 100
                adj_sigma = c2.slider("Volatilidade Esperada (%)", 5.0, 50.0, sigma_hist*100) / 100
                
                mc_df = run_monte_carlo(
                    initial_balance=dca_amount, # Começa com 1 aporte
                    monthly_contrib=dca_amount,
                    mu_annual=adj_mu,
                    sigma_annual=adj_sigma,
                    years=mc_years
                )
                
                final_val = mc_df.iloc[-1]
                m1, m2, m3 = st.columns(3)
                m1.metric("Cenário Pessimista (5%)", f"R$ {final_val['Pessimista (5%)']:,.2f}")
                m2.metric("Cenário Base (50%)", f"R$ {final_val['Base (50%)']:,.2f}")
                m3.metric("Cenário Otimista (95%)", f"R$ {final_val['Otimista (95%)']:,.2f}")
                
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df['Otimista (95%)'], mode='lines', line=dict(width=0), showlegend=False))
                fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df['Pessimista (5%)'], mode='lines', fill='tonexty', fillcolor='rgba(0, 204, 150, 0.2)', line=dict(width=0), name='Intervalo 90% Confiança'))
                fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df['Base (50%)'], mode='lines', line=dict(color='#00CC96', width=2), name='Cenário Mediano'))
                fig_mc.update_layout(title="Cone de Probabilidade Patrimonial", yaxis_title="Patrimônio (R$)", hovermode="x unified")
                st.plotly_chart(fig_mc, use_container_width=True)

        with tab4:
            st.subheader("Factor Timing & Correlações")
            if not factor_timing_df.empty:
                norm_factors = factor_timing_df / factor_timing_df.iloc[0]
                st.plotly_chart(px.line(norm_factors, title="Performance Relativa dos Fatores"), use_container_width=True)
                
                corr = factor_timing_df.pct_change().corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlação Histórica"), use_container_width=True)

        with tab5:
            st.dataframe(fundamentals)

if __name__ == "__main__":
    main()
