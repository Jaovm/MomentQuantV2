import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING (Busca de Dados & Liquidez)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_market_data(tickers: list, start_date: str, end_date: str):
    t_list = list(tickers)
    if 'BOVA11.SA' not in t_list:
        t_list.append('BOVA11.SA')
    
    try:
        raw_data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False,
            group_by='ticker'
        )
        
        prices = pd.DataFrame()
        volumes = pd.DataFrame()
        
        for t in t_list:
            if t in raw_data.columns.levels[0]:
                 prices[t] = raw_data[t]['Adj Close']
                 volumes[t] = raw_data[t]['Volume']
            elif t in raw_data.columns:
                 prices[t] = raw_data['Adj Close']
                 volumes[t] = raw_data['Volume']

        return prices.dropna(how='all'), volumes.dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao baixar dados de mercado: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    data = []
    clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
    
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            
            sector = info.get('sector', 'Unknown')
            if sector in ['Unknown', 'N/A'] and 'longName' in info:
                 if 'Banco' in info['longName'] or 'Financeira' in info['longName']:
                     sector = 'Financial Services'
            
            data.append({
                'ticker': t,
                'sector': sector,
                'marketCap': info.get('marketCap', np.nan),
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'dividendYield': info.get('dividendYield', np.nan),
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
    return pd.DataFrame(data).set_index('ticker')

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
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
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            resid = model.resid[:-skip]
            sigma = np.std(resid)
            scores[ticker] = (np.sum(resid) / sigma) if sigma > 0 else 0
        except:
            scores[ticker] = 0
    return pd.Series(scores, name='Residual_Momentum')

def compute_low_volatility_score(price_df: pd.DataFrame, lookback=252) -> pd.Series:
    daily_rets = price_df.pct_change().tail(lookback).dropna()
    if 'BOVA11.SA' not in daily_rets.columns: return pd.Series(dtype=float)
    
    market_rets = daily_rets['BOVA11.SA']
    stats = {}
    
    for ticker in daily_rets.columns:
        if ticker == 'BOVA11.SA': continue
        asset_rets = daily_rets[ticker]
        if len(asset_rets) < lookback * 0.8: continue
        
        vol = asset_rets.std() * np.sqrt(252)
        
        try:
            covariance = np.cov(asset_rets, market_rets)
            beta = covariance[0, 1] / covariance[1, 1]
        except:
            beta = 1.0
            
        stats[ticker] = {'vol': vol, 'beta': beta}
        
    df_stats = pd.DataFrame(stats).T
    if df_stats.empty: return pd.Series(dtype=float)

    z_vol = (df_stats['vol'] - df_stats['vol'].mean()) / df_stats['vol'].std()
    z_beta = (df_stats['beta'] - df_stats['beta'].mean()) / df_stats['beta'].std()
    
    combined_score = -1 * (z_vol + z_beta) / 2
    return combined_score.rename("Low_Volatility")

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    metrics = ['earningsGrowth', 'revenueGrowth']
    temp_df = pd.DataFrame(index=fund_df.index)
    for m in metrics:
        if m in fund_df.columns:
            s = fund_df[m].fillna(fund_df[m].median())
            temp_df[m] = (s - s.mean()) / s.std()
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=fund_df.index)
    
    if 'forwardPE' in fund_df: 
        scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    
    if 'priceToBook' in fund_df: 
        scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
        
    if 'enterpriseToEbitda' in fund_df:
        scores['EbitdaYield'] = np.where(fund_df['enterpriseToEbitda'] > 0, 1/fund_df['enterpriseToEbitda'], 0)
        
    if 'dividendYield' in fund_df:
        scores['DY'] = fund_df['dividendYield'].fillna(0)

    for col in scores.columns:
        scores[col] = (scores[col] - scores[col].mean()) / scores[col].std()
        
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: scores['DE_Inv'] = -1 * fund_df['debtToEquity']
    return scores.mean(axis=1).rename("Quality_Score")

def compute_size_score(fund_df: pd.DataFrame) -> pd.Series:
    if 'marketCap' not in fund_df.columns: return pd.Series(dtype=float)
    mcap = fund_df['marketCap'].replace(0, np.nan)
    log_mcap = np.log(mcap)
    return (-1 * log_mcap).rename("Size_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO (Advanced)
# ==============================================================================

def winsorize_series(series: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
    if series.empty: return series
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower=lower, upper=upper)

def normalize_factor(series: pd.Series, use_rank_based: bool = False) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    
    if use_rank_based:
        ranks = s.rank(pct=True)
        return (ranks - 0.5) * 6
    else:
        s_win = winsorize_series(s)
        median = s_win.median()
        mad = (s_win - median).abs().median()
        if mad == 0 or mad < 1e-6: return s_win - median
        z = (s_win - median) / (mad * 1.4826)
        return z.clip(-3, 3)

def build_composite_score(df_master: pd.DataFrame, weights: dict, use_rank_based: bool) -> pd.DataFrame:
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    
    for col, weight in weights.items():
        if col in df.columns and weight > 0:
            df[col + '_Norm'] = normalize_factor(df[col], use_rank_based)
            df['Composite_Score'] += df[col + '_Norm'].fillna(0) * weight
            
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: PORTFOLIO & OPTIMIZATION & BACKTEST
# ==============================================================================

def enforce_constraints(weights: pd.Series, sector_series: pd.Series, 
                        max_asset_weight: float, max_sector_weight: float) -> pd.Series:
    w = weights.copy()
    for _ in range(15):
        w = w.clip(upper=max_asset_weight)
        w = w / w.sum()
        
        if sector_series is not None:
            df_w = w.to_frame('weight')
            df_w['sector'] = sector_series.loc[w.index]
            sector_weights = df_w.groupby('sector')['weight'].sum()
            
            over_sectors = sector_weights[sector_weights > max_sector_weight].index
            if not over_sectors.empty:
                for sec in over_sectors:
                    scale = max_sector_weight / sector_weights[sec]
                    tickers_in_sec = df_w[df_w['sector'] == sec].index
                    w.loc[tickers_in_sec] *= scale
                
                w = w / w.sum()
            else:
                if (w <= max_asset_weight + 1e-4).all():
                    break
    return w

def optimize_portfolio(selected_tickers, cov_matrix, method='risk_parity'):
    n = len(selected_tickers)
    initial_weights = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if method == 'risk_parity':
        def risk_parity_obj(w, cov):
            port_vol = np.sqrt(w.T @ cov @ w)
            mrc = (cov @ w) / port_vol
            rc = w * mrc
            target_rc = port_vol / n
            return np.sum((rc - target_rc)**2)
            
        res = minimize(risk_parity_obj, initial_weights, args=(cov_matrix), 
                       method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(res.x, index=selected_tickers)
        
    elif method == 'min_vol':
        def min_vol_obj(w, cov):
            return np.sqrt(w.T @ cov @ w)
            
        res = minimize(min_vol_obj, initial_weights, args=(cov_matrix), 
                       method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(res.x, index=selected_tickers)

    return pd.Series(initial_weights, index=selected_tickers)

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, 
                        config: dict, sector_series: pd.Series):
    top_n = config['top_n']
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()
    
    recent_rets = prices[selected].pct_change().tail(252).dropna()
    if recent_rets.empty: return pd.Series(1/len(selected), index=selected)
    cov_matrix = recent_rets.cov() * 252

    if config['method'] == 'risk_parity':
        raw_weights = optimize_portfolio(selected, cov_matrix.values, 'risk_parity')
    elif config['method'] == 'inverse_vol':
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / vols
        raw_weights = pd.Series(inv_vols / inv_vols.sum(), index=selected)
    else:
        raw_weights = pd.Series(1.0/len(selected), index=selected)

    final_weights = enforce_constraints(
        raw_weights, 
        sector_series, 
        config['max_asset_pct'], 
        config['max_sector_pct']
    )
    
    return final_weights.sort_values(ascending=False)

def calculate_metrics(daily_returns: pd.Series, risk_free=0.10):
    if daily_returns.empty: return {}
    
    total_ret = (1 + daily_returns).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_returns)) - 1
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (ann_ret - risk_free) / vol if vol > 0 else 0
    
    cum_ret = (1 + daily_returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    neg_rets = daily_returns[daily_returns < 0]
    down_dev = neg_rets.std() * np.sqrt(252)
    sortino = (ann_ret - risk_free) / down_dev if down_dev > 0 else 0
    
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "Total Return": total_ret,
        "Annualized Return": ann_ret,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd
    }

def run_backtest_engine(
    prices: pd.DataFrame, 
    fundamentals: pd.DataFrame, 
    config: dict, 
    volume_df: pd.DataFrame = None
):
    start_date = config['start_date']
    end_date = prices.index[-1]
    
    rebal_dates = prices.loc[start_date:end_date].resample('MS').first().index.tolist()
    
    if not rebal_dates: return pd.DataFrame()

    daily_rets = []
    
    transaction_cost_pct = config.get('transaction_cost_pct', 0.002)
    prev_weights = pd.Series(dtype=float)

    for i, rebal_date in enumerate(rebal_dates):
        next_date = rebal_dates[i+1] if i < len(rebal_dates) - 1 else end_date
        
        hist_prices = prices.loc[:rebal_date]
        
        mom_window = hist_prices.tail(400)
        res_mom = compute_residual_momentum(mom_window)
        low_vol = compute_low_volatility_score(mom_window)
        
        fund_mom = compute_fundamental_momentum(fundamentals)
        val_score = compute_value_score(fundamentals)
        qual_score = compute_quality_score(fundamentals)
        size_score = compute_size_score(fundamentals)
        
        liq_mask = pd.Series(True, index=prices.columns)
        if volume_df is not None:
            avg_vol = volume_df.loc[:rebal_date].tail(63).mean() * prices.loc[:rebal_date].tail(63).mean()
            liq_mask = avg_vol > config['min_liquidity']
            
        valid_tickers = liq_mask[liq_mask].index.tolist()
        valid_tickers = [t for t in valid_tickers if t != 'BOVA11.SA']

        df_step = pd.DataFrame(index=valid_tickers)
        df_step['Res_Mom'] = res_mom
        df_step['Low_Vol'] = low_vol
        df_step['Fund_Mom'] = fund_mom
        df_step['Value'] = val_score
        df_step['Quality'] = qual_score
        df_step['Size'] = size_score
        
        if 'sector' in fundamentals.columns: 
             df_step['Sector'] = fundamentals['sector']

        df_step.dropna(thresh=2, inplace=True)
        
        w_map = config['factor_weights']
        ranked = build_composite_score(df_step, w_map, config['use_rank_based'])
        
        current_weights = construct_portfolio(
            ranked, 
            hist_prices, 
            config['portfolio_config'],
            sector_series=fundamentals['sector'] if 'sector' in fundamentals else None
        )
        
        market_period = prices.loc[rebal_date:next_date].iloc[1:]
        period_pct = market_period.pct_change().dropna(how='all')
        
        if not current_weights.empty and not period_pct.empty:
            common_tickers = list(set(current_weights.index) & set(period_pct.columns))
            strat_ret = period_pct[common_tickers].dot(current_weights[common_tickers])
            
            turnover = 0.0
            if prev_weights.empty:
                turnover = 1.0
            else:
                all_tkrs = list(set(current_weights.index) | set(prev_weights.index))
                w_curr = current_weights.reindex(all_tkrs).fillna(0)
                w_prev = prev_weights.reindex(all_tkrs).fillna(0)
                turnover = np.abs(w_curr - w_prev).sum() / 2
            
            cost = turnover * transaction_cost_pct
            strat_ret.iloc[0] -= cost
            
            daily_rets.append(strat_ret)
            prev_weights = current_weights
        else:
            daily_rets.append(pd.Series(0.0, index=period_pct.index))
            
    if daily_rets:
        full_series = pd.concat(daily_rets)
        full_series = full_series[~full_series.index.duplicated(keep='first')]
        return full_series
    return pd.Series()

# ==============================================================================
# NOVO MÓDULO: GESTÃO DE CAPITAL E APORTES (DCA)
# ==============================================================================
def calculate_dca_history(strategy_rets, initial_capital, monthly_investment):
    """
    Simula o crescimento do patrimônio considerando um aporte mensal fixo (DCA).
    """
    # Datas de início de cada mês no período do backtest
    monthly_dates = strategy_rets.resample('MS').first().index
    history = []
    current_balance = initial_capital
   
    for date in strategy_rets.index:
        # Aporte no primeiro dia útil do mês
        if date in monthly_dates:
            current_balance += monthly_investment
           
        # Aplica retorno diário
        current_balance *= (1 + strategy_rets.loc[date])
        history.append({'Date': date, 'Equity': current_balance})
       
    return pd.DataFrame(history).set_index('Date')

# ==============================================================================
# APP PRINCIPAL (STREAMLIT UI)
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab Pro v2.0")
    st.markdown("""
    **Institutional-Grade Screener & Backtester for B3.**
    Engine atualizada com Winsorization, Risk Parity Otimizado, Filtros de Liquidez e Custos de Transação.
    """)

    with st.sidebar:
        st.header("⚙️ Configuração do Universo")
        default_univ = "ITUB3.SA, VALE3.SA, PETR4.SA, WEGE3.SA, PRIO3.SA, BBAS3.SA, RENT3.SA, B3SA3.SA, SUZB3.SA, GGBR4.SA, JBSS3.SA, RAIL3.SA, VIVT3.SA, CPLE6.SA, PSSA3.SA, TOTS3.SA, EQTL3.SA, LREN3.SA, RADL3.SA, CMIG4.SA"
        ticker_input = st.text_area("Tickers (CSV)", default_univ, height=80)
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        st.divider()
        st.header("1. Definição de Alpha (Pesos)")
        w_rm = st.slider("Residual Momentum", 0.0, 1.0, 0.30)
        w_lv = st.slider("Low Volatility / Beta", 0.0, 1.0, 0.20)
        w_fm = st.slider("Fundamental Momentum", 0.0, 1.0, 0.20)
        w_val = st.slider("Value (Enhanced)", 0.0, 1.0, 0.20)
        w_qual = st.slider("Quality", 0.0, 1.0, 0.10)
        w_size = st.slider("Size (Small Cap Bias)", 0.0, 1.0, 0.00)
        
        factor_weights = {
            'Res_Mom': w_rm, 'Low_Vol': w_lv, 'Fund_Mom': w_fm, 
            'Value': w_val, 'Quality': w_qual, 'Size': w_size
        }
        
        st.divider()
        st.header("2. Normalização e Filtros")
        use_rank_based = st.checkbox("Usar Rank-Based Scoring?", value=False, help="Se ativo, ignora Z-Score e usa percentis transformados.")
        min_liquidity = st.number_input("Liquidez Mínima Diária (R$)", value=3000000, step=500000, format="%d")
        
        st.divider()
        st.header("3. Construção de Portfólio")
        top_n = st.number_input("Top N Ativos", 5, 30, 10)
        weight_method = st.selectbox("Método de Pesos", ["Inverse Volatility", "Risk Parity (Opt)", "Equal Weight"])
        max_asset_cap = st.slider("Cap Máximo por Ativo (%)", 0.05, 0.50, 0.15)
        max_sector_cap = st.slider("Cap Máximo por Setor (%)", 0.10, 1.00, 0.35)
        
        st.divider()
        st.header("4. Parâmetros de Backtest")
        trans_cost = st.slider("Custo de Transação (%)", 0.0, 1.0, 0.20) / 100
        years_backtest = st.slider("Anos de Backtest", 1, 5, 3)
        add_benchmarks = st.checkbox("Benchmarks Extras (^BVSP, IDIV)", False)
        
        st.divider()
        st.header("💰 Gestão de Capital")
        capital_inicial = st.number_input("Capital Inicial (R$)", value=100000, step=10000)
        aporte_mensal = st.number_input("Aporte Mensal (R$)", value=2000, step=500)
        
        run_btn = st.button("🚀 Executar Engine Quant", type="primary")

    if run_btn:
        if not tickers:
            st.error("Insira tickers válidos.")
            return

        with st.status("Processando Pipeline Quantitativo...", expanded=True) as status:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * (years_backtest + 1))
            
            st.write("📥 Baixando Dados de Mercado (Preço e Volume)...")
            prices, volumes = fetch_market_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            st.write("📊 Baixando Fundamentos...")
            fundamentals = fetch_fundamentals(tickers)
            
            if prices.empty or fundamentals.empty:
                st.error("Falha na obtenção de dados.")
                status.update(label="Erro", state="error")
                return

            st.write("🧮 Calculando Fatores Atuais...")
            current_mom_prices = prices.tail(400)
            
            res_mom = compute_residual_momentum(current_mom_prices)
            low_vol = compute_low_volatility_score(current_mom_prices)
            fund_mom = compute_fundamental_momentum(fundamentals)
            val_score = compute_value_score(fundamentals)
            qual_score = compute_quality_score(fundamentals)
            size_score = compute_size_score(fundamentals)
            
            df_current = pd.DataFrame(index=tickers)
            df_current['Res_Mom'] = res_mom
            df_current['Low_Vol'] = low_vol
            df_current['Fund_Mom'] = fund_mom
            df_current['Value'] = val_score
            df_current['Quality'] = qual_score
            df_current['Size'] = size_score
            if 'sector' in fundamentals.columns: df_current['Sector'] = fundamentals['sector']
            
            avg_liq = volumes.tail(63).mean() * prices.tail(63).mean()
            liquid_tickers = avg_liq[avg_liq >= min_liquidity].index            
            df_current = df_current.loc[df_current.index.intersection(liquid_tickers)]
            
            ranked_current = build_composite_score(df_current, factor_weights, use_rank_based)
            
            port_config = {
                'top_n': top_n,
                'method': 'risk_parity' if 'Risk Parity' in weight_method else ('inverse_vol' if 'Inverse' in weight_method else 'equal'),
                'max_asset_pct': max_asset_cap,
                'max_sector_pct': max_sector_cap
            }
            
            current_weights = construct_portfolio(
                ranked_current, 
                current_mom_prices,
                port_config,
                sector_series=fundamentals['sector'] if 'sector' in fundamentals else None
            )
            
            st.write("⏳ Rodando Backtest Walk-Forward com Custos...")
            backtest_config = {
                'start_date': end_date - timedelta(days=365 * years_backtest),
                'factor_weights': factor_weights,
                'use_rank_based': use_rank_based,
                'min_liquidity': min_liquidity,
                'portfolio_config': port_config,
                'transaction_cost_pct': trans_cost
            }
            
            strategy_rets = run_backtest_engine(prices, fundamentals, backtest_config, volumes)
            
            bench_rets = prices['BOVA11.SA'].pct_change().loc[strategy_rets.index]
            
            status.update(label="Cálculos Finalizados!", state="complete", expanded=False)

        # ======================================================================
        # VISUALIZAÇÃO DOS RESULTADOS
        # ======================================================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Ranking & Alocação",
            "📈 Performance & Risco",
            "🔍 Fatores & Correlação",
            "📋 Plano de Execução",
            "💾 Exportar Dados"
        ])
        
        # Preparar dataframe de execução (usado nas abas 4 e 5)
        exec_df = pd.DataFrame()
        if not current_weights.empty:
            last_prices = prices.iloc[-1]
            exec_df = current_weights.to_frame('Peso (%)')
            exec_df['Financeiro (R$)'] = exec_df['Peso (%)'] * capital_inicial
            available_prices = last_prices.reindex(exec_df.index)
            exec_df['Preço Atual (R$)'] = available_prices
            exec_df['Cotas (Estimadas)'] = (exec_df['Financeiro (R$)'] / exec_df['Preço Atual (R$)']).fillna(0).astype(int)

        with tab1:
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.subheader("Top Picks (Composite Score)")
                
                disp_cols = ['Composite_Score', 'Sector'] + [c for c in ['Res_Mom_Norm', 'Low_Vol_Norm', 'Value_Norm'] if c in ranked_current.columns]
                st.dataframe(
                    ranked_current[disp_cols].head(top_n).style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']),
                    use_container_width=True
                )
                
                if not current_weights.empty:
                    st.subheader("Exposição Setorial")
                    df_w = current_weights.to_frame('Weight')
                    df_w['Sector'] = fundamentals.loc[df_w.index, 'sector']
                    fig_tree = px.treemap(df_w.reset_index(), path=['Sector', 'index'], values='Weight', 
                                          title="Alocação por Setor e Ativo")
                    st.plotly_chart(fig_tree, use_container_width=True)

            with col2:
                st.subheader("Pesos do Portfólio Otimizado")
                if not current_weights.empty:
                    w_df_disp = current_weights.to_frame("Peso")
                    w_df_disp["Peso"] = w_df_disp["Peso"].map("{:.2%}".format)
                    st.table(w_df_disp)
                    
                    avg_score = ranked_current['Composite_Score'].head(top_n).mean()
                    if avg_score > 1.5:
                        st.success("✅ Regime Favorável: High Factor Scores")
                    else:
                        st.info("ℹ️ Regime Neutro/Fraco para Fatores")

        with tab2:
            st.subheader(f"Performance Histórica ({years_backtest} Anos)")
            if not strategy_rets.empty:
                cum_strat = (1 + strategy_rets).cumprod()
                cum_bench = (1 + bench_rets).cumprod()
                
                df_perf = pd.DataFrame({'Strategy': cum_strat, 'BOVA11': cum_bench})
                
                m_strat = calculate_metrics(strategy_rets)
                m_bench = calculate_metrics(bench_rets)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno Total", f"{m_strat['Total Return']:.2%}", delta=f"{m_strat['Total Return']-m_bench['Total Return']:.2%}")
                c2.metric("Sharpe Ratio", f"{m_strat['Sharpe']:.2f}", delta=f"{m_strat['Sharpe']-m_bench['Sharpe']:.2f}")
                c3.metric("Volatilidade", f"{m_strat['Volatility']:.2%}")
                c4.metric("Max Drawdown", f"{m_strat['Max Drawdown']:.2%}")
                
                st.plotly_chart(px.line(df_perf, title="Evolução Patrimonial (Base 1.0)", color_discrete_map={'Strategy': '#00CC96', 'BOVA11': '#EF553B'}), use_container_width=True)
                
                with st.expander("Métricas Avançadas (Tabela)"):
                    metrics_df = pd.DataFrame([m_strat, m_bench], index=['Strategy', 'Benchmark']).T
                    st.table(metrics_df.style.format("{:.2f}"))

        with tab3:
            st.subheader("Análise de Fatores")
            if not ranked_current.empty:
                factor_cols = [c for c in ranked_current.columns if '_Norm' in c]
                if factor_cols:
                    corr = ranked_current[factor_cols].corr()
                    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlação entre Fatores (Universo Atual)"), use_container_width=True)
            
            st.subheader("Dados Fundamentais Brutos")
            st.dataframe(fundamentals)

        with tab4:
            st.subheader("🎯 Ordem de Execução Atual")
            st.markdown(f"Baseado em um capital de **R$ {capital_inicial:,.2f}**")
           
            if not current_weights.empty and not exec_df.empty:
                st.dataframe(
                    exec_df.style.format({
                        'Peso (%)': '{:.2%}',
                        'Financeiro (R$)': 'R$ {:,.2f}',
                        'Preço Atual (R$)': 'R$ {:,.2f}'
                    }),
                    use_container_width=True
                )
           
            st.divider()
            st.subheader("⏳ Histórico de Acúmulo (DCA)")
           
            if not strategy_rets.empty:
                dca_df = calculate_dca_history(strategy_rets, capital_inicial, aporte_mensal)
               
                col_m1, col_m2 = st.columns(2)
                final_equity = dca_df['Equity'].iloc[-1]
                
                # Cálculo do número de meses baseado no resampling mensal usado na função DCA
                num_months = len(strategy_rets.resample('MS'))
                total_invested = capital_inicial + (aporte_mensal * num_months)
               
                col_m1.metric("Patrimônio Final Est.", f"R$ {final_equity:,.2f}")
                col_m2.metric("Total Investido", f"R$ {total_invested:,.2f}",
                              delta=f"Lucro: R$ {final_equity - total_invested:,.2f}")
               
                st.plotly_chart(
                    px.area(dca_df, y='Equity', title="Crescimento do Patrimônio (Capital + Aportes + Retornos)",
                            labels={'Equity': 'Saldo (R$)', 'Date': 'Data'},
                            color_discrete_sequence=['#00CC96']),
                    use_container_width=True
                )
            else:
                st.info("Backtest vazio – não há dados para simulação DCA.")

        with tab5:
            st.subheader("Exportar Resultados")
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                csv_rank = ranked_current.to_csv().encode('utf-8')
                st.download_button("📥 Baixar Ranking Completo (CSV)", data=csv_rank, file_name="factor_ranking.csv", mime="text/csv")
            
            with col_d2:
                if not strategy_rets.empty:
                    cum_strat = (1 + strategy_rets).cumprod().to_frame('Strategy')
                    cum_bench = (1 + bench_rets).cumprod().to_frame('BOVA11')
                    csv_perf = pd.concat([cum_strat, cum_bench], axis=1).to_csv().encode('utf-8')
                    st.download_button("📥 Baixar Série de Retornos (CSV)", data=csv_perf, file_name="strategy_performance.csv", mime="text/csv")
            
            with col_d3:
                if not exec_df.empty:
                    csv_exec = exec_df.to_csv().encode('utf-8')
                    st.download_button("📥 Exportar Plano de Compras (CSV)", data=csv_exec, file_name="plano_execucao.csv", mime="text/csv")

if __name__ == "__main__":
    main()
