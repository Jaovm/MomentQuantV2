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
    t_list = list(dict.fromkeys(tickers)) # Remove duplicatas mantendo ordem
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
            if sector in ['Unknown', 'N/A', None] and 'longName' in info:
                 if any(word in info['longName'] for word in ['Banco', 'Financeira', 'Investimentos']):
                     sector = 'Financial Services'
            
            data.append({
                'ticker': t,
                'sector': sector if sector else 'Unknown',
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
# MÓDULO 2: CÁLCULO DE FATORES
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
            temp_df[m] = (s - s.mean()) / (s.std() + 1e-6)
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=fund_df.index)
    if 'forwardPE' in fund_df: scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    if 'priceToBook' in fund_df: scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
    if 'enterpriseToEbitda' in fund_df: scores['EbitdaYield'] = np.where(fund_df['enterpriseToEbitda'] > 0, 1/fund_df['enterpriseToEbitda'], 0)
    if 'dividendYield' in fund_df: scores['DY'] = fund_df['dividendYield'].fillna(0)

    for col in scores.columns:
        std = scores[col].std()
        scores[col] = (scores[col] - scores[col].mean()) / (std if std > 0 else 1)
        
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
    # Normalizado para Z-Score para consistência
    z_size = -1 * (log_mcap - log_mcap.mean()) / log_mcap.std()
    return z_size.rename("Size_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def winsorize_series(series: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
    if series.empty: return series
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower=lower, upper=upper)

def normalize_factor(series: pd.Series, use_rank_based: bool = False) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty: return s
    if use_rank_based:
        ranks = s.rank(pct=True)
        return (ranks - 0.5) * 6
    else:
        s_win = winsorize_series(s)
        median = s_win.median()
        mad = (s_win - median).abs().median()
        if mad < 1e-6: return s_win - median
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
# MÓDULO 4: PORTFOLIO & BACKTEST
# ==============================================================================

def enforce_constraints(weights: pd.Series, sector_series: pd.Series, 
                        max_asset_weight: float, max_sector_weight: float) -> pd.Series:
    w = weights.copy()
    for _ in range(15):
        w = w.clip(upper=max_asset_weight)
        w = w / w.sum()
        if sector_series is not None:
            # Proteção contra tickers faltantes no Series de setores
            valid_sectors = sector_series.reindex(w.index).fillna('Unknown')
            df_w = w.to_frame('weight')
            df_w['sector'] = valid_sectors
            sector_weights = df_w.groupby('sector')['weight'].sum()
            over_sectors = sector_weights[sector_weights > max_sector_weight].index
            if not over_sectors.empty:
                for sec in over_sectors:
                    scale = max_sector_weight / sector_weights[sec]
                    tickers_in_sec = df_w[df_w['sector'] == sec].index
                    w.loc[tickers_in_sec] *= scale
                w = w / w.sum()
            else:
                if (w <= max_asset_weight + 1e-4).all(): break
    return w

def optimize_portfolio(selected_tickers, cov_matrix, method='risk_parity'):
    n = len(selected_tickers)
    if n == 0: return pd.Series()
    initial_weights = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if method == 'risk_parity':
        def risk_parity_obj(w, cov):
            port_vol = np.sqrt(w.T @ cov @ w)
            if port_vol <= 0: return 0
            mrc = (cov @ w) / port_vol
            rc = w * mrc
            target_rc = port_vol / n
            return np.sum((rc - target_rc)**2)
        res = minimize(risk_parity_obj, initial_weights, args=(cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(res.x, index=selected_tickers)
    elif method == 'min_vol':
        def min_vol_obj(w, cov): return np.sqrt(w.T @ cov @ w)
        res = minimize(min_vol_obj, initial_weights, args=(cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(res.x, index=selected_tickers)
    return pd.Series(initial_weights, index=selected_tickers)

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, config: dict, sector_series: pd.Series):
    top_n = config['top_n']
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()
    
    recent_rets = prices[selected].pct_change().tail(252).dropna(how='all').fillna(0)
    if recent_rets.empty: return pd.Series(1/len(selected), index=selected)
    cov_matrix = recent_rets.cov() * 252

    if config['method'] == 'risk_parity':
        raw_weights = optimize_portfolio(selected, cov_matrix.values, 'risk_parity')
    elif config['method'] == 'inverse_vol':
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / np.where(vols > 0, vols, vols.mean() if vols.mean() > 0 else 1)
        raw_weights = pd.Series(inv_vols / inv_vols.sum(), index=selected)
    else:
        raw_weights = pd.Series(1.0/len(selected), index=selected)

    return enforce_constraints(raw_weights, sector_series, config['max_asset_pct'], config['max_sector_pct'])

def calculate_metrics(daily_returns: pd.Series, risk_free=0.10):
    if daily_returns.empty: return {k: 0.0 for k in ["Total Return", "Annualized Return", "Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown"]}
    total_ret = (1 + daily_returns).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_returns)) - 1
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (ann_ret - risk_free) / vol if vol > 0 else 0
    cum_ret = (1 + daily_returns).cumprod()
    peak = cum_ret.cummax()
    max_dd = ((cum_ret - peak) / peak).min()
    return {"Total Return": total_ret, "Annualized Return": ann_ret, "Volatility": vol, "Sharpe": sharpe, "Max Drawdown": max_dd}

def run_backtest_engine(prices: pd.DataFrame, fundamentals: pd.DataFrame, config: dict, volume_df: pd.DataFrame = None):
    start_date = config['start_date']
    end_date = prices.index[-1]
    rebal_dates = prices.loc[start_date:end_date].resample('MS').first().index.tolist()
    if not rebal_dates: return pd.Series()

    daily_rets = []
    prev_weights = pd.Series(dtype=float)

    for i, rebal_date in enumerate(rebal_dates):
        next_date = rebal_dates[i+1] if i < len(rebal_dates) - 1 else end_date
        hist_prices = prices.loc[:rebal_date]
        if len(hist_prices) < 60: continue
        
        # Filtro Liquidez
        valid_tickers = prices.columns.tolist()
        if volume_df is not None:
            avg_vol_fin = (volume_df.loc[:rebal_date].tail(63) * prices.loc[:rebal_date].tail(63)).mean()
            valid_tickers = avg_vol_fin[avg_vol_fin >= config['min_liquidity']].index.tolist()
        
        valid_tickers = [t for t in valid_tickers if t != 'BOVA11.SA']
        mom_window = hist_prices.tail(400)
        
        df_step = pd.DataFrame(index=valid_tickers)
        df_step['Res_Mom'] = compute_residual_momentum(mom_window).reindex(valid_tickers)
        df_step['Low_Vol'] = compute_low_volatility_score(mom_window).reindex(valid_tickers)
        df_step['Fund_Mom'] = compute_fundamental_momentum(fundamentals).reindex(valid_tickers)
        df_step['Value'] = compute_value_score(fundamentals).reindex(valid_tickers)
        df_step.dropna(thresh=2, inplace=True)
        
        ranked = build_composite_score(df_step, config['factor_weights'], config['use_rank_based'])
        current_weights = construct_portfolio(ranked, hist_prices, config['portfolio_config'], fundamentals['sector'])
        
        period_pct = prices.loc[rebal_date:next_date].iloc[1:].pct_change().dropna(how='all')
        if not current_weights.empty and not period_pct.empty:
            common = list(set(current_weights.index) & set(period_pct.columns))
            strat_ret = period_pct[common].dot(current_weights[common])
            
            # Turnover e Custos
            all_tkrs = list(set(current_weights.index) | set(prev_weights.index))
            turnover = np.abs(current_weights.reindex(all_tkrs).fillna(0) - prev_weights.reindex(all_tkrs).fillna(0)).sum() / 2
            strat_ret.iloc[0] -= (turnover * config['transaction_cost_pct'])
            
            daily_rets.append(strat_ret)
            prev_weights = current_weights
            
    return pd.concat(daily_rets) if daily_rets else pd.Series()

# ==============================================================================
# GESTÃO DE CAPITAL
# ==============================================================================

def calculate_dca_history(strategy_rets, initial_capital, monthly_investment):
    if strategy_rets.empty: return pd.DataFrame()
    # Identifica inícios de meses no índice real do backtest
    monthly_dates = strategy_rets.index[strategy_rets.index.to_series().dt.is_month_start]
    if monthly_dates.empty: # Fallback se não houver 'dia 1' exato (feriados)
        monthly_dates = strategy_rets.resample('MS').first().index

    history = []
    current_balance = initial_capital
    for date in strategy_rets.index:
        if date in monthly_dates:
            current_balance += monthly_investment
        current_balance *= (1 + strategy_rets.loc[date])
        history.append({'Date': date, 'Equity': current_balance})
    return pd.DataFrame(history).set_index('Date')

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab Pro v2.0")

    with st.sidebar:
        st.header("⚙️ Configuração")
        default_univ = "ITUB3.SA, VALE3.SA, PETR4.SA, WEGE3.SA, PRIO3.SA, BBAS3.SA, RENT3.SA, B3SA3.SA, SUZB3.SA, GGBR4.SA, JBSS3.SA, RAIL3.SA, VIVT3.SA, CPLE6.SA, PSSA3.SA, TOTS3.SA, EQTL3.SA, LREN3.SA, RADL3.SA, CMIG4.SA"
        ticker_input = st.text_area("Tickers (CSV)", default_univ, height=80)
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        st.divider()
        factor_weights = {
            'Res_Mom': st.slider("Residual Momentum", 0.0, 1.0, 0.3),
            'Low_Vol': st.slider("Low Volatility", 0.0, 1.0, 0.2),
            'Fund_Mom': st.slider("Fundamental Momentum", 0.0, 1.0, 0.2),
            'Value': st.slider("Value", 0.0, 1.0, 0.3),
            'Quality': 0.0, 'Size': 0.0
        }
        
        use_rank_based = st.checkbox("Rank-Based Scoring", value=False)
        min_liquidity = st.number_input("Liquidez Mín. Diária (R$)", value=3000000)
        top_n = st.number_input("Top N Ativos", 5, 30, 10)
        weight_method = st.selectbox("Método de Pesos", ["Risk Parity (Opt)", "Inverse Volatility", "Equal Weight"])
        
        st.divider()
        capital_inicial = st.number_input("Capital Inicial (R$)", value=100000)
        aporte_mensal = st.number_input("Aporte Mensal (R$)", value=2000)
        years_backtest = st.slider("Anos de Backtest", 1, 5, 3)
        trans_cost = st.slider("Custo Transação (%)", 0.0, 1.0, 0.2) / 100
        
        run_btn = st.button("🚀 Executar Engine", type="primary", use_container_width=True)

    if run_btn:
        with st.status("Processando Pipeline...", expanded=True) as status:
            end_date_dt = datetime.now()
            start_date_dt = end_date_dt - timedelta(days=365 * (years_backtest + 1))
            
            prices, volumes = fetch_market_data(tickers, start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d'))
            fundamentals = fetch_fundamentals(tickers)
            
            if prices.empty or fundamentals.empty:
                st.error("Dados insuficientes.")
                return

            # Pipeline Atual
            current_mom = prices.tail(400)
            df_current = pd.DataFrame(index=fundamentals.index)
            df_current['Res_Mom'] = compute_residual_momentum(current_mom)
            df_current['Low_Vol'] = compute_low_volatility_score(current_mom)
            df_current['Fund_Mom'] = compute_fundamental_momentum(fundamentals)
            df_current['Value'] = compute_value_score(fundamentals)
            
            ranked_current = build_composite_score(df_current, factor_weights, use_rank_based)
            port_config = {
                'top_n': top_n, 'method': 'risk_parity' if 'Risk' in weight_method else ('inverse_vol' if 'Inverse' in weight_method else 'equal'),
                'max_asset_pct': 0.20, 'max_sector_pct': 0.40
            }
            
            current_weights = construct_portfolio(ranked_current, current_mom, port_config, fundamentals['sector'])
            
            # Backtest
            backtest_config = {
                'start_date': end_date_dt - timedelta(days=365 * years_backtest),
                'factor_weights': factor_weights, 'use_rank_based': use_rank_based,
                'min_liquidity': min_liquidity, 'portfolio_config': port_config, 'transaction_cost_pct': trans_cost
            }
            strategy_rets = run_backtest_engine(prices, fundamentals, backtest_config, volumes)
            bench_rets = prices['BOVA11.SA'].pct_change().reindex(strategy_rets.index).fillna(0)
            
            status.update(label="Concluído!", state="complete")

        # Visualização
        tab1, tab2, tab3 = st.tabs(["🏆 Alocação", "📈 Performance", "🎯 Execução"])
        
        with tab1:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.subheader("Ranking de Ativos")
                st.dataframe(ranked_current[['Composite_Score']].head(top_n).style.background_gradient(cmap='RdYlGn'), use_container_width=True)
            with c2:
                st.subheader("Pesos")
                if not current_weights.empty:
                    fig = px.pie(values=current_weights, names=current_weights.index, hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if not strategy_rets.empty:
                m_s = calculate_metrics(strategy_rets)
                m_b = calculate_metrics(bench_rets)
                
                cols = st.columns(4)
                cols[0].metric("Retorno Anual", f"{m_s['Annualized Return']:.2%}", f"{m_s['Annualized Return']-m_b['Annualized Return']:.2%}")
                cols[1].metric("Sharpe", f"{m_s['Sharpe']:.2f}")
                cols[2].metric("Max DD", f"{m_s['Max Drawdown']:.2%}")
                cols[3].metric("Volatilidade", f"{m_s['Volatility']:.2%}")
                
                res_df = pd.DataFrame({'Estratégia': (1+strategy_rets).cumprod(), 'BOVA11': (1+bench_rets).cumprod()})
                st.plotly_chart(px.line(res_df, title="Retorno Acumulado (Lump Sum)"), use_container_width=True)

        with tab3:
            if not current_weights.empty:
                last_prices = prices.iloc[-1]
                exec_df = current_weights.to_frame('Peso (%)')
                exec_df['Financeiro (R$)'] = exec_df['Peso (%)'] * capital_inicial
                
                # Tratamento de Preços Faltantes
                available_prices = last_prices.reindex(exec_df.index)
                if available_prices.isna().any():
                    st.warning(f"Atenção: Ativos sem preço recente: {list(available_prices[available_prices.isna()].index)}")
                
                exec_df['Preço (R$)'] = available_prices
                exec_df['Cotas'] = (exec_df['Financeiro (R$)'] / exec_df['Preço (R$)']).fillna(0).astype(int)
                
                total_alloc = exec_df['Financeiro (R$)'].sum()
                st.metric("Total Alocado", f"R$ {total_alloc:,.2f}", delta=f"Cash: R$ {capital_inicial - total_alloc:,.2f}")
                st.table(exec_df.style.format({'Peso (%)': '{:.2%}', 'Financeiro (R$)': '{:,.2f}', 'Preço (R$)': '{:,.2f}'}))
                
                # Gráfico DCA
                st.divider()
                st.subheader("Simulação DCA (Acúmulo)")
                dca_history = calculate_dca_history(strategy_rets, capital_inicial, aporte_mensal)
                if not dca_history.empty:
                    # Correção Crítica da Contagem de Meses
                    monthly_dates = strategy_rets.resample('MS').first().index
                    months_count = len(monthly_dates)
                    total_invested = capital_inicial + (aporte_mensal * months_count)
                    final_value = dca_history['Equity'].iloc[-1]
                    
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Patrimônio Final", f"R$ {final_value:,.2f}")
                    mc2.metric("Total Investido", f"R$ {total_invested:,.2f}", delta=f"Lucro: R$ {final_value - total_invested:,.2f}")
                    st.plotly_chart(px.area(dca_history, title="Evolução com Aportes Mensais"), use_container_width=True)

if __name__ == "__main__":
    main()
