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
    page_title="Quant Factor Lab Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados, incluindo DIVO11.SA como benchmark."""
    t_list = list(set(tickers))  # Remove duplicatas
    if 'DIVO11.SA' not in t_list:
        t_list.append('DIVO11.SA')
    
    try:
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=True  # Já usa preços ajustados
        )['Close' if len(t_list) == 1 else 'Adj Close']
        
        if data.empty:
            return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """Busca snapshot fundamental atual."""
    data = []
    clean_tickers = [t for t in tickers if t != 'DIVO11.SA']
    
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            info = yf.Ticker(t).info
            sector = info.get('sector', 'Unknown')
            if sector in ['Unknown', 'N/A'] and 'longName' in info:
                if any(word in info['longName'] for word in ['Banco', 'Financeira', 'Seguros']):
                    sector = 'Financial Services'
            
            data.append({
                'ticker': t,
                'sector': sector,
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
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
    """Residual Momentum (Alpha) vs DIVO11.SA."""
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'DIVO11.SA' not in rets.columns:
        return pd.Series(dtype=float)
        
    market = rets['DIVO11.SA']
    scores = {}
    window = lookback + skip
    
    for ticker in rets.columns:
        if ticker == 'DIVO11.SA':
            continue
        
        y = rets[ticker].tail(window)
        x = market.tail(window)
        
        if len(y) < window:
            continue
            
        try:
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            resid = model.resid[:-skip]
            
            if np.std(resid) == 0 or len(resid) < 2:
                scores[ticker] = 0
            else:
                scores[ticker] = np.sum(resid) / np.std(resid)
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

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
    if 'forwardPE' in fund_df.columns:
        scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    if 'priceToBook' in fund_df.columns:
        scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df.columns:
        scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df.columns:
        scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df.columns:
        scores['DE_Inv'] = -fund_df['debtToEquity']
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6:
        return series - median
    z = (series - median) / (mad * 1.4826)
    return z.clip(-3, 3)

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: PORTFOLIO & BACKTEST
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    selected = ranked_df.head(top_n).index.tolist()
    if not selected:
        return pd.Series()

    if vol_target is not None:
        recent_rets = prices[selected].pct_change().tail(63)
        vols = recent_rets.std() * np.sqrt(252)
        vols.replace(0, 1e-6, inplace=True)
        raw_weights = 1 / vols
        weights = raw_weights / raw_weights.sum()
    else:
        weights = pd.Series(1.0 / len(selected), index=selected)
        
    return weights.sort_values(ascending=False)

def run_dynamic_backtest(all_prices, all_fundamentals, weights_config, top_n, use_vol_target, use_sector_neutrality, start_date_backtest):
    end_date = all_prices.index[-1]
    subset_prices = all_prices.loc[start_date_backtest - timedelta(days=400):end_date]
    rebalance_dates = subset_prices.loc[start_date_backtest:end_date].resample('MS').first().index.tolist()
    
    if len(rebalance_dates) < 2:
        return pd.DataFrame()

    strategy_daily_rets = []
    benchmark_daily_rets = []

    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates)-1 else end_date + timedelta(days=1)
        
        prices_historical = subset_prices.loc[:rebal_date]
        mom_window = prices_historical.tail(400)
        risk_window = prices_historical.tail(90)
        
        res_mom = compute_residual_momentum(mom_window)
        
        # Point-in-time fundamentals
        fundamentals_at_rebal = all_fundamentals.loc[rebal_date].unstack(level='Metric') if all_fundamentals.index.names == ['Date', 'Ticker'] else all_fundamentals.loc[rebal_date]
        
        fund_mom = compute_fundamental_momentum(fundamentals_at_rebal)
        val_score = compute_value_score(fundamentals_at_rebal)
        qual_score = compute_quality_score(fundamentals_at_rebal)
        
        df_period = pd.DataFrame(index=[c for c in all_prices.columns if c != 'DIVO11.SA'])
        df_period['Res_Mom'] = res_mom
        df_period['Fund_Mom'] = fund_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        if 'sector' in fundamentals_at_rebal.columns:
            df_period['Sector'] = fundamentals_at_rebal['sector']
        
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
        
        period_prices = subset_prices.loc[rebal_date:next_date].iloc[1:]
        period_pct = period_prices.pct_change().dropna()
        if period_pct.empty or current_weights.empty:
            continue
            
        strategy_rets = (period_pct[current_weights.index] * current_weights).sum(axis=1)
        strategy_daily_rets.append(strategy_rets)
        benchmark_daily_rets.append(period_pct['DIVO11.SA'])

    if not strategy_daily_rets:
        return pd.DataFrame()

    strategy_rets_series = pd.concat(strategy_daily_rets)
    benchmark_rets_series = pd.concat(benchmark_daily_rets)
    
    df_backtest = pd.DataFrame({
        'Strategy': (1 + strategy_rets_series).cumprod(),
        'DIVO11.SA': (1 + benchmark_rets_series).cumprod()
    })
    
    return df_backtest

def run_dca_backtest(all_prices, all_fundamentals, weights_config, top_n, dca_amount, use_vol_target, use_sector_neutrality, start_date, end_date):
    subset_prices = all_prices.loc[start_date - timedelta(days=400):end_date]
    rebalance_dates = subset_prices.loc[start_date:end_date].resample('MS').first().index.tolist()
    
    if len(rebalance_dates) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    holdings = {}  # {ticker: quantidade}
    cash = 0.0
    transactions = []
    portfolio_values = []
    benchmark_units = 0.0  # Unidades do BOVA11 compradas com aportes

    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates)-1 else end_date + timedelta(days=1)
        
        # === Seleção e pesos ===
        prices_hist = subset_prices.loc[:rebal_date]
        mom_window = prices_hist.tail(400)
        risk_window = prices_hist.tail(90)
        
        res_mom = compute_residual_momentum(mom_window)
        fundamentals_at_rebal = all_fundamentals.loc[rebal_date].unstack(level='Metric') if all_fundamentals.index.names == ['Date', 'Ticker'] else all_fundamentals.loc[rebal_date]
        
        fund_mom = compute_fundamental_momentum(fundamentals_at_rebal)
        val_score = compute_value_score(fundamentals_at_rebal)
        qual_score = compute_quality_score(fundamentals_at_rebal)
        
        df_period = pd.DataFrame(index=[c for c in all_prices.columns if c != 'DIVO11.SA'])
        df_period['Res_Mom'] = res_mom
        df_period['Fund_Mom'] = fund_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        if 'sector' in fundamentals_at_rebal.columns:
            df_period['Sector'] = fundamentals_at_rebal['sector']
        
        df_period.dropna(thresh=2, inplace=True)
        
        norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
        w_keys = {}
        if use_sector_neutrality and 'Sector' in df_period.columns and df_period['Sector'].nunique() > 1:
            for c in norm_cols:
                if c in df_period.columns:
                    df_period[f"{c}_Z"] = df_period.groupby('Sector')[c].transform(
                        lambda x: robust_zscore(x) if len(x) > 1 else x - x.median())
                    w_keys[f"{c}_Z"] = weights_config.get(c, 0.0)
        else:
            for c in norm_cols:
                if c in df_period.columns:
                    df_period[f"{c}_Z"] = robust_zscore(df_period[c])
                    w_keys[f"{c}_Z"] = weights_config.get(c, 0.0)
        
        ranked = build_composite_score(df_period, w_keys)
        target_weights = construct_portfolio(ranked, risk_window, top_n, 0.15 if use_vol_target else None)
        
        # === Valor atual do portfólio ===
        current_prices = subset_prices.loc[rebal_date]
        portfolio_value = sum(holdings.get(t, 0) * current_prices.get(t, np.nan) for t in holdings.keys() if t in current_prices.index)
        total_value = portfolio_value + cash
        
        # === Aporte mensal ===
        cash += dca_amount
        total_value += dca_amount
        
        # === Aporte no benchmark ===
        bench_price = current_prices.get('DIVO11.SA', np.nan)
        if not np.isnan(bench_price) and bench_price > 0:
            benchmark_units += dca_amount / bench_price
        
        # === Rebalanceamento ===
        target_allocation = total_value * target_weights
        
        for t in target_weights.index:
            current_qty = holdings.get(t, 0)
            current_val = current_qty * current_prices.get(t, 0)
            target_val = target_allocation.get(t, 0)
            delta_val = target_val - current_val
            
            price = current_prices.get(t, 0)
            if price > 0:
                qty_delta = np.floor(delta_val / price + 0.5)  # Arredondamento padrão
                holdings[t] = current_qty + qty_delta
                cash -= qty_delta * price
                
                if qty_delta != 0:
                    transactions.append({
                        'Date': rebal_date,
                        'Ticker': t,
                        'Tipo': 'Compra' if qty_delta > 0 else 'Venda',
                        'Qtd': abs(qty_delta),
                        'Preço': price,
                        'Valor': abs(qty_delta * price)
                    })
        
        # === Registro diário do valor do portfólio ===
        period_dates = subset_prices.loc[rebal_date:next_date].index
        for date in period_dates:
            if date > end_date:
                break
            prices_date = subset_prices.loc[date]
            strat_val = sum(holdings.get(t, 0) * prices_date.get(t, 0) for t in holdings.keys())
            bench_val = benchmark_units * prices_date.get('DIVO11.SA', 0)
            portfolio_values.append({
                'Date': date,
                'Strategy_DCA': strat_val + cash,
                'DIVO11.SA_DCA': bench_val
            })
    
    df_curve = pd.DataFrame(portfolio_values).set_index('Date').drop_duplicates()
    df_transactions = pd.DataFrame(transactions)
    final_holdings = {t: q for t, q in holdings.items() if q > 0}
    
    return df_curve, df_transactions, final_holdings

# ==============================================================================
# MÓDULO 5: STREAMLIT APP
# ==============================================================================

def main():
    st.title("🚀 Quant Factor Lab Pro")

    st.sidebar.header("Configurações")
    tickers_input = st.sidebar.text_area(
        "Tickers (vírgula)", 
        "PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA, WEGE3.SA, MGLU3.SA, B3SA3.SA, SUZB3.SA, RADL3.SA, LREN3.SA"
    )
    start_date_str = st.sidebar.text_input("Início", "2018-01-01")
    end_date_str = st.sidebar.text_input("Fim", datetime.now().strftime("%Y-%m-%d"))
    top_n = st.sidebar.slider("Top N", 1, 20, 10)
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100.0, 5000.0, 500.0, 100.0)
    dca_years = st.sidebar.slider("Período DCA (anos)", 1, 10, 5)

    st.sidebar.subheader("Pesos dos Fatores")
    w_res = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.3, 0.05)
    w_fund = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.3, 0.05)
    w_val = st.sidebar.slider("Value", 0.0, 1.0, 0.2, 0.05)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.2, 0.05)

    st.sidebar.subheader("Opções")
    use_vol = st.sidebar.checkbox("Volatilidade Alvo 15%", False)
    use_sector = st.sidebar.checkbox("Neutralidade Setorial", False)

    weights_map = {'Res_Mom': w_res, 'Fund_Mom': w_fund, 'Value': w_val, 'Quality': w_qual}

    if st.sidebar.button("Rodar Análise"):
        with st.status("Processando...", expanded=True) as status:
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                start_1yr = end_date - timedelta(days=365)
                start_dca = end_date - timedelta(days=int(365.25 * dca_years))
            except:
                st.error("Data inválida.")
                return

            status.update(label="Baixando preços...")
            prices = fetch_price_data(tickers, start_date_str, end_date_str)
            if prices.empty:
                st.error("Erro ao baixar preços.")
                return

            status.update(label="Baixando fundamentos...")
            fundamentals_snapshot = fetch_fundamentals(tickers)

            # Simulação point-in-time
            idx = pd.MultiIndex.from_product([prices.index, fundamentals_snapshot.index], names=['Date', 'Ticker'])
            all_fundamentals = pd.DataFrame(index=idx, columns=fundamentals_snapshot.columns)
            last_date = prices.index[-1]
            for t in fundamentals_snapshot.index:
                all_fundamentals.loc[(last_date, t), :] = fundamentals_snapshot.loc[t]
            all_fundamentals = all_fundamentals.groupby('Ticker').ffill()

            # Ranking atual
            status.update(label="Calculando ranking atual...")
            mom_window = prices.tail(400)
            res_mom = compute_residual_momentum(mom_window)
            fund_mom = compute_fundamental_momentum(fundamentals_snapshot)
            val_score = compute_value_score(fundamentals_snapshot)
            qual_score = compute_quality_score(fundamentals_snapshot)

            final_df = pd.DataFrame(index=[c for c in prices.columns if c != 'DIVO11.SA'])
            final_df['Res_Mom'] = res_mom
            final_df['Fund_Mom'] = fund_mom
            final_df['Value'] = val_score
            final_df['Quality'] = qual_score
            if 'sector' in fundamentals_snapshot.columns:
                final_df['Sector'] = fundamentals_snapshot['sector']
            final_df.dropna(thresh=2, inplace=True)

            for c in ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']:
                if c in final_df.columns:
                    final_df[f"{c}_Z"] = robust_zscore(final_df[c])
            final_df = build_composite_score(final_df, {f"{c}_Z": weights_map[c] for c in weights_map if f"{c}_Z" in final_df.columns})

            # Backtests
            status.update(label="Backtesting...")
            bt_1yr = run_dynamic_backtest(prices, all_fundamentals, weights_map, top_n, use_vol, use_sector, start_1yr)
            bt_full = run_dynamic_backtest(prices, all_fundamentals, weights_map, top_n, use_vol, use_sector, start_dca)
            dca_curve, dca_tx, dca_hold = run_dca_backtest(prices, all_fundamentals, weights_map, top_n, dca_amount, use_vol, use_sector, start_dca, end_date)

            status.update(label="Concluído!", state="complete")

        # === TABS ===
        tabs = st.tabs(["🏆 Ranking", "📈 1 Ano", "💰 DCA", "💼 Carteira", "🔍 Detalhes"])

        with tabs[0]:
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader("Top Picks")
                cols = ['Composite_Score', 'Sector', 'Res_Mom', 'Fund_Mom', 'Value', 'Quality']
                st.dataframe(final_df[cols].head(top_n).style.background_gradient(subset=['Composite_Score'], cmap='RdYlGn'), use_container_width=True)
            with col2:
                st.subheader(f"Sugestão (R${dca_amount:.0f})")
                weights = construct_portfolio(final_df, prices.tail(90), top_n, 0.15 if use_vol else None)
                if not weights.empty:
                    p = prices.iloc[-1]
                    wdf = weights.to_frame("Peso")
                    wdf["Preço"] = p.reindex(wdf.index)
                    wdf["Alocação"] = wdf["Peso"] * dca_amount
                    wdf["Qtd"] = np.floor(wdf["Alocação"] / wdf["Preço"] + 0.5)
                    disp = wdf.copy()
                    disp["Peso"] = disp["Peso"].map("{:.1%}".format)
                    disp["Preço"] = disp["Preço"].map("R${:,.2f}".format)
                    disp["Alocação"] = disp["Alocação"].map("R${:,.0f}".format)
                    disp["Qtd"] = disp["Qtd"].astype(int)
                    st.dataframe(disp[["Peso", "Preço", "Qtd", "Alocação"]])
                    st.plotly_chart(px.pie(weights, names=weights.index, values=weights), use_container_width=True)

        # Demais tabs seguem lógica similar — você pode manter suas versões originais aqui

if __name__ == "__main__":
    main()

