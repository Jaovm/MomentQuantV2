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
    """Busca snapshots fundamentais expandidos para Value Robusto."""
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
            
            # Dados para Value Composite
            mkt_cap = info.get('marketCap', np.nan)
            op_cashflow = info.get('operatingCashflow', np.nan)
            ocf_yield = (op_cashflow / mkt_cap) if (mkt_cap and op_cashflow and mkt_cap > 0) else np.nan

            data.append({
                'ticker': t,
                'sector': sector,
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
        except:
            pass
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).set_index('ticker')

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic Aprimorados)
# ==============================================================================

def compute_residual_momentum_enhanced(price_df: pd.DataFrame, lookback=36, skip=1) -> pd.Series:
    """
    Residual Momentum Clássico (Blitz) com Volatility Scaling (Barroso-Santa-Clara).
    1. Regressão OLS de 36 meses (ou janela disponível).
    2. Calcula retornos residuais.
    3. Score = (Soma dos Resíduos 12m) / (Volatilidade dos Resíduos).
    """
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    
    # Janela de formação de 12 meses (Blitz 12-1) dentro de uma janela de regressão maior
    formation_window = 12 
    
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        
        # Pega até 36 meses para regressão estável, mas precisa de no mínimo 12
        y_full = rets[ticker].tail(lookback + skip)
        x_full = market.tail(lookback + skip)
        
        if len(y_full) < 12: continue
            
        try:
            # 1. Regressão para achar Alpha e Beta
            X = sm.add_constant(x_full.values)
            model = sm.OLS(y_full.values, X).fit()
            
            # 2. Extrair Resíduos (Retorno - (Alpha + Beta*Mkt))
            # Na prática, model.resid já nos dá isso
            residuals = pd.Series(model.resid, index=y_full.index)
            
            # 3. Momentum Formation: Soma dos resíduos dos últimos 12 meses, pulando o último (12-1)
            # Isso evita o efeito de reversão de curto prazo (reversal)
            resid_12m = residuals.iloc[-(formation_window + skip) : -skip]
            
            if len(resid_12m) == 0:
                scores[ticker] = 0
                continue

            # Momentum acumulado (Blitz)
            raw_momentum = resid_12m.sum()
            
            # 4. Volatility Scaling (Barroso-Santa-Clara simplificado)
            # Normalizamos pelo desvio padrão dos resíduos (Information Ratio proxy)
            # Isso penaliza momentum construído com alta volatilidade
            resid_vol = residuals.std()
            
            if resid_vol == 0:
                scores[ticker] = 0
            else:
                scores[ticker] = raw_momentum / resid_vol

        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    """Z-Score combinado de crescimento de Receita e Lucro."""
    metrics = ['earningsGrowth', 'revenueGrowth']
    temp_df = pd.DataFrame(index=fund_df.index)
    for m in metrics:
        if m in fund_df.columns:
            s = fund_df[m].fillna(fund_df[m].median())
            temp_df[m] = (s - s.mean()) / s.std()
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_robust(fund_df: pd.DataFrame) -> pd.Series:
    """
    Score de Valor Robusto (Composite).
    Média Z-Score de: 1/PE(Fwd), 1/PE(Trail), 1/PB, 1/EV_EBITDA, OCF_Yield.
    """
    scores = pd.DataFrame(index=fund_df.index)
    
    # Helper para inverter múltiplos (quanto menor melhor -> score alto)
    def invert_metric(series):
        # Trata valores negativos ou zero (ex: prejuízo) como score ruim (0 ou Z baixo)
        # Aqui, substituímos <=0 por NaN temporariamente ou um valor alto para penalizar
        valid = np.where(series > 0, 1/series, np.nan) 
        return pd.Series(valid, index=series.index)

    if 'forwardPE' in fund_df: scores['EP_Fwd'] = invert_metric(fund_df['forwardPE'])
    if 'trailingPE' in fund_df: scores['EP_Trail'] = invert_metric(fund_df['trailingPE'])
    if 'priceToBook' in fund_df: scores['BP'] = invert_metric(fund_df['priceToBook'])
    if 'enterpriseToEbitda' in fund_df: scores['EBITDA_Yld'] = invert_metric(fund_df['enterpriseToEbitda'])
    if 'ocfYield' in fund_df: scores['OCF_Yld'] = fund_df['ocfYield'] # Já é yield (quanto maior melhor)

    # Normaliza cada coluna (Z-score) antes de combinar
    for col in scores.columns:
        # Preenche NaNs com a média para não penalizar excessivamente falta de dados pontual, 
        # mas idealmente penaliza empresas com P/L negativo (que viraram NaN no invert)
        filled = scores[col].fillna(scores[col].min()) 
        scores[col] = (filled - filled.mean()) / filled.std()

    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: scores['DE_Inv'] = -1 * fund_df['debtToEquity'] # Menor dívida é melhor
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto usando Mediana e MAD (Mean Absolute Deviation)."""
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: return series - median 
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
# MÓDULO 4: PORTFOLIO & BACKTEST 
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio e ordena do maior para o menor peso."""
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    # Volatility Weighting (Risk Parity simples)
    if vol_target is not None:
        recent_rets = prices[selected].pct_change().tail(63) # ~3 meses de vol recente
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
    """
    Backtest Walk-Forward. 
    Retorna:
    1. Equity Curve da Estratégia Composta.
    2. Equity Curves Individuais dos Fatores (Para Factor Timing).
    """
    end_date = all_prices.index[-1]
    # Pega dados suficientes antes do start para cálculo de momentum
    subset_prices = all_prices.loc[start_date_backtest - timedelta(days=500):end_date]
    rebalance_dates = subset_prices.loc[start_date_backtest:end_date].resample('MS').first().index.tolist()
    
    if not rebalance_dates:
        return pd.DataFrame(), pd.DataFrame()

    strategy_rets = []
    bench_rets = []
    
    # Dicionário para guardar retornos isolados dos fatores
    factor_tracking = {k: [] for k in ['Res_Mom', 'Value', 'Quality']}

    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
        
        # Janelas de dados
        prices_historical = subset_prices.loc[:rebal_date]
        mom_window = prices_historical.tail(500) 
        risk_window = prices_historical.tail(90)
        
        # Cálculos de Fatores (Raw)
        res_mom = compute_residual_momentum_enhanced(mom_window)
        val_score = compute_value_robust(all_fundamentals)
        qual_score = compute_quality_score(all_fundamentals)
        
        # DataFrame do Período
        df_period = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_period['Res_Mom'] = res_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        
        if 'sector' in all_fundamentals.columns: 
             df_period['Sector'] = all_fundamentals['sector']
        
        df_period.dropna(thresh=2, inplace=True)
        
        # Normalização (Z-Score)
        norm_cols = ['Res_Mom', 'Value', 'Quality']
        w_keys = {}
        
        # Neutralidade Setorial ou Global
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
        
        # --- 1. Estratégia Principal (Composite) ---
        ranked_period = build_composite_score(df_period, w_keys)
        current_weights = construct_portfolio(ranked_period, risk_window, top_n, 0.15 if use_vol_target else None)
        
        # Retornos do Período (Próximo Mês)
        market_period = subset_prices.loc[rebal_date:next_date].iloc[1:] 
        period_pct = market_period.pct_change().dropna()
        if period_pct.empty: continue
            
        valid_tickers = [t for t in current_weights.index if t in period_pct.columns]
        strat_ret = period_pct[valid_tickers].dot(current_weights[valid_tickers]) if valid_tickers else pd.Series(0.0, index=period_pct.index)
        
        strategy_rets.append(strat_ret)
        bench_rets.append(period_pct['BOVA11.SA'] if 'BOVA11.SA' in period_pct.columns else pd.Series(0.0, index=period_pct.index))
        
        # --- 2. Tracking Individual de Fatores (Para Gráfico de Timing) ---
        # Calculamos carteiras "Puras" (Top N stocks baseadas só naquele score)
        for factor in ['Res_Mom', 'Value', 'Quality']:
            col_z = f"{factor}_Z"
            if col_z in df_period.columns:
                # Top N só por esse fator
                top_factor = df_period.sort_values(col_z, ascending=False).head(top_n).index
                valid_f = [t for t in top_factor if t in period_pct.columns]
                if valid_f:
                    # Equal weight para simplificar a visualização do fator puro
                    f_ret = period_pct[valid_f].mean(axis=1) 
                    factor_tracking[factor].append(f_ret)
                else:
                    factor_tracking[factor].append(pd.Series(0.0, index=period_pct.index))

    # Consolidação
    if strategy_rets:
        full_strategy = pd.concat(strategy_rets)
        full_benchmark = pd.concat(bench_rets)
        full_strategy = full_strategy[~full_strategy.index.duplicated(keep='first')]
        full_benchmark = full_benchmark[~full_benchmark.index.duplicated(keep='first')]
        
        cumulative = pd.DataFrame({
            'Strategy': (1 + full_strategy).cumprod(),
            'BOVA11.SA': (1 + full_benchmark).cumprod()
        })
        
        # Consolida Fatores
        factor_cum = pd.DataFrame(index=cumulative.index)
        for f, rets_list in factor_tracking.items():
            if rets_list:
                s = pd.concat(rets_list)
                s = s[~s.index.duplicated(keep='first')]
                # Reindex para alinhar com cumulative caso falte dias
                s = s.reindex(cumulative.index).fillna(0)
                factor_cum[f] = (1 + s).cumprod()
        
        return cumulative.dropna(), factor_cum.dropna()
        
    return pd.DataFrame(), pd.DataFrame()

def run_dca_backtest(
    all_prices: pd.DataFrame, 
    all_fundamentals: pd.DataFrame, 
    factor_weights: dict, 
    top_n: int, 
    dca_amount: float, 
    use_vol_target: bool,
    use_sector_neutrality: bool, 
    start_date: datetime,
    end_date: datetime
):
    """Simula um Backtest com Aportes Mensais (DCA) e rebalanceamento."""
    all_prices = all_prices.ffill() 
    dca_start = start_date + timedelta(days=30) 
    dates = all_prices.loc[dca_start:end_date].resample('MS').first().index.tolist()
    
    if not dates or len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    portfolio_value = pd.Series(0.0, index=all_prices.index)
    benchmark_value = pd.Series(0.0, index=all_prices.index)
    portfolio_holdings = {} 
    benchmark_holdings = {'BOVA11.SA': 0.0}
    monthly_transactions = []
    
    for i, month_start in enumerate(dates):
        eval_date = month_start - timedelta(days=1)
        mom_start = month_start - timedelta(days=500) 
        prices_for_mom = all_prices.loc[mom_start:eval_date] 
        risk_start = month_start - timedelta(days=90)
        prices_for_risk = all_prices.loc[risk_start:eval_date]
        
        res_mom = compute_residual_momentum_enhanced(prices_for_mom) if not prices_for_mom.empty else pd.Series(dtype=float)
        val_score = compute_value_robust(all_fundamentals)
        qual_score = compute_quality_score(all_fundamentals)

        df_master = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_master['Res_Mom'] = res_mom
        df_master['Value'] = val_score
        df_master['Quality'] = qual_score
        
        if 'sector' in all_fundamentals.columns: 
             df_master['Sector'] = all_fundamentals['sector']
        df_master.dropna(thresh=2, inplace=True)
        
        norm_cols_dca = ['Res_Mom', 'Value', 'Quality']
        weights_keys = {}
        
        if use_sector_neutrality and 'Sector' in df_master.columns and df_master['Sector'].nunique() > 1:
            for c in norm_cols_dca:
                if c in df_master.columns:
                    new_col = f"{c}_Z_Sector"
                    df_master[new_col] = df_master.groupby('Sector')[c].transform(
                        lambda x: robust_zscore(x) if len(x) > 1 else x - x.median() 
                    )
                    weights_keys[new_col] = factor_weights.get(c, 0.0)
        else:
            for c in norm_cols_dca:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    weights_keys[new_col] = factor_weights.get(c, 0.0)

        final_df = build_composite_score(df_master, weights_keys)
        current_weights = construct_portfolio(final_df, prices_for_risk, top_n, 0.15 if use_vol_target else None)
        
        try:
            rebal_price = all_prices.loc[all_prices.index >= month_start].iloc[0].to_frame().T
        except IndexError:
            break
            
        cash_for_strategy = dca_amount 
        bova_price = rebal_price['BOVA11.SA'].iloc[0]
        if not np.isnan(bova_price) and bova_price > 0:
            q_bova = dca_amount / bova_price
            benchmark_holdings['BOVA11.SA'] += q_bova
            monthly_transactions.append({
                'Date': rebal_price.index[0], 
                'Ticker': 'BOVA11.SA',
                'Action': 'Buy (DCA)',
                'Quantity': q_bova,
                'Price': bova_price,
                'Value_R$': dca_amount
            })
            
        for ticker, weight in current_weights.items():
            if ticker in rebal_price.columns and not rebal_price[ticker].isna().iloc[0]:
                price = rebal_price[ticker].iloc[0]
                if price > 0 and weight > 0:
                    amount = cash_for_strategy * weight
                    quantity = amount / price
                    portfolio_holdings[ticker] = portfolio_holdings.get(ticker, 0.0) + quantity
                    monthly_transactions.append({
                        'Date': rebal_price.index[0], 
                        'Ticker': ticker,
                        'Action': 'Buy (DCA)',
                        'Quantity': quantity,
                        'Price': price,
                        'Value_R$': amount
                    })
        
        next_month_start = dates[i+1] if i < len(dates) - 1 else end_date
        valuation_dates = all_prices.loc[rebal_price.index[0]:next_month_start].index
        
        for current_date in valuation_dates:
            current_port_value = 0.0
            for ticker, quantity in portfolio_holdings.items():
                if ticker in all_prices.columns and current_date in all_prices.index:
                    price = all_prices.loc[current_date, ticker]
                    current_port_value += price * quantity
            portfolio_value[current_date] = current_port_value
            benchmark_value[current_date] = all_prices.loc[current_date, 'BOVA11.SA'] * benchmark_holdings['BOVA11.SA']
        
    portfolio_value = portfolio_value[portfolio_value > 0].ffill().dropna()
    benchmark_value = benchmark_value[benchmark_value > 0].ffill().dropna()
    
    equity_curve = pd.DataFrame({'Strategy_DCA': portfolio_value, 'BOVA11.SA_DCA': benchmark_value})
    final_holdings = {k: v for k, v in portfolio_holdings.items() if v > 0}
    
    return equity_curve, pd.DataFrame(monthly_transactions), final_holdings

# ==============================================================================
# APP PRINCIPAL (STREAMLIT UI)
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Pro v2")
    st.markdown("Otimização de carteira Long-Only baseada em **Robust Value** e **Residual Momentum (Blitz)**.")

    st.sidebar.header("1. Universo e Dados (BOVESPA)")
    default_univ = "ITUB3.SA, TOTS3.SA, MDIA3.SA, TAEE3.SA, BBSE3.SA, WEGE3.SA, PSSA3.SA, EGIE3.SA, B3SA3.SA, VIVT3.SA, AGRO3.SA, PRIO3.SA, BBAS3.SA, BPAC11.SA, SBSP3.SA, SAPR4.SA, CMIG3.SA, UNIP6.SA, FRAS3.SA, VALE3.SA, PETR4.SA, RENT3.SA, LREN3.SA"
    ticker_input = st.sidebar.text_area("Tickers (Separados por vírgula)", default_univ, height=100)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores (Alpha)")
    st.sidebar.info("Nota: Value agora é Composite (P/E + EV/EBITDA + Yield). Momentum usa Resíduos + Vol Scaling.")
    w_rm = st.sidebar.slider("Residual Momentum (Enhanced)", 0.0, 1.0, 0.50)
    w_val = st.sidebar.slider("Robust Value", 0.0, 1.0, 0.30)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Construção de Portfólio (Risco)")
    top_n = st.sidebar.number_input("Número de Ativos (Top N)", 1, 20, 8)
    use_vol_target = st.sidebar.checkbox("Usar Ponderação por Risco Inverso?", True)
    target_vol = st.sidebar.slider("Volatilidade Alvo (Ref)", 0.05, 0.30, 0.15) if use_vol_target else None
    
    st.sidebar.markdown("---")
    st.sidebar.header("4. Diversificação")
    use_sector_neutrality = st.sidebar.checkbox("Usar Neutralidade Setorial?", True)
    
    st.sidebar.header("5. Simulação Mensal (DCA)")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 10000, 2000)
    dca_years = st.sidebar.slider("Anos de Backtest DCA", 1, 5, 3)
    
    run_btn = st.sidebar.button("🚀 Rodar Análise", type="primary")

    if run_btn:
        if not tickers:
            st.error("Insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant...", expanded=True) as status:
            end_date = datetime.now()
            # Precisamos de histórico longo para Momentum Clássico (36m)
            start_date_total = end_date - timedelta(days=365 * (dca_years + 3)) 
            start_date_1yr = end_date - timedelta(days=365)
            start_date_dca_period = end_date - timedelta(days=365 * dca_years)

            prices = fetch_price_data(tickers, start_date_total, end_date)
            fundamentals = fetch_fundamentals(tickers) 
            
            if prices.empty or fundamentals.empty:
                st.error("Dados insuficientes.")
                status.update(label="Erro!", state="error")
                return
            
            # --- Cálculo Atual (Snapshot) ---
            res_mom = compute_residual_momentum_enhanced(prices)
            val_score = compute_value_robust(fundamentals)
            qual_score = compute_quality_score(fundamentals)

            df_master = pd.DataFrame(index=tickers)
            df_master['Res_Mom'] = res_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            
            if 'sector' in fundamentals.columns: df_master['Sector'] = fundamentals['sector']
            df_master.dropna(thresh=2, inplace=True)

            norm_cols = ['Res_Mom', 'Value', 'Quality']
            cols_show = []
            weights_map = {'Res_Mom': w_rm, 'Value': w_val, 'Quality': w_qual}
            weights_keys = {}

            if use_sector_neutrality and 'Sector' in df_master.columns and df_master['Sector'].nunique() > 1:
                for c in norm_cols:
                    if c in df_master.columns:
                        new_col = f"{c}_Z_Sector"
                        df_master[new_col] = df_master.groupby('Sector')[c].transform(
                            lambda x: robust_zscore(x) if len(x) > 1 else x - x.median()
                        )
                        weights_keys[new_col] = weights_map.get(c, 0.0)
                        cols_show.append(new_col)
            else:
                for c in norm_cols:
                    if c in df_master.columns:
                        new_col = f"{c}_Z"
                        df_master[new_col] = robust_zscore(df_master[c])
                        weights_keys[new_col] = weights_map.get(c, 0.0)
                        cols_show.append(new_col)
            
            final_df = build_composite_score(df_master, weights_keys)
            weights = construct_portfolio(final_df, prices, top_n, target_vol)
            
            # --- Backtests ---
            backtest_dynamic, factor_timing_df = run_dynamic_backtest(prices, fundamentals, weights_map, top_n, use_vol_target, use_sector_neutrality, start_date_dca_period)
            dca_curve, dca_transactions, dca_holdings = run_dca_backtest(prices, fundamentals, weights_map, top_n, dca_amount, use_vol_target, use_sector_neutrality, start_date_dca_period, end_date)

            status.update(label="Concluído!", state="complete", expanded=False)

        # ==============================================================================
        # DASHBOARD
        # ==============================================================================
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 Ranking Atual", 
            "📈 Performance & TIR", 
            "📊 Factor Timing", 
            "💰 Backtest DCA", 
            "💼 Alocação Final", 
            "🔍 Detalhes"
        ])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Top Picks (Score Atual)")
                show_list = ['Composite_Score', 'Sector'] + cols_show
                st.dataframe(final_df[show_list].head(top_n).style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']), height=400, width=1000)
            with col2:
                st.subheader("Boleta Sugerida")
                if not weights.empty:
                    latest_prices = prices.iloc[-1]
                    w_df = weights.to_frame(name="Peso")
                    w_df["Preço (R$)"] = latest_prices.reindex(w_df.index).fillna(0)
                    w_df["Alocação (R$)"] = w_df["Peso"] * dca_amount
                    w_df["Qtd. Aprox"] = np.where(w_df["Preço (R$)"] > 0, np.floor((w_df["Alocação (R$)"] / w_df["Preço (R$)"]) + 0.5), 0)

                    display_df = w_df.copy()
                    display_df["Peso"] = display_df["Peso"].map("{:.2%}".format)
                    display_df["Preço (R$)"] = display_df["Preço (R$)"].map("R$ {:,.2f}".format)
                    display_df["Alocação (R$)"] = display_df["Alocação (R$)"].map("R$ {:,.2f}".format)
                    display_df["Qtd. Aprox"] = display_df["Qtd. Aprox"].map("{:.0f}".format)
                    st.table(display_df[["Qtd. Aprox", "Alocação (R$)"]])
                    
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Alocação Teórica")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader(f"Performance Dinâmica ({dca_years} Anos)")
            if not backtest_dynamic.empty:
                # Métricas
                total_ret = backtest_dynamic.iloc[-1] - 1
                daily = backtest_dynamic.pct_change().dropna()
                vol = daily.std() * (252**0.5)
                sharpe = (total_ret - (0.10 * dca_years)) / vol # Risk free approx
                max_drawdown = (backtest_dynamic / backtest_dynamic.cummax() - 1).min()

                # Cálculo de CAGR (TIR Histórica)
                days = (backtest_dynamic.index[-1] - backtest_dynamic.index[0]).days
                cagr = (backtest_dynamic.iloc[-1])**(365/days) - 1

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CAGR (TIR Histórica)", f"{cagr['Strategy']:.2%}", delta=f"vs Bench {cagr['BOVA11.SA']:.2%}")
                c2.metric("Sharpe Ratio", f"{sharpe['Strategy']:.2f}")
                c3.metric("Max Drawdown", f"{max_drawdown['Strategy']:.2%}")
                c4.metric("Volatilidade", f"{vol['Strategy']:.2%}")
                
                st.plotly_chart(px.line(backtest_dynamic, title="Curva de Retorno (Base 1.0)", color_discrete_map={'Strategy': '#00CC96', 'BOVA11.SA': '#EF553B'}), use_container_width=True)
                
                # --- Projeção de TIR ---
                st.markdown("---")
                st.subheader("🔮 Projeção de Patrimônio (Baseado no CAGR Histórico)")
                pc1, pc2 = st.columns([1, 2])
                with pc1:
                    proj_years = st.slider("Anos a Projetar", 1, 20, 10)
                    initial_inv = st.number_input("Investimento Inicial", value=float(dca_amount))
                    monthly_inv = st.number_input("Aporte Mensal Futuro", value=float(dca_amount))
                    safety_margin = st.slider("Margem de Segurança (Reduzir CAGR)", 0.0, 0.5, 0.20, help="Reduz o CAGR histórico em X% para ser conservador.")
                
                with pc2:
                    adjusted_rate = cagr['Strategy'] * (1 - safety_margin)
                    monthly_rate = (1 + adjusted_rate)**(1/12) - 1
                    
                    future_values = []
                    dates_future = []
                    current_val = initial_inv
                    today = datetime.now()
                    
                    for m in range(proj_years * 12):
                        current_val = current_val * (1 + monthly_rate) + monthly_inv
                        future_values.append(current_val)
                        dates_future.append(today + timedelta(days=30*m))
                        
                    df_proj = pd.DataFrame({'Patrimônio Projetado': future_values}, index=dates_future)
                    final_proj_val = future_values[-1]
                    
                    st.metric(f"Patrimônio em {proj_years} anos (CAGR Aj. {adjusted_rate:.1%})", f"R$ {final_proj_val:,.2f}")
                    st.area_chart(df_proj)

        with tab3:
            st.subheader("📊 Factor Timing: Qual estilo está vencendo?")
            st.markdown("Comparativo de performance acumulada de portfólios 'Puros' (Top N ativos classificados apenas por aquele fator).")
            
            if not factor_timing_df.empty:
                # Adiciona o benchmark para contexto
                plot_data = factor_timing_df.copy()
                plot_data['BOVA11'] = backtest_dynamic['BOVA11.SA']
                
                # Gráfico Linha
                fig_factors = px.line(plot_data, title="Batalha dos Fatores (Base 1.0)")
                st.plotly_chart(fig_factors, use_container_width=True)
                
                # Correlação Recente (Rolling)
                st.subheader("Correlação Dinâmica (Últimos 12 meses)")
                recent_corr = plot_data.pct_change().tail(252).corr()
                st.heatmap = px.imshow(recent_corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlação de Retornos Recentes")
                st.plotly_chart(st.heatmap, use_container_width=True)
            else:
                st.warning("Dados insuficientes para gerar Factor Timing.")

        with tab4:
            st.header(f"Simulação DCA Realista")
            if not dca_curve.empty:
                final_strat = dca_curve['Strategy_DCA'].iloc[-1]
                invested = len(dca_transactions['Date'].unique()) * dca_amount
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Investido (Cash)", f"R${invested:,.2f}")
                c2.metric("Saldo Estratégia", f"R${final_strat:,.2f}", delta=f"{((final_strat/invested)-1):.2%}")
                c3.metric("Saldo Benchmark", f"R${dca_curve['BOVA11.SA_DCA'].iloc[-1]:,.2f}")
                
                st.plotly_chart(px.line(dca_curve, title="Evolução Patrimonial (Aportes Mensais)"), use_container_width=True)
                st.subheader("Histórico de Execução (Boletas)")
                dca_transactions['Date'] = pd.to_datetime(dca_transactions['Date']).dt.strftime('%Y-%m-%d')
                st.dataframe(dca_transactions.set_index('Date'), height=300, use_container_width=True)

        with tab5:
            st.header("💼 Carteira Atual (Backtest DCA)")
            if dca_holdings:
                clean_holdings = {k: v for k, v in dca_holdings.items() if k != 'BOVA11.SA'}
                if clean_holdings:
                    last_prices = prices.iloc[-1]
                    alloc_data = []
                    total_val = 0
                    for t, qtd in clean_holdings.items():
                        if t in last_prices:
                            val = qtd * last_prices[t]
                            alloc_data.append({'Ticker': t, 'Quantidade': qtd, 'Valor_Atual': val})
                            total_val += val
                    if total_val > 0:
                        df_alloc = pd.DataFrame(alloc_data)
                        df_alloc['Peso (%)'] = df_alloc['Valor_Atual'] / total_val
                        df_alloc = df_alloc.sort_values('Peso (%)', ascending=False)
                        col_chart, col_table = st.columns([1, 1])
                        with col_chart:
                            fig_donut = px.pie(df_alloc, values='Valor_Atual', names='Ticker', title="Distribuição Financeira", hole=0.45)
                            st.plotly_chart(fig_donut, use_container_width=True)
                        with col_table:
                            st.subheader("Detalhamento de Custódia")
                            df_display = df_alloc.copy()
                            df_display['Valor_Atual'] = df_display['Valor_Atual'].map('R${:,.2f}'.format)
                            df_display['Peso (%)'] = df_display['Peso (%)'].map('{:.2%}'.format)
                            st.dataframe(df_display.set_index('Ticker'), use_container_width=True)
            else:
                st.info("Nenhuma posição em custódia no momento.")

        with tab6:
            st.subheader("Dados Fundamentais Brutos")
            st.dataframe(fundamentals)
            st.subheader("Matriz de Correlação dos Fatores (Ranking Atual)")
            if cols_show:
                corr = final_df[cols_show].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

if __name__ == "__main__":
    main()
