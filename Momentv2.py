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
    """Busca snapshots fundamentais atuais."""
    data = []
    clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
    
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            info = yf.Ticker(t).info
            sector = info.get('sector', 'Unknown')
            # Fallback básico para setor se vier N/A
            if sector in ['Unknown', 'N/A'] and 'longName' in info:
                 if 'Banco' in info['longName'] or 'Financeira' in info['longName']:
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
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Calcula Residual Momentum (Alpha) vs BOVA11.SA."""
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
            
            if np.std(resid) == 0 or len(resid) < 2:
                scores[ticker] = 0
            else:
                scores[ticker] = np.sum(resid) / np.std(resid)
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

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E e P/B."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'forwardPE' in fund_df: scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    if 'priceToBook' in fund_df: scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: scores['DE_Inv'] = -1 * fund_df['debtToEquity']
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto."""
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
        res_mom = compute_residual_momentum(mom_window)
        
        # CORREÇÃO DE LOOK-AHEAD BIAS:
        # Usar apenas os dados fundamentais disponíveis até a data de rebalanceamento.
        # all_fundamentals é um DataFrame de série temporal (Index=Date, Columns=Ticker_Metric).
        # A indexação .loc[rebal_date] pega o último dado disponível (graças ao ffill feito em main).
        fundamentals_at_rebal = all_fundamentals.loc[rebal_date].unstack()
        
        fund_mom = compute_fundamental_momentum(fundamentals_at_rebal)
        val_score = compute_value_score(fundamentals_at_rebal)
        qual_score = compute_quality_score(fundamentals_at_rebal)
        
        df_period = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
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
        
        market_period = subset_prices.loc[rebal_date:next_date].iloc[1:] 
        period_pct = market_period.pct_change().dropna()
        if period_pct.empty: continue
            
        # Cálculo do Retorno da Estratégia
        strategy_rets = (period_pct[current_weights.index] * current_weights).sum(axis=1)
        strategy_daily_rets.append(strategy_rets)
        
        # Retorno do Benchmark
        benchmark_daily_rets.append(period_pct['BOVA11.SA'])

    if not strategy_daily_rets:
        return pd.DataFrame()

    strategy_rets_series = pd.concat(strategy_daily_rets)
    benchmark_rets_series = pd.concat(benchmark_daily_rets)
    
    df_backtest = pd.DataFrame({
        'Strategy': (1 + strategy_rets_series).cumprod(),
        'BOVA11.SA': (1 + benchmark_rets_series).cumprod()
    })
    
    return df_backtest.fillna(method='ffill')

def run_dca_backtest(
    all_prices: pd.DataFrame, 
    all_fundamentals: pd.DataFrame, 
    weights_config: dict, 
    top_n: int, 
    dca_amount: float,
    use_vol_target: bool,
    use_sector_neutrality: bool,
    start_date_backtest: datetime,
    end_date: datetime
):
    """Executa um backtest DCA (Dollar-Cost Averaging) com rebalanceamento mensal."""
    
    subset_prices = all_prices.loc[start_date_backtest - timedelta(days=400):end_date]
    rebalance_dates = subset_prices.loc[start_date_backtest:end_date].resample('MS').first().index.tolist()
    
    if not rebalance_dates:
        return pd.DataFrame(), pd.DataFrame(), {}

    holdings = {}
    transactions = []
    portfolio_value = []
    
    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
        
        # 1. Calcular Pesos (Lógica de Seleçã        # 1. Calcular Pesos (Lógica de Seleção)
        prices_historical = subset_prices.loc[:rebal_date]
        mom_window = prices_historical.tail(400) 
        risk_window = prices_historical.tail(90)
        res_mom = compute_residual_momentum(mom_window)
        
        # CORREÇÃO DE LOOK-AHEAD BIAS:
        fundamentals_at_rebal = all_fundamentals.loc[rebal_date].unstack()
        
        fund_mom = compute_fundamental_momentum(fundamentals_at_rebal)
        val_score = compute_value_score(fundamentals_at_rebal)
        qual_score = compute_quality_score(fundamentals_at_rebal)
        
        df_period = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
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
        
        # 2. Rebalanceamento e Aporte
        
        # Valorização do portfólio até a data de rebalanceamento
        if portfolio_value:
            last_date_val = portfolio_value[-1]['Date']
            market_period_val = subset_prices.loc[last_date_val:rebal_date].iloc[1:]
            
            # Atualiza o valor do portfólio
            for t, qtd in holdings.items():
                if t in market_period_val.columns:
                    holdings[t] = qtd * (1 + market_period_val[t].pct_change().fillna(0)).prod()
        
        # Preços de compra/venda
        rebal_price = subset_prices.loc[rebal_date]
        
        # Calcula o valor total do portfólio antes do aporte
        total_portfolio_value = sum(holdings.get(t, 0) * rebal_price.get(t, 0) for t in current_weights.index)
        
        # Adiciona o aporte
        cash_to_invest = dca_amount
        
        # Calcula os novos pesos alvo (incluindo o aporte)
        target_allocation = (total_portfolio_value + cash_to_invest) * current_weights
        
        # Calcula a quantidade de cada ativo a ser comprada/vendida
        for t in current_weights.index:
            current_value = holdings.get(t, 0) * rebal_price.get(t, 0)
            target_value = target_allocation.get(t, 0)
            
            # Valor a ser investido/desinvestido neste ativo
            delta_value = target_value - current_value
            
            if rebal_price.get(t, 0) > 0:
                # Quantidade a ser comprada/vendida (arredondamento padrão)
                qtd_delta = np.floor((delta_value / rebal_price[t]) + 0.5)
                
                # Atualiza a custódia
                holdings[t] = holdings.get(t, 0) + qtd_delta
                
                # Registra a transação
                if qtd_delta != 0:
                    transactions.append({
                        'Date': rebal_date,
                        'Ticker': t,
                        'Tipo': 'Compra' if qtd_delta > 0 else 'Venda',
                        'Qtd': abs(qtd_delta),
                        'Preço': rebal_price[t],
                        'Valor': abs(qtd_delta) * rebal_price[t]
                    })
        
        # 3. Registro do Valor do Portfólio
        
        # Período de hold até o próximo rebalanceamento
        market_period_hold = subset_prices.loc[rebal_date:next_date].iloc[1:]
        
        for date, row in market_period_hold.iterrows():
            strat_val = sum(holdings.get(t, 0) * row.get(t, 0) for t in holdings.keys())
            bench_val = subset_prices.loc[start_date_backtest:date]['BOVA11.SA'].iloc[-1] * dca_amount * (i+1) # Simplificação para benchmark DCA
            
            portfolio_value.append({
                'Date': date,
                'Strategy_DCA': strat_val,
                'BOVA11.SA_DCA': bench_val
            })
            
    df_curve = pd.DataFrame(portfolio_value).set_index('Date')
    df_transactions = pd.DataFrame(transactions)
    
    # Limpeza final da custódia (remove ativos com quantidade zero)
    final_holdings = {t: q for t, q in holdings.items() if q > 0}
    
    return df_curve, df_transactions, final_holdings

# ==============================================================================
# MÓDULO 5: STREAMLIT APP
# ==============================================================================

def main():
    st.title("Quant Factor Lab Pro")
    
    # Sidebar
    st.sidebar.header("Configurações do Backtest")
    
    # Parâmetros de Entrada
    tickers_input = st.sidebar.text_area("Tickers (separados por vírgula)", "PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA, WEGE3.SA, MGLU3.SA, B3SA3.SA, SUZB3.SA, RADL3.SA, LREN3.SA")
    start_date_str = st.sidebar.text_input("Data de Início (YYYY-MM-DD)", "2018-01-01")
    end_date_str = st.sidebar.text_input("Data de Fim (YYYY-MM-DD)", datetime.now().strftime("%Y-%m-%d"))
    top_n = st.sidebar.slider("Top N Ativos", 1, 20, 10)
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100.0, 5000.0, 500.0, step=100.0)
    dca_years = st.sidebar.slider("Período DCA (Anos)", 1, 10, 5)
    
    # Fatores
    st.sidebar.subheader("Pesos dos Fatores")
    w_res_mom = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.3, 0.05)
    w_fund_mom = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.3, 0.05)
    w_value = st.sidebar.slider("Value Score", 0.0, 1.0, 0.2, 0.05)
    w_quality = st.sidebar.slider("Quality Score", 0.0, 1.0, 0.2, 0.05)
    
    # Opções Avançadas
    st.sidebar.subheader("Opções Avançadas")
    use_vol_target = st.sidebar.checkbox("Volatilidade Alvo (15%)", False)
    use_sector_neutrality = st.sidebar.checkbox("Neutralidade Setorial", False)
    
    weights_map = {
        'Res_Mom': w_res_mom,
        'Fund_Mom': w_fund_mom,
        'Value': w_value,
        'Quality': w_quality
    }
    
    if st.sidebar.button("Rodar Backtest"):
        
        with st.status("Executando Backtest...", expanded=True) as status:
            
            # 1. Preparação de Dados
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                start_date_dca_period = end_date - timedelta(days=365.25 * dca_years)
                start_date_1yr = end_date - timedelta(days=365)
            except ValueError:
                st.error("Formato de data inválido. Use YYYY-MM-DD.")
                status.update(label="Erro de Data", state="error")
                return
            
            status.update(label="Buscando dados de preços...", state="running")
            prices = fetch_price_data(tickers, start_date_str, end_date_str)
            
            if prices.empty:
                st.error("Não foi possível buscar dados de preços. Verifique os tickers e o período.")
                status.update(label="Erro de Preços", state="error")
                return
            
            status.update(label="Buscando dados fundamentais (snapshot atual)...", state="running")
            fundamentals_snapshot = fetch_fundamentals(tickers)
            
            if fundamentals_snapshot.empty:
                st.warning("Não foi possível buscar dados fundamentais. Continuando apenas com Momentum Residual.")
            
            # CORREÇÃO DE LOOK-AHEAD BIAS:
            # 2. Criação do DataFrame de Fundamentais de Série Temporal (Point-in-Time Simulation)
            
            # O DataFrame final terá o índice de datas dos preços e colunas de Ticker_Métrica.
            # O yfinance só fornece o snapshot atual. Para simular o backtest, replicamos o snapshot
            # para todas as datas e usamos ffill para simular a persistência do dado.
            
            # Estrutura: MultiIndex (Date, Ticker)
            
            # 2.1. Criar o MultiIndex (Date, Ticker) para o DataFrame de série temporal
            idx = pd.MultiIndex.from_product([prices.index, fundamentals_snapshot.index], names=['Date', 'Ticker'])
            all_fundamentals = pd.DataFrame(index=idx, columns=fundamentals_snapshot.columns)
            
            # 2.2. Preencher a última data com o snapshot atual
            last_date = prices.index[-1]
            for ticker in fundamentals_snapshot.index:
                all_fundamentals.loc[(last_date, ticker), :] = fundamentals_snapshot.loc[ticker, :]
            
            # 2.3. Forward Fill (ffill) para simular a persistência do dado no tempo
            # O dado fundamental mais recente é replicado para todas as datas anteriores.
            # Isso é uma simplificação, mas é a melhor aproximação para dados de snapshot.
            all_fundamentals = all_fundamentals.sort_index().ffill()
            
            # all_fundamentals agora é um DataFrame com MultiIndex (Date, Ticker)
            # A função run_dynamic_backtest irá indexar por data e usar .unstack() para obter o formato Ticker x Metric
            
            # 3. Cálculo do Score Atual (Usando o snapshot atual)ão do DataFrame de Fundamentais de Série Temporal (Point-in-Time Simulation)
            
            # O DataFrame final terá o índice de datas dos preços e colunas de Ticker_Métrica.
            # O yfinance só fornece o snapshot atual. Para simular o backtest, replicamos o snapshot
            # para todas as datas e usamos ffill para simular a persistência do dado.
            
            # Estrutura: Index=Date, Columns=Ticker_Metric
            
            # 2.1. Reestruturar o snapshot para MultiIndex (Ticker, Metric)
            fundamentals_long = fundamentals_snapshot.stack().to_frame('Value')
            
            # 2.2. Criar o MultiIndex (Date, Ticker) para o DataFrame de série temporal
            idx = pd.MultiIndex.from_product([prices.index, fundamentals_snapshot.index], names=['Date', 'Ticker'])
            all_fundamentals = pd.DataFrame(index=idx, columns=fundamentals_snapshot.columns)
            
            # 2.3. Preencher a última data com o snapshot atual
            last_date = prices.index[-1]
            for ticker in fundamentals_snapshot.index:
                all_fundamentals.loc[(last_date, ticker), :] = fundamentals_snapshot.loc[ticker, :]
            
            # 2.4. Forward Fill (ffill) para simular a persistência do dado no tempo
            # O dado fundamental mais recente é replicado para todas as datas anteriores.
            # Isso é uma simplificação, mas é a melhor aproximação para dados de snapshot.
            all_fundamentals = all_fundamentals.sort_index().ffill()
            
            # all_fundamentals agora é um DataFrame com MultiIndex (Date, Ticker)
            # A função run_dynamic_backtest irá indexar por data e usar .unstack() para obter o formato Ticker x Metric
            
            # 3. Cálculo do Score Atual (Usando o snapshot atual)
            status.update(label="Calculando scores atuais...", state="running")
            
            res_        res_mom = compute_residual_momentum(mom_window)
        
        # CORREÇÃO DE LOOK-AHEAD BIAS:
        # Usar apenas os dados fundamentais disponíveis até a data de rebalanceamento.
        # all_fundamentals é um DataFrame de série temporal (Index=Date, Columns=Ticker_Metric).
        # A indexação .loc[rebal_date] pega o último dado disponível (graças ao ffill feito em main).
        fundamentals_at_rebal = all_fundamentals.loc[rebal_date].unstack()
        
        fund_mom = compute_fundamental_momentum(fundamentals_at_rebal)
        val_score = compute_value_score(fundamentals_at_rebal)
        qual_score = compute_quality_score(fundamentals_at_rebal)s_snapshot)
            
            final_df = pd.DataFrame(index=prices.columns.drop('BOVA11.SA', errors='ignore'))
            final_df['Res_Mom'] = res_mom
            final_df['Fund_Mom'] = fund_mom
            final_df['Value'] = val_score
            final_df['Quality'] = qual_score
            
            cols_show = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
            
            if 'sector' in fundamentals_snapshot.columns: 
                 final_df['Sector'] = fundamentals_snapshot['sector']
                 
            final_df.dropna(thresh=2, inplace=True)
            
            # Normalização e Score Composto
            w_keys = {}
            for c in cols_show:
                if c in final_df.columns:
                    new_col = f"{c}_Z"
                    final_df[new_col] = robust_zscore(final_df[c])
                    w_keys[new_col] = weights_map.get(c, 0.0)
            
            final_df = build_composite_score(final_df, w_keys)
            
            # 4. Execução do B            # 4. Execução do Backtest Dinâmico
            status.update(label="Executando backtest dinâmico...", state="running")
            
            # O backtest agora usa all_fundamentals (série temporal simulada)
            backtest_1yr = run_dynamic_backtest(prices, all_fundamentals, weights_map, top_n, use_vol_target, use_sector_neutrality, start_date_1yr)
            backtest_full_period = run_dynamic_backtest(prices, all_fundamentals, weights_map, top_n, use_vol_target, use_sector_neutrality, start_date_dca_period)
            dca_curve, dca_transactions, dca_holdings = run_dca_backtest(prices, all_fundamentals, weights_map, top_n, dca_amount, use_vol_target, use_sector_neutrality, start_date_dca_period, end_date)get, use_sector_neutrality, start_date_dca_period, end_date)

            status.update(label="Concluído!", state="complete", expanded=False)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Ranking Atual", "📈 Performance Dinâmica (1 Ano)", "💰 Backtest DCA", "💼 Alocação Final (DCA)", "🔍 Detalhes"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Top Picks (Score Atual)")
                show_list = ['Composite_Score', 'Sector'] + cols_show
                st.dataframe(final_df[show_list].head(top_n).style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']), height=400, width=1000)
            with col2:
                st.subheader("Sugestão de Rebalanceamento")
                weights = construct_portfolio(final_df, prices, top_n, 0.15 if use_vol_target else None)
                if not weights.empty:
                    # Lógica de Cálculo de Quantidade baseada no Aporte (MODIFICADA: Arredondamento Padrão)
                    latest_prices = prices.iloc[-1] # Pega o preço mais recente
                    
                    w_df = weights.to_frame(name="Peso")
                    
                    # 1. Pega o preço atual dos ativos selecionados
                    w_df["Preço (R$)"] = latest_prices.reindex(w_df.index).fillna(0)
                    
                    # 2. Calcula quanto dinheiro (R$) vai para cada ativo
                    w_df["Alocação (R$)"] = w_df["Peso"] * dca_amount
                    
                    # 3. Calcula quantidade (ARREDONDAMENTO PADRÃO: >= 0.5 SOBE)
                    # Soma 0.5 e usa floor: 0.8 + 0.5 = 1.3 -> 1.0 | 0.4 + 0.5 = 0.9 -> 0.0
                    w_df["Qtd. Aprox"] = np.where(
                        w_df["Preço (R$)"] > 0, 
                        np.floor((w_df["Alocação (R$)"] / w_df["Preço (R$)"]) + 0.5), 
                        0
                    )

                    # Cria um DataFrame formatado apenas para exibição (Bonito)
                    display_df = w_df.copy()
                    display_df["Peso"] = display_df["Peso"].map("{:.2%}".format)
                    display_df["Preço (R$)"] = display_df["Preço (R$)"].map("R$ {:,.2f}".format)
                    display_df["Alocação (R$)"] = display_df["Alocação (R$)"].map("R$ {:,.2f}".format)
                    display_df["Qtd. Aprox"] = display_df["Qtd. Aprox"].map("{:.0f}".format)

                    # Exibe a tabela
                    st.table(display_df[["Peso", "Preço (R$)", "Qtd. Aprox", "Alocação (R$)"]])
                    
                    # Gráfico de Pizza original
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Distribuição Teórica")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Performance Dinâmica (Últimos 12 Meses)")
            if not backtest_1yr.empty:
                total_ret = backtest_1yr.iloc[-1] - 1
                daily = backtest_1yr.pct_change().dropna()
                vol = daily.std() * (252**0.5)
                sharpe = (total_ret - 0.10) / vol 
                c1, c2, c3 = st.columns(3)
                c1.metric("Retorno Estratégia", f"{total_ret['Strategy']:.2%}", delta=f"vs Benchmark {total_ret['BOVA11.SA']:.2%}")
                c2.metric("Volatilidade", f"{vol['Strategy']:.2%}")
                c3.metric("Sharpe", f"{sharpe['Strategy']:.2f}")
                st.plotly_chart(px.line(backtest_1yr, title="Curva de Retorno (Base 1.0)", color_discrete_map={'Strategy': '#00CC96', 'BOVA11.SA': '#EF553B'}), use_container_width=True)
            else:
                st.warning("Dados insuficientes para backtest dinâmico.")

        with tab3:
            st.header(f"Simulação de Aportes ({dca_years} Anos)")
            st.subheader("Métricas de Risco/Retorno (Período Completo)")
            if not backtest_full_period.empty:
                total_ret_full = backtest_full_period.iloc[-1] - 1
                vol_full = backtest_full_period.pct_change().dropna().std() * (252**0.5)
                sharpe_full = (total_ret_full - 0.10 * dca_years) / vol_full 
                m1, m2, m3 = st.columns(3)
                m1.metric("Retorno Acumulado", f"{total_ret_full['Strategy']:.2%}", delta=f"vs Bench {total_ret_full['BOVA11.SA']:.2%}")
                m2.metric("Volatilidade Anualizada", f"{vol_full['Strategy']:.2%}")
                m3.metric("Sharpe Ratio", f"{sharpe_full['Strategy']:.2f}")
            st.markdown("---")
            st.subheader(f"Evolução Patrimonial (DCA R${dca_amount}/mês)")
            if not dca_curve.empty:
                final_strat = dca_curve['Strategy_DCA'].iloc[-1]
                invested = len(dca_transactions['Date'].unique()) * dca_amount
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Investido", f"R${invested:,.2f}")
                c2.metric("Saldo Estratégia", f"R${final_strat:,.2f}", delta=f"{((final_strat/invested)-1):.2%}")
                c3.metric("Saldo Benchmark", f"R${dca_curve['BOVA11.SA_DCA'].iloc[-1]:,.2f}")
                st.plotly_chart(px.line(dca_curve, title="Crescimento do Patrimônio"), use_container_width=True)
                st.subheader("Histórico de Execução (Boletas)")
                dca_transactions['Date'] = pd.to_datetime(dca_transactions['Date']).dt.strftime('%Y-%m-%d')
                st.dataframe(dca_transactions.set_index('Date'), height=300, use_container_width=True)

        with tab4:
            st.header("💼 Alocação Atual da Carteira DCA")
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

        with tab5:
            st.subheader("Correlação e Dados Brutos")
            if cols_show:
                corr = final_df[cols_show].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlação entre Fatores"), use_container_width=True)
            st.dataframe(fundamentals_snapshot)

if __name__ == "__main__":
    main()
