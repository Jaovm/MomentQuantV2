import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import requests
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro (FMP Data)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API KEY (Configurada no código conforme solicitado)
FMP_API_KEY = "rd6uBzkLLSPG68s9GcSx3folN76IxRhV"

# ==============================================================================
# MÓDULO 1: DATA FETCHING (PREÇOS & FMP FUNDAMENTALS)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados, garantindo o benchmark DIVO11.SA."""
    t_list = list(tickers)
    if 'DIVO11.SA' not in t_list:
        t_list.append('DIVO11.SA')
    
    try:
        # Download otimizado
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False
        )['Adj Close']
        
        # Tratamento para MultiIndex (caso yfinance retorne assim)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fmp_fundamentals_history(tickers: list, api_key: str) -> tuple:
    all_data = []
    sector_map = {}
    
    column_mapping = {
        'date': 'date', 'symbol': 'ticker', 'peRatio': 'forwardPE',
        'pbRatio': 'priceToBook', 'roe': 'returnOnEquity',
        'netProfitMargin': 'profitMargins', 'debtToEquity': 'debtToEquity',
        'enterpriseValueOverEBITDA': 'enterpriseToEbitda',
        'revenueGrowth': 'revenueGrowth', 'netIncomeGrowth': 'earningsGrowth'
    }

    session = requests.Session()
    my_bar = st.progress(0, text="Acessando API FMP...")
    
    for i, t in enumerate(tickers):
        if t == 'BOVA11.SA': continue
            
        # CORREÇÃO 1: Remover .SA para consulta na FMP (A maioria dos dados BR na FMP usa apenas o código ou código.SA, mas o tratamento garante compatibilidade)
        fmp_ticker = t # Tente manter .SA primeiro, se falhar o código abaixo trata
        
        try:
            # 1. Busca Métricas
            url_metrics = f"https://financialmodelingprep.com/api/v3/key-metrics/{fmp_ticker}?period=quarter&limit=60&apikey={api_key}"
            resp = session.get(url_metrics)
            data_metrics = resp.json()

            # Se a lista vier vazia com .SA, tenta sem o sufixo
            if not data_metrics and ".SA" in t:
                fmp_ticker = t.replace(".SA", "")
                url_metrics = f"https://financialmodelingprep.com/api/v3/key-metrics/{fmp_ticker}?period=quarter&limit=60&apikey={api_key}"
                resp = session.get(url_metrics)
                data_metrics = resp.json()

            # 2. Processar Profile (Setor)
            url_profile = f"https://financialmodelingprep.com/api/v3/profile/{fmp_ticker}?apikey={api_key}"
            data_prof = session.get(url_profile).json()
            sector_map[t] = data_prof[0].get('sector', 'Unknown') if data_prof else 'Unknown'

            # 3. Validar e concatenar
            if isinstance(data_metrics, list) and len(data_metrics) > 0:
                df = pd.DataFrame(data_metrics)
                cols_exist = [c for c in column_mapping.keys() if c in df.columns]
                df = df[cols_exist].rename(columns=column_mapping)
                df['ticker'] = t # Mantém o ticker original (com .SA) para dar match com o yfinance
                df['date'] = pd.to_datetime(df['date'])
                
                # Preencher colunas faltantes com NaN
                for target_col in column_mapping.values():
                    if target_col not in df.columns and target_col not in ['date', 'ticker']:
                        df[target_col] = np.nan
                all_data.append(df)
            else:
                st.warning(f"FMP não retornou dados para {t}")

        except Exception as e:
            st.error(f"Erro na API FMP para {t}: {e}")
            
        my_bar.progress((i + 1) / len(tickers))
        
    my_bar.empty()
    
    if not all_data:
        return pd.DataFrame(), pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True), pd.DataFrame(list(sector_map.items()), columns=['ticker', 'sector']).set_index('ticker')
    
def align_fundamentals_to_prices(fund_df, price_df):
    """
    Expande os dados trimestrais para diários (Point-in-Time) usando Forward Fill.
    Retorna MultiIndex DataFrame: Index(Date, Ticker).
    """
    if fund_df.empty: return pd.DataFrame()
    
    # 1. Pivot: Index=Date, Columns=Ticker
    # Precisamos pivotar cada métrica separadamente ou fazer um pivot complexo
    # Abordagem: Pivot table com multiplas valores
    
    metrics = [c for c in fund_df.columns if c not in ['date', 'ticker']]
    
    # Remove duplicatas de data/ticker (pega o mais recente se houver erro na API)
    fund_df = fund_df.sort_values('date').drop_duplicates(subset=['date', 'ticker'], keep='last')
    
    pivot_df = fund_df.pivot(index='date', columns='ticker', values=metrics)
    
    # 2. Reindexar para o calendário de preços e Forward Fill
    # ffill garante que usamos o dado do último balanço até sair o próximo
    aligned = pivot_df.reindex(price_df.index, method='ffill')
    
    # 3. Stack para voltar ao formato tabular longo (Date, Ticker)
    # Stack level 1 (Ticker)
    stacked = aligned.stack(level='ticker', future_stack=True)
    
    return stacked

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Calcula Residual Momentum (Alpha) vs DIVO11.SA."""
    df = price_df.copy()
    # Resample para mensal para cálculo clássico de momentum
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'DIVO11.SA' not in rets.columns: 
        # Tenta recuperar se o nome for diferente ou retorna vazio
        return pd.Series(dtype=float)
        
    market = rets['DIVO11.SA']
    scores = {}
    window = lookback + skip
    
    for ticker in rets.columns:
        if ticker == 'DIVO11.SA': continue
        
        y = rets[ticker].tail(window)
        x = market.tail(window)
        
        if len(y) < window: continue
            
        try:
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            resid = model.resid[:-skip] # Exclui o mês mais recente (skip)
            
            if np.std(resid) == 0 or len(resid) < 2:
                scores[ticker] = 0
            else:
                scores[ticker] = np.sum(resid) / np.std(resid)
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    """
    Score baseado em crescimento ou qualidade da tendência.
    No modo histórico FMP, fund_df é um snapshot de uma data específica.
    """
    # Se tiver colunas de crescimento explícito
    metrics = ['earningsGrowth', 'revenueGrowth']
    components = []
    
    for m in metrics:
        if m in fund_df.columns:
            s = fund_df[m].fillna(0)
            # Clip outlier extremes for growth
            s = s.clip(lower=-0.5, upper=1.0) 
            components.append(s)
            
    # Se não tiver crescimento, usa ROE alto como proxy de Momentum Fundamental
    if not components and 'returnOnEquity' in fund_df.columns:
        return fund_df['returnOnEquity'].fillna(0).rename("Fundamental_Momentum")
        
    if components:
        df_comp = pd.concat(components, axis=1)
        # Normaliza Z-score interno
        return df_comp.mean(axis=1).rename("Fundamental_Momentum")
    
    return pd.Series(0, index=fund_df.index, name="Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E e P/B."""
    scores = pd.DataFrame(index=fund_df.index)
    
    if 'forwardPE' in fund_df: 
        # Inverso do PE, tratando zeros e negativos
        pe = fund_df['forwardPE']
        scores['EP'] = np.where(pe > 0, 1/pe, 0)
        
    if 'priceToBook' in fund_df: 
        pb = fund_df['priceToBook']
        scores['BP'] = np.where(pb > 0, 1/pb, 0)
        
    return scores.mean(axis=1).fillna(0).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: 
        # Menor divida é melhor. Inverte sinal ou faz 1/(1+x)
        scores['DE_Inv'] = -1 * fund_df['debtToEquity']
        
    return scores.mean(axis=1).fillna(0).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto."""
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or np.isnan(mad): return series - median 
    z = (series - median) / (mad * 1.4826) 
    return z.clip(-3, 3).fillna(0)

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calcula score final ponderado."""
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
            
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: PORTFOLIO & BACKTEST (POINT-IN-TIME CORRIGIDO)
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio."""
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    # Valida se os ativos selecionados têm dados de preço recentes
    valid_selected = [s for s in selected if s in prices.columns]
    if not valid_selected: return pd.Series()

    if vol_target is not None:
        recent_rets = prices[valid_selected].pct_change().tail(63)
        vols = recent_rets.std() * (252**0.5)
        vols = vols.replace(0, 1e-6) # Evita div por zero
        raw_weights_inv = 1 / vols
        weights = raw_weights_inv / raw_weights_inv.sum() 
    else:
        weights = pd.Series(1.0/len(valid_selected), index=valid_selected)
        
    return weights.sort_values(ascending=False)

def run_dynamic_backtest(
    all_prices: pd.DataFrame, 
    aligned_fundamentals: pd.DataFrame, # MultiIndex (Date, Ticker)
    sector_map: pd.DataFrame,
    weights_config: dict, 
    top_n: int, 
    use_vol_target: bool,
    use_sector_neutrality: bool,
    start_date_backtest: datetime
):
    """Backtest Walk-Forward usando dados Point-in-Time."""
    end_date = all_prices.index[-1]
    
    # Datas de rebalanceamento (Mensal)
    subset_prices = all_prices.loc[start_date_backtest - timedelta(days=400):end_date]
    rebalance_dates = subset_prices.loc[start_date_backtest:end_date].resample('MS').first().index.tolist()
    
    if not rebalance_dates:
        return pd.DataFrame()

    strategy_daily_rets = []
    benchmark_daily_rets = []

    for i, rebal_date in enumerate(rebalance_dates):
        # 1. Obter dados até a data de rebalanceamento
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
        prices_historical = subset_prices.loc[:rebal_date]
        
        # Janelas
        mom_window = prices_historical.tail(400) 
        risk_window = prices_historical.tail(90)
        
        # 2. Obter Fundamentais Point-in-Time (Slice do MultiIndex)
        try:
            # Tenta pegar dados exatos do dia. Se for feriado no rebal_date, pega o anterior mais próx.
            # O .loc no MultiIndex Date retorna um DF com index=Ticker
            if rebal_date in aligned_fundamentals.index.get_level_values(0):
                fund_snapshot = aligned_fundamentals.loc[rebal_date]
            else:
                # Fallback para o dia anterior disponível no índice de fundamentais
                avail_dates = aligned_fundamentals.index.get_level_values(0).unique()
                closest = avail_dates[avail_dates <= rebal_date].max()
                fund_snapshot = aligned_fundamentals.loc[closest]
        except:
            # Se falhar totalmente, pula o mês
            continue
            
        # Filtro de tickers válidos (Preço e Fundamento)
        # Tickers que existem no preço E nos fundamentais
        valid_tickers = list(set(mom_window.columns) & set(fund_snapshot.index))
        if len(valid_tickers) < top_n: continue
        
        # Slice final
        fund_snapshot = fund_snapshot.loc[valid_tickers]
        mom_window_valid = mom_window[valid_tickers]
        
        # 3. Cálculo de Fatores
        res_mom = compute_residual_momentum(mom_window_valid)
        fund_mom = compute_fundamental_momentum(fund_snapshot)
        val_score = compute_value_score(fund_snapshot)
        qual_score = compute_quality_score(fund_snapshot)
        
        df_period = pd.DataFrame(index=valid_tickers)
        df_period['Res_Mom'] = res_mom
        df_period['Fund_Mom'] = fund_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        
        # Adiciona Setor (Map estático é ok, pois setor muda pouco, ou extrair do snapshot se disponível)
        if not sector_map.empty:
            df_period = df_period.join(sector_map)
            df_period['sector'] = df_period['sector'].fillna('Unknown')
        else:
            df_period['sector'] = 'Unknown'
        
        df_period.dropna(thresh=2, inplace=True)
        norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
        w_keys = {}
        
        # 4. Neutralidade e Ranking
        if use_sector_neutrality and 'sector' in df_period.columns and df_period['sector'].nunique() > 1:
            for c in norm_cols:
                if c in df_period.columns:
                    new_col = f"{c}_Z"
                    # Z-score dentro do setor
                    df_period[new_col] = df_period.groupby('sector')[c].transform(
                         lambda x: robust_zscore(x) if len(x) > 2 else x - x.median()
                    )
                    w_keys[new_col] = weights_config.get(c, 0.0)
        else:
            for c in norm_cols:
                if c in df_period.columns:
                    new_col = f"{c}_Z"
                    df_period[new_col] = robust_zscore(df_period[c])
                    w_keys[new_col] = weights_config.get(c, 0.0)
                    
        ranked_period = build_composite_score(df_period, w_keys)
        
        # 5. Construção de Pesos
        current_weights = construct_portfolio(ranked_period, risk_window, top_n, 0.15 if use_vol_target else None)
        
        # 6. Retorno do Período
        market_period = subset_prices.loc[rebal_date:next_date].iloc[1:] 
        period_pct = market_period.pct_change().dropna()
        if period_pct.empty: continue
            
        valid_w_tickers = [t for t in current_weights.index if t in period_pct.columns]
        if valid_w_tickers:
            strat_ret = period_pct[valid_w_tickers].dot(current_weights[valid_w_tickers])
        else:
            strat_ret = pd.Series(0.0, index=period_pct.index)
            
        if 'DIVO11.SA' in period_pct.columns:
            bench_ret = period_pct['DIVO11.SA']
        else:
            bench_ret = pd.Series(0.0, index=period_pct.index)
            
        strategy_daily_rets.append(strat_ret)
        benchmark_daily_rets.append(bench_ret)
        
    if strategy_daily_rets:
        full_strategy = pd.concat(strategy_daily_rets)
        full_benchmark = pd.concat(benchmark_daily_rets)
        # Remove duplicatas de index (dias de rebal)
        full_strategy = full_strategy[~full_strategy.index.duplicated(keep='first')]
        full_benchmark = full_benchmark[~full_benchmark.index.duplicated(keep='first')]
        
        cumulative = pd.DataFrame({
            'Strategy': (1 + full_strategy).cumprod(),
            'DIVO11.SA': (1 + full_benchmark).cumprod()
        })
        return cumulative.dropna()
    return pd.DataFrame()

def run_dca_backtest(
    all_prices: pd.DataFrame, 
    aligned_fundamentals: pd.DataFrame,
    sector_map: pd.DataFrame,
    factor_weights: dict, 
    top_n: int, 
    dca_amount: float, 
    use_vol_target: bool,
    use_sector_neutrality: bool, 
    start_date: datetime,
    end_date: datetime
):
    """Simula DCA usando dados Point-in-Time."""
    # Preenchimento básico de preços
    all_prices = all_prices.ffill() 
    
    # Datas de aporte (Mensal)
    dca_dates = all_prices.loc[start_date:end_date].resample('MS').first().index.tolist()
    
    if len(dca_dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    portfolio_value = pd.Series(0.0, index=all_prices.index)
    benchmark_value = pd.Series(0.0, index=all_prices.index)
    portfolio_holdings = {} 
    benchmark_holdings = {'DIVO11.SA': 0.0}
    monthly_transactions = []
    
    for i, rebal_date in enumerate(dca_dates):
        # Setup similar ao dynamic_backtest...
        prices_for_mom = all_prices.loc[:rebal_date].tail(400)
        risk_window = all_prices.loc[:rebal_date].tail(90)
        
        # PIT Fundamentals
        try:
            if rebal_date in aligned_fundamentals.index.get_level_values(0):
                fund_snapshot = aligned_fundamentals.loc[rebal_date]
            else:
                avail_dates = aligned_fundamentals.index.get_level_values(0).unique()
                closest = avail_dates[avail_dates <= rebal_date].max()
                fund_snapshot = aligned_fundamentals.loc[closest]
        except:
            continue

        valid_tickers = list(set(prices_for_mom.columns) & set(fund_snapshot.index))
        if len(valid_tickers) < top_n: continue

        fund_snapshot = fund_snapshot.loc[valid_tickers]
        prices_for_mom = prices_for_mom[valid_tickers]

        # Scores
        res_mom = compute_residual_momentum(prices_for_mom)
        fund_mom = compute_fundamental_momentum(fund_snapshot)
        val_score = compute_value_score(fund_snapshot)
        qual_score = compute_quality_score(fund_snapshot)

        df_master = pd.DataFrame(index=valid_tickers)
        df_master['Res_Mom'] = res_mom
        df_master['Fund_Mom'] = fund_mom
        df_master['Value'] = val_score
        df_master['Quality'] = qual_score
        
        if not sector_map.empty:
            df_master = df_master.join(sector_map)
            df_master['sector'] = df_master['sector'].fillna('Unknown')
        else:
            df_master['sector'] = 'Unknown'

        df_master.dropna(thresh=2, inplace=True)
        
        norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
        w_keys = {}
        
        # Scoring Logic
        if use_sector_neutrality and 'sector' in df_master.columns and df_master['sector'].nunique() > 1:
            for c in norm_cols:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = df_master.groupby('sector')[c].transform(
                        lambda x: robust_zscore(x) if len(x) > 1 else x - x.median() 
                    )
                    w_keys[new_col] = factor_weights.get(c, 0.0)
        else:
            for c in norm_cols:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    w_keys[new_col] = factor_weights.get(c, 0.0)

        final_df = build_composite_score(df_master, w_keys)
        current_weights = construct_portfolio(final_df, risk_window, top_n, 0.15 if use_vol_target else None)
        
        # --- EXECUÇÃO DO APORTE (DCA) ---
        try:
            # Pega o preço exato do dia (ou mais próximo)
            curr_prices_row = all_prices.loc[rebal_date]
        except KeyError:
            continue
            
        # 1. Compra Benchmark
        bova_price = curr_prices_row.get('DIVO11.SA', np.nan)
        if not np.isnan(bova_price) and bova_price > 0:
            q_bova = dca_amount / bova_price
            benchmark_holdings['DIVO11.SA'] += q_bova
        
        # 2. Compra Estratégia
        for ticker, weight in current_weights.items():
            price = curr_prices_row.get(ticker, np.nan)
            if not np.isnan(price) and price > 0 and weight > 0:
                amount = dca_amount * weight
                quantity = amount / price
                portfolio_holdings[ticker] = portfolio_holdings.get(ticker, 0.0) + quantity
                monthly_transactions.append({
                    'Date': rebal_date, 
                    'Ticker': ticker,
                    'Action': 'Buy',
                    'Quantity': quantity,
                    'Price': price,
                    'Value_R$': amount
                })
        
        # --- VALUATION DIÁRIO ATÉ O PRÓXIMO MÊS ---
        next_month = dca_dates[i+1] if i < len(dca_dates) - 1 else end_date
        val_range = all_prices.loc[rebal_date:next_month].index
        
        for d in val_range:
            # Valor Estratégia
            strat_val = 0.0
            day_prices = all_prices.loc[d]
            for t, q in portfolio_holdings.items():
                if t in day_prices and not np.isnan(day_prices[t]):
                    strat_val += day_prices[t] * q
            portfolio_value[d] = strat_val
            
            # Valor Bench
            if 'DIVO11.SA' in day_prices:
                benchmark_value[d] = day_prices['DIVO11.SA'] * benchmark_holdings['DIVO11.SA']
        
    # Limpeza final
    portfolio_value = portfolio_value[portfolio_value > 0].ffill()
    benchmark_value = benchmark_value[benchmark_value > 0].ffill()
    
    equity_curve = pd.DataFrame({'Strategy_DCA': portfolio_value, 'DIVO11.SA_DCA': benchmark_value}).dropna()
    final_holdings = {k: v for k, v in portfolio_holdings.items() if v > 0}
    
    return equity_curve, pd.DataFrame(monthly_transactions), final_holdings

# ==============================================================================
# APP PRINCIPAL (STREAMLIT UI)
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab Pro: FMP Integration")
    st.markdown("Otimização de carteira com dados fundamentais **Históricos (Point-in-Time)**.")

    st.sidebar.header("1. Universo e Dados (BOVESPA)")
    default_univ = "ITUB3.SA, TOTS3.SA, MDIA3.SA, TAEE3.SA, BBSE3.SA, WEGE3.SA, PSSA3.SA, EGIE3.SA, B3SA3.SA, VIVT3.SA, AGRO3.SA, PRIO3.SA, BBAS3.SA, BPAC11.SA, SBSP3.SA, SAPR4.SA, CMIG3.SA, UNIP6.SA, FRAS3.SA, VALE3.SA, PETR4.SA, LREN3.SA"
    ticker_input = st.sidebar.text_area("Tickers (Separados por vírgula)", default_univ, height=100)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores (Alpha)")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_fm = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.20)
    w_val = st.sidebar.slider("Value", 0.0, 1.0, 0.20)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Construção de Portfólio (Risco)")
    top_n = st.sidebar.number_input("Número de Ativos (Top N)", 1, 20, 5)
    use_vol_target = st.sidebar.checkbox("Usar Ponderação por Risco Inverso?", True)
    target_vol = st.sidebar.slider("Volatilidade Alvo (Ref)", 0.05, 0.30, 0.15) if use_vol_target else None
    
    st.sidebar.markdown("---")
    st.sidebar.header("4. Diversificação")
    use_sector_neutrality = st.sidebar.checkbox("Usar Neutralidade Setorial?", True)
    
    st.sidebar.header("5. Simulação Mensal (DCA)")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 5000, 1000)
    dca_years = st.sidebar.slider("Anos de Backtest DCA", 1, 5, 3)
    
    run_btn = st.sidebar.button("🚀 Rodar Análise Completa", type="primary")

    if run_btn:
        if not tickers:
            st.error("Insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant (FMP)...", expanded=True) as status:
            end_date = datetime.now()
            start_date_total = end_date - timedelta(days=365 * (dca_years + 2)) # Buffer para métricas
            
            # 1. Fetch Price
            st.write("📡 Baixando Preços...")
            prices = fetch_price_data(tickers, start_date_total, end_date)
            
            # 2. Fetch Fundamentals (FMP)
            st.write("📡 Baixando Fundamentos Históricos (FMP)...")
            raw_history, sector_df = fetch_fmp_fundamentals_history(tickers, FMP_API_KEY)
            
            # No main(), verifique se há interseção de tickers antes de rodar o backtest
            if prices.empty:
                st.error("Erro: yfinance não retornou preços. Verifique sua conexão ou tickers.")
                return
            
            if raw_history.empty:
                st.error("Erro: FMP não retornou fundamentos. Verifique se sua API Key é válida para o plano 'Starter' ou superior.")
                return
            
            # Verificação de tickers comuns
            common_tickers = set(prices.columns).intersection(set(raw_history['ticker'].unique()))
            if not common_tickers:
                st.error(f"Erro de Sincronia: Os tickers do Preço ({list(prices.columns)[:3]}...) não batem com os da FMP ({raw_history['ticker'].unique()[:3]}...).")
                return
            
            # 3. Align Fundamentals (Point-in-Time)
            st.write("⚙️ Processando Point-in-Time Data...")
            aligned_fundamentals = align_fundamentals_to_prices(raw_history, prices)
            
            # --- CÁLCULO DO SNAPSHOT ATUAL PARA RANKING ---
            # Pega o último dia disponível nos preços
            last_date = prices.index[-1]
            try:
                # Tenta pegar dados alinhados da última data
                if last_date in aligned_fundamentals.index.get_level_values(0):
                    current_fund_snapshot = aligned_fundamentals.loc[last_date]
                else:
                    current_fund_snapshot = aligned_fundamentals.xs(aligned_fundamentals.index.get_level_values(0)[-1], level=0)
            except:
                st.warning("Não foi possível alinhar dados para hoje. Usando último disponível.")
                current_fund_snapshot = raw_history.sort_values('date').drop_duplicates('ticker', keep='last').set_index('ticker')

            # Filtra universo para calculo atual
            valid_curr = list(set(current_fund_snapshot.index) & set(prices.columns))
            current_fund_snapshot = current_fund_snapshot.loc[valid_curr]
            prices_curr = prices[valid_curr]

            # Fatores Atuais
            res_mom_curr = compute_residual_momentum(prices_curr.tail(400))
            fund_mom_curr = compute_fundamental_momentum(current_fund_snapshot)
            val_score_curr = compute_value_score(current_fund_snapshot)
            qual_score_curr = compute_quality_score(current_fund_snapshot)

            df_now = pd.DataFrame(index=valid_curr)
            df_now['Res_Mom'] = res_mom_curr
            df_now['Fund_Mom'] = fund_mom_curr
            df_now['Value'] = val_score_curr
            df_now['Quality'] = qual_score_curr
            
            # Adiciona setor
            if not sector_df.empty:
                df_now = df_now.join(sector_df)
                df_now['sector'] = df_now['sector'].fillna('Unknown')
            else:
                df_now['sector'] = 'Unknown'
            
            df_now.dropna(thresh=2, inplace=True)
            
            # Normalização Atual
            w_map = {'Res_Mom': w_rm, 'Fund_Mom': w_fm, 'Value': w_val, 'Quality': w_qual}
            w_keys_curr = {}
            cols_show = []

            norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
            if use_sector_neutrality and 'sector' in df_now.columns and df_now['sector'].nunique() > 1:
                for c in norm_cols:
                    if c in df_now.columns:
                        nc = f"{c}_Z_Sector"
                        df_now[nc] = df_now.groupby('sector')[c].transform(lambda x: robust_zscore(x))
                        w_keys_curr[nc] = w_map.get(c, 0.0)
                        cols_show.append(nc)
            else:
                for c in norm_cols:
                    if c in df_now.columns:
                        nc = f"{c}_Z"
                        df_now[nc] = robust_zscore(df_now[c])
                        w_keys_curr[nc] = w_map.get(c, 0.0)
                        cols_show.append(nc)

            final_df_now = build_composite_score(df_now, w_keys_curr)
            weights_now = construct_portfolio(final_df_now, prices_curr, top_n, target_vol)

            # --- BACKTESTS ---
            st.write("🔄 Rodando Backtests Dinâmicos...")
            start_date_1yr = end_date - timedelta(days=365)
            start_date_dca = end_date - timedelta(days=365 * dca_years)

            # Backtest 1 Ano (Performance Pura)
            bt_1yr = run_dynamic_backtest(
                prices, aligned_fundamentals, sector_df, w_map, top_n, 
                use_vol_target, use_sector_neutrality, start_date_1yr
            )
            
            # Backtest DCA (Patrimônio)
            dca_equity, dca_trans, dca_hold = run_dca_backtest(
                prices, aligned_fundamentals, sector_df, w_map, top_n, 
                dca_amount, use_vol_target, use_sector_neutrality, start_date_dca, end_date
            )

            status.update(label="Concluído!", state="complete", expanded=False)

        # --- VISUALIZAÇÃO ---
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking Atual", "📈 Performance (1 Ano)", "💰 DCA Simulator", "💼 Custódia Final"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Top Picks (Dados Mais Recentes)")
                show_cols = ['Composite_Score', 'sector'] + cols_show
                st.dataframe(
                    final_df_now[show_cols].head(top_n).style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']), 
                    height=400, use_container_width=True
                )
            with col2:
                st.subheader("Pesos Sugeridos")
                if not weights_now.empty:
                    w_disp = weights_now.to_frame(name="Peso")
                    w_disp["Peso"] = w_disp["Peso"].map("{:.2%}".format)
                    st.table(w_disp)
                    fig_pie = px.pie(values=weights_now.values, names=weights_now.index, title="Alocação")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Backtest Walk-Forward (Últimos 12 Meses)")
            if not bt_1yr.empty:
                ret_strat = bt_1yr['Strategy'].iloc[-1] - 1
                ret_bench = bt_1yr['DIVO11.SA'].iloc[-1] - 1
                vol = bt_1yr.pct_change().std() * (252**0.5)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Retorno Estratégia", f"{ret_strat:.2%}", delta=f"vs Bench {ret_bench:.2%}")
                c2.metric("Volatilidade", f"{vol['Strategy']:.2%}")
                c3.metric("Sharpe", f"{(ret_strat)/vol['Strategy']:.2f}")
                
                st.plotly_chart(px.line(bt_1yr, title="Curva de Retorno Comparativa", 
                                       color_discrete_map={'Strategy': '#00CC96', 'DIVO11.SA': '#EF553B'}), 
                                use_container_width=True)
            else:
                st.warning("Dados insuficientes para gerar backtest deste período.")

        with tab3:
            st.subheader(f"Simulação de Aportes Mensais ({dca_years} Anos)")
            if not dca_equity.empty:
                final_val = dca_equity['Strategy_DCA'].iloc[-1]
                invested = len(dca_trans['Date'].unique()) * dca_amount
                
                m1, m2 = st.columns(2)
                m1.metric("Total Investido", f"R$ {invested:,.2f}")
                m2.metric("Saldo Final", f"R$ {final_val:,.2f}", delta=f"Rentab: {((final_val/invested)-1):.2%}")
                
                st.plotly_chart(px.line(dca_equity, title="Evolução Patrimonial (R$)"), use_container_width=True)
                
                with st.expander("Ver Histórico de Transações"):
                    dca_trans['Date'] = pd.to_datetime(dca_trans['Date']).dt.date
                    st.dataframe(dca_trans, use_container_width=True)
            else:
                st.warning("Não foi possível gerar simulação DCA.")

        with tab4:
            st.subheader("Alocação Final da Carteira DCA")
            if dca_hold:
                # Calcula valor atual
                cur_p = prices.iloc[-1]
                alloc_list = []
                for t, q in dca_hold.items():
                    if t in cur_p:
                        val = q * cur_p[t]
                        alloc_list.append({'Ticker': t, 'Qtd': q, 'Valor': val})
                
                df_alloc = pd.DataFrame(alloc_list).sort_values('Valor', ascending=False)
                if not df_alloc.empty:
                    total = df_alloc['Valor'].sum()
                    df_alloc['%'] = df_alloc['Valor'] / total
                    
                    c_chart, c_tbl = st.columns(2)
                    with c_chart:
                        st.plotly_chart(px.pie(df_alloc, values='Valor', names='Ticker', hole=0.4), use_container_width=True)
                    with c_tbl:
                        df_alloc['Valor'] = df_alloc['Valor'].map('R$ {:,.2f}'.format)
                        df_alloc['%'] = df_alloc['%'].map('{:.2%}'.format)
                        st.dataframe(df_alloc, use_container_width=True)
            else:
                st.info("Carteira vazia.")

if __name__ == "__main__":
    main()


