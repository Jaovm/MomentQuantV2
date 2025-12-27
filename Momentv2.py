import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

# ==============================================================================
# CONFIGURAÃ‡ÃƒO DA PÃGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v3 (Brapi Edition)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
BRAPI_TOKEN = "5gVedSQ928pxhFuTvBFPfr"

# ==============================================================================
# MÃ“DULO 1: DATA FETCHING (PREÃ‡OS & BRAPI)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histÃ³rico de preÃ§os ajustados via YFinance (melhor para OHLCV histÃ³rico)."""
    t_list = list(tickers)
    # Garante benchmarks
    for bench in ['BOVA11.SA', 'DIVO11.SA']:
        if bench not in t_list:
            t_list.append(bench)
    
    try:
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False, # Ajustaremos manualmente se necessÃ¡rio, mas 'Adj Close' costuma ser melhor
            threads=True
        )['Adj Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = data.dropna(axis=1, how='all')
        return data
    except Exception as e:
        st.error(f"Erro crÃ­tico ao baixar preÃ§os: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*6)
def fetch_fundamentals_brapi(tickers: list, token: str) -> pd.DataFrame:
    """
    Busca fundamentos ATUAIS via Brapi.dev para o Ranking de SeleÃ§Ã£o.
    Agrupa requisiÃ§Ãµes para economizar chamadas.
    """
    clean_tickers = [t for t in tickers if t not in ['BOVA11.SA', 'DIVO11.SA']]
    if not clean_tickers:
        return pd.DataFrame()

    # Brapi suporta virgula: PETR4,VALE3 (limite de url, vamos fazer chunks de 10)
    chunk_size = 15
    chunks = [clean_tickers[i:i + chunk_size] for i in range(0, len(clean_tickers), chunk_size)]
    
    fundamental_data = []
    failed_tickers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Consultando Brapi API: Lote {i+1}/{len(chunks)}")
        ticker_str = ','.join(chunk)
        url = f"https://brapi.dev/api/quote/{ticker_str}"
        params = {
            'token': token,
            'fundamental': 'true', # Solicita dados fundamentais
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    try:
                        # ExtraÃ§Ã£o segura de dados
                        symbol = item.get('symbol')
                        price = item.get('regularMarketPrice', np.nan)
                        
                        # Setor (Brapi nem sempre retorna setor explicitamente na rota quote simples, 
                        # mas vamos tentar pegar do logName ou usar placeholder)
                        sector = item.get('sector', 'Unknown')
                        if sector == 'Unknown' or not sector:
                            # Tenta inferir ou deixa Unknown
                            sector = 'General'

                        # ExtraÃ§Ã£o de MÃ©tricas de Valor e Qualidade
                        # A estrutura da Brapi pode variar, usamos get com default
                        market_cap = item.get('marketCap', np.nan)
                        
                        # As vezes as chaves vÃªm diretas, as vezes dentro de summaryProfile ou financialData
                        # Brapi padrÃ£o 'quote':
                        pe_ratio = item.get('priceEarnings', np.nan) # P/L
                        
                        # Tenta buscar mÃ©tricas adicionais se disponÃ­veis no payload
                        # Nota: A rota gratuita/simples pode nÃ£o ter todos os campos complexos do YF.
                        # Vamos mapear o que Ã© comum.
                        
                        # Value
                        p_vp = item.get('priceToBook', np.nan) # P/VP
                        ev_ebitda = item.get('enterpriseValueToEBITDA', np.nan)
                        
                        # Quality
                        roe = item.get('returnOnEquity', np.nan)
                        net_margin = item.get('profitMargin', np.nan)
                        
                        # Se ROE vier nulo, tentar calcular se tiver earnings
                        # Dados para Value Composite
                        
                        fundamental_data.append({
                            'ticker': symbol,
                            'sector': sector,
                            'currentPrice': price,
                            'marketCap': market_cap,
                            # Value
                            'PE': pe_ratio,
                            'P_VP': p_vp,
                            'EV_EBITDA': ev_ebitda,
                            # Quality
                            'ROE': roe,
                            'Net_Margin': net_margin,
                        })
                    except Exception as e_inner:
                        print(f"Erro ao parsear {item.get('symbol')}: {e_inner}")
            else:
                print(f"Erro na requisiÃ§Ã£o Brapi: {response.status_code}")
                
        except Exception as e:
            st.warning(f"Erro de conexÃ£o com Brapi: {e}")
        
        progress_bar.progress((i + 1) / len(chunks))
        time.sleep(0.5) # Respeita rate limit gentilmente

    progress_bar.empty()
    status_text.empty()
    
    if not fundamental_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(fundamental_data).set_index('ticker')
    
    # Limpeza bÃ¡sica de dados zerados ou irreais
    cols_to_clean = ['PE', 'P_VP', 'EV_EBITDA', 'ROE', 'Net_Margin']
    for col in cols_to_clean:
        if col in df.columns:
            # Substitui 0 exato por NaN para nÃ£o distorcer mÃ©dias, exceto onde 0 faz sentido
            df[col] = df[col].replace(0, np.nan)
            
    return df

# ==============================================================================
# MÃ“DULO 2: CÃLCULO DE FATORES
# ==============================================================================

def compute_residual_momentum_enhanced(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """
    Residual Momentum (Blitz) com Volatility Scaling.
    Janela padrÃ£o de 12 meses (252 dias aprox), pulando o Ãºltimo mÃªs (21 dias).
    """
    # AproximaÃ§Ã£o mensal para otimizaÃ§Ã£o de performance
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    
    # 36 meses de lookback para regressÃ£o (3 anos), momentum formation de 12m
    regression_window = 36 
    
    for ticker in rets.columns:
        if ticker in ['BOVA11.SA', 'DIVO11.SA']: continue
        
        # Pega dados suficientes para a regressÃ£o
        y_full = rets[ticker].tail(regression_window + skip)
        x_full = market.tail(regression_window + skip)
        
        if len(y_full) < 12: continue
            
        try:
            # Alinha Ã­ndices
            common_idx = y_full.index.intersection(x_full.index)
            y_full = y_full.loc[common_idx]
            x_full = x_full.loc[common_idx]

            X = sm.add_constant(x_full.values)
            model = sm.OLS(y_full.values, X).fit()
            residuals = pd.Series(model.resid, index=y_full.index)
            
            # Momentum Residual: Soma dos resÃ­duos dos Ãºltimos 12 meses (excluindo mÃªs atual)
            # Janela de formaÃ§Ã£o: t-12 a t-1
            resid_12m = residuals.iloc[-(12 + skip) : -skip]
            
            if len(resid_12m) == 0:
                scores[ticker] = 0
                continue

            raw_momentum = resid_12m.sum()
            resid_vol = residuals.std()
            
            # Volatility Scaling (Score = Mom / Vol do erro)
            if resid_vol == 0:
                scores[ticker] = 0
            else:
                scores[ticker] = raw_momentum / resid_vol 
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_value_robust(fund_df: pd.DataFrame) -> pd.Series:
    """Composite Value Score usando dados da Brapi."""
    scores = pd.DataFrame(index=fund_df.index)
    
    def invert_metric(series):
        # Earning Yield (1/PE), Book Yield (1/PB)
        # Se P/L < 0, yield Ã© ruim. Se P/L > 0, menor Ã© melhor (yield maior).
        # Tratamento: 1/PE. 
        # Cuidado com P/L negativo: empresa com prejuÃ­zo. Yield negativo.
        return 1.0 / series.replace(0, np.nan)

    if 'PE' in fund_df: scores['Earnings_Yield'] = invert_metric(fund_df['PE'])
    if 'P_VP' in fund_df: scores['Book_Yield'] = invert_metric(fund_df['P_VP'])
    if 'EV_EBITDA' in fund_df: scores['EBITDA_Yield'] = invert_metric(fund_df['EV_EBITDA'])

    # Z-Score por mÃ©trica e mÃ©dia
    for col in scores.columns:
        # Preenche NaNs com a mediana (neutro) antes de normalizar
        filled = scores[col].fillna(scores[col].median())
        if filled.std() > 0:
            scores[col] = (filled - filled.mean()) / filled.std()
        else:
            scores[col] = 0

    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Composite Quality Score."""
    scores = pd.DataFrame(index=fund_df.index)
    
    # Quanto maior, melhor
    if 'ROE' in fund_df: scores['ROE'] = fund_df['ROE']
    if 'Net_Margin' in fund_df: scores['Margin'] = fund_df['Net_Margin']
    
    # Z-Score
    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        if filled.std() > 0:
            scores[col] = (filled - filled.mean()) / filled.std()
        else:
            scores[col] = 0
            
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÃ“DULO 3: MATEMÃTICA E MÃ‰TRICAS
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: return series - median 
    z = (series - median) / (mad * 1.4826) 
    return z.clip(-3, 3) 

def calculate_advanced_metrics(prices_series: pd.Series, risk_free_rate_annual: float = 0.10):
    if prices_series.empty or len(prices_series) < 2:
        return {}
    
    daily_rets = prices_series.pct_change().dropna()
    total_ret = (prices_series.iloc[-1] / prices_series.iloc[0]) - 1
    days = (prices_series.index[-1] - prices_series.index[0]).days
    cagr = (1 + total_ret)**(365/days) - 1 if days > 0 else 0
    vol_ann = daily_rets.std() * np.sqrt(252)
    
    rf_daily = (1 + risk_free_rate_annual)**(1/252) - 1
    excess_rets = daily_rets - rf_daily
    sharpe = (excess_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    
    downside_rets = excess_rets[excess_rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (excess_rets.mean() * 252) / downside_std if downside_std > 0 else 0
    
    cum_rets = (1 + daily_rets).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
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
    sma200 = prices.rolling(window=200).mean().iloc[-1]
    curr_price = prices.iloc[-1]
    valid_tickers = [t for t in prices.columns if t != benchmark]
    if not valid_tickers: return 0, 0
    above_sma = (curr_price[valid_tickers] > sma200[valid_tickers]).sum()
    total = len(valid_tickers)
    return above_sma / total if total > 0 else 0, total

# ==============================================================================
# MÃ“DULO 4: SIMULAÃ‡ÃƒO MONTE CARLO
# ==============================================================================

def run_monte_carlo(initial_balance, monthly_contrib, mu_annual, sigma_annual, years, simulations=1000):
    months = years * 12
    dt = 1/12
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    shock = sigma_annual * np.sqrt(dt) * np.random.normal(0, 1, (months, simulations))
    monthly_returns = np.exp(drift + shock) - 1
    
    portfolio_paths = np.zeros((months + 1, simulations))
    portfolio_paths[0] = initial_balance
    
    for t in range(1, months + 1):
        portfolio_paths[t] = portfolio_paths[t-1] * (1 + monthly_returns[t-1]) + monthly_contrib
        
    percentiles = np.percentile(portfolio_paths, [5, 50, 95], axis=1)
    dates = [datetime.now() + timedelta(days=30*i) for i in range(months + 1)]
    
    return pd.DataFrame({
        'Pessimista (5%)': percentiles[0],
        'Base (50%)': percentiles[1],
        'Otimista (95%)': percentiles[2]
    }, index=dates)

# ==============================================================================
# MÃ“DULO 5: BACKTEST & ENGINE (COM CORREÃ‡ÃƒO DE VIÃ‰S)
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: bool = False):
    """ConstrÃ³i pesos. Se vol_target=True, usa Inverse Volatility (Risk Parity)."""
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    if vol_target:
        # Volatilidade de 3 meses (63 dias)
        recent_rets = prices[selected].pct_change().tail(63) 
        vols = recent_rets.std() * (252**0.5)
        # Evita divisÃ£o por zero
        vols = vols.replace(0, 1e-6)
        raw_weights_inv = 1 / vols
        weights = raw_weights_inv / raw_weights_inv.sum() 
    else:
        weights = pd.Series(1.0/len(selected), index=selected)
    return weights.sort_values(ascending=False)

def run_dca_backtest_robust(
    all_prices: pd.DataFrame,
    # all_fundamentals NÃƒO Ã© usado no backtest histÃ³rico para evitar lookahead bias, 
    # a menos que tivÃ©ssemos histÃ³rico. Usaremos apenas Momentum e Vol.
    top_n: int,
    dca_amount: float,
    use_vol_target: bool,
    start_date: datetime,
    end_date: datetime
):
    """
    Simula aportes mensais constantes.
    **MODO ROBUSTO**: Utiliza apenas Price-Action (Momentum e Volatilidade) para decisÃ£o histÃ³rica
    para evitar Lookahead Bias (usar P/L de 2024 em 2020).
    """
    all_prices = all_prices.ffill()
    dca_start = start_date + timedelta(days=30)
    
    # --- CORREÃ‡ÃƒO AQUI ---
    # O cÃ³digo anterior usava .resample('MS').first().index, o que gerava datas como '2018-04-01' (Domingo).
    # Agora usamos .apply() para pegar o primeiro Ã­ndice VÃLIDO (dia Ãºtil) de cada mÃªs.
    dates_series = all_prices.loc[dca_start:end_date].resample('MS').apply(lambda x: x.index[0] if not x.empty else None)
    dates = dates_series.dropna().tolist()
    # ---------------------

    if not dates or len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    portfolio_value = pd.Series(0.0, index=all_prices.index)
    portfolio_holdings = {} # {ticker: qtd}
    monthly_transactions = []
    cash = 0.0 # Caixa residual

    for i, month_start in enumerate(dates):
        # 1. DefiniÃ§Ã£o das janelas de dados (InformaÃ§Ã£o disponÃ­vel no momento T)
        eval_date = month_start - timedelta(days=1)
        mom_start = month_start - timedelta(days=365*3) # 3 anos para momentum robusto
        prices_historical = all_prices.loc[mom_start:eval_date]
        
        # 2. Screening (Apenas Momentum Residual neste modo robusto)
        res_mom = compute_residual_momentum_enhanced(prices_historical, lookback=12, skip=1)
        
        if res_mom.empty:
            continue
            
        # Ranking apenas por Momentum Z-Score
        df_rank = pd.DataFrame(index=res_mom.index)
        df_rank['Score'] = robust_zscore(res_mom)
        df_rank = df_rank.sort_values('Score', ascending=False)
        
        # 3. DefiniÃ§Ã£o de Pesos (Risk Parity)
        risk_window = prices_historical.tail(90)
        target_weights = construct_portfolio(df_rank, risk_window, top_n, use_vol_target)
        
        # 4. ExecuÃ§Ã£o de Aportes e Rebalanceamento
        # Tenta pegar preÃ§os do dia exato (agora garantido pela correÃ§Ã£o das datas)
        try:
            current_date_prices = all_prices.loc[month_start]
        except KeyError:
            # Fallback de seguranÃ§a: se mesmo assim falhar, pega o dia seguinte mais prÃ³ximo
            # Isso protege contra feriados locais nÃ£o padronizados
            next_idx = all_prices.index[all_prices.index > month_start]
            if next_idx.empty:
                break
            current_date_prices = all_prices.loc[next_idx[0]]
            month_start = next_idx[0] # Atualiza data da transaÃ§Ã£o

        
        total_portfolio_val = cash + dca_amount # ComeÃ§a com caixa + novo aporte
        
        # Liquida posiÃ§Ãµes antigas (valor teÃ³rico)
        for t, qtd in portfolio_holdings.items():
            if t in current_date_prices:
                price = current_date_prices[t]
                if not np.isnan(price):
                    total_portfolio_val += qtd * price
        
        # Nova distribuiÃ§Ã£o
        new_holdings = {}
        
        for ticker, weight in target_weights.items():
            if ticker in current_date_prices and not np.isnan(current_date_prices[ticker]):
                price = current_date_prices[ticker]
                if price > 0:
                    alloc_val = total_portfolio_val * weight
                    qty = alloc_val / price
                    new_holdings[ticker] = qty
                    
                    monthly_transactions.append({
                        'Date': month_start,
                        'Ticker': ticker,
                        'Action': 'Rebalance/Buy',
                        'Price': price,
                        'Weight': weight
                    })
        
        portfolio_holdings = new_holdings
        
        # 5. MarcaÃ§Ã£o a Mercado atÃ© o prÃ³ximo mÃªs
        next_month = dates[i+1] if i < len(dates)-1 else end_date
        valuation_dates = all_prices.loc[month_start:next_month].index
        
        for d in valuation_dates:
            val = 0
            for t, q in portfolio_holdings.items():
                p = all_prices.at[d, t]
                if not np.isnan(p):
                    val += q * p
            portfolio_value[d] = val

    # Limpeza e FormataÃ§Ã£o
    portfolio_value = portfolio_value[portfolio_value > 0].sort_index()
    equity_curve = pd.DataFrame({'Strategy_DCA': portfolio_value})
    
    transactions_df = pd.DataFrame(monthly_transactions)
    final_holdings = portfolio_holdings 

    return equity_curve, transactions_df, final_holdings


# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("ðŸ§ª Quant Factor Lab: Pro v3 (Brapi Edition)")
    st.markdown("""
    **OtimizaÃ§Ã£o Multifator Institucional**
    * **Ranking Atual:** Dados da API Brapi.dev (Value, Quality, Momentum).
    * **Backtest Robusto:** SimulaÃ§Ã£o histÃ³rica baseada puramente em Price-Action (Momentum + Volatilidade) para eliminar ViÃ©s de AntecipaÃ§Ã£o (Lookahead Bias), dado que APIs comuns nÃ£o fornecem balanÃ§os histÃ³ricos point-in-time.
    """)

    st.sidebar.header("1. Universo e Dados")
    default_univ = "ITUB4, VALE3, WEGE3, PRIO3, BBAS3, PETR4, RENT3, B3SA3, EQTL3, LREN3, RADL3, RAIL3, SUZB3, JBSS3, VIVT3, CMIG4, ELET3, BBSE3, GOAU4, TOTS3, MDIA3"
    ticker_input = st.sidebar.text_area("Tickers (Brapi Format - Sem .SA)", default_univ, height=100)
    # Adiciona sufixo .SA para YFinance (PreÃ§os), remove para Brapi
    raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    yf_tickers = [f"{t}.SA" for t in raw_tickers]
    
    st.sidebar.header("2. Pesos (Ranking Atual)")
    st.sidebar.caption("Define a importÃ¢ncia no Score de SeleÃ§Ã£o de HOJE.")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_val = st.sidebar.slider("Value (P/L, P/VP, EBITDA)", 0.0, 1.0, 0.40)
    w_qual = st.sidebar.slider("Quality (ROE, Margem)", 0.0, 1.0, 0.20)

    st.sidebar.header("3. ParÃ¢metros de GestÃ£o")
    top_n = st.sidebar.number_input("NÃºmero de Ativos", 4, 30, 10)
    use_vol_target = st.sidebar.checkbox("Risk Parity (Inv Vol)", True, help="Atribui menos peso a ativos mais volÃ¡teis.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("4. Backtest & Monte Carlo")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 100000, 2000)
    dca_years = st.sidebar.slider("Anos de HistÃ³rico", 2, 10, 5)
    mc_years = st.sidebar.slider("ProjeÃ§Ã£o Futura (Anos)", 1, 20, 5)
    
    run_btn = st.sidebar.button("ðŸš€ Executar AnÃ¡lise Institucional", type="primary")

    if run_btn:
        if not raw_tickers:
            st.error("Insira pelo menos um ticker.")
            return

        with st.status("Processando Pipeline Quantitativo...", expanded=True) as status:
            end_date = datetime.now()
            start_date_total = end_date - timedelta(days=365 * (dca_years + 3)) # +3 anos para warm-up do momentum
            start_date_backtest = end_date - timedelta(days=365 * dca_years)

            # 1. Dados de PreÃ§o (YFinance - Melhor histÃ³rico)
            status.write("ðŸ“¥ Baixando HistÃ³rico de PreÃ§os (YFinance)...")
            prices = fetch_price_data(yf_tickers, start_date_total, end_date)
            
            if prices.empty:
                st.error("Falha ao baixar preÃ§os.")
                status.update(label="Erro", state="error")
                return

            # 2. Dados Fundamentais ATUAIS (Brapi - Melhor snapshot)
            status.write("ðŸ” Consultando Fundamentos Atuais (Brapi.dev)...")
            fundamentals = fetch_fundamentals_brapi(raw_tickers, BRAPI_TOKEN)
            
            # Adiciona sufixo .SA no index do fundamentals para dar match com prices do YF
            if not fundamentals.empty:
                fundamentals.index = [f"{t}.SA" for t in fundamentals.index]

            # 3. CÃ¡lculo do RANKING ATUAL (Multifator Completo)
            status.write("ðŸ§® Calculando Scores Atuais...")
            # Momentum recente
            curr_mom = compute_residual_momentum_enhanced(prices)
            
            # Scores Fundamentais (apenas se tiver dados)
            if not fundamentals.empty:
                curr_val = compute_value_robust(fundamentals)
                curr_qual = compute_quality_score(fundamentals)
            else:
                curr_val = pd.Series(0, index=prices.columns)
                curr_qual = pd.Series(0, index=prices.columns)

            # ConsolidaÃ§Ã£o do Ranking
            df_master = pd.DataFrame(index=prices.columns)
            df_master['Res_Mom'] = curr_mom
            df_master['Value'] = curr_val
            df_master['Quality'] = curr_qual
            
            if not fundamentals.empty and 'sector' in fundamentals.columns:
                df_master['Sector'] = fundamentals['sector']
                
            df_master.dropna(thresh=1, inplace=True) # MantÃ©m se tiver pelo menos 1 fator

            # Z-Scores e Pesos Finais
            cols_map = {'Res_Mom': w_rm, 'Value': w_val, 'Quality': w_qual}
            df_master['Composite_Score'] = 0.0
            
            for col, weight in cols_map.items():
                if col in df_master.columns:
                    # Robust Z-Score
                    z = robust_zscore(df_master[col])
                    df_master[f'{col}_Z'] = z
                    df_master['Composite_Score'] += z * weight
            
            df_master = df_master.sort_values('Composite_Score', ascending=False)

            # 4. ExecuÃ§Ã£o do BACKTEST (Modo Robusto: Momentum Only)
            status.write("âš™ï¸ Rodando Backtest Robusto (Sem Lookahead Bias)...")
            dca_curve, dca_transactions, dca_holdings = run_dca_backtest_robust(
                prices,
                top_n,
                dca_amount,
                use_vol_target,
                start_date_backtest,
                end_date
            )

            status.update(label="AnÃ¡lise ConcluÃ­da!", state="complete", expanded=False)

        # ==============================================================================
        # DASHBOARD DE RESULTADOS
        # ==============================================================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "ðŸ† Ranking Atual (Multifator)", 
            "ðŸ“ˆ Performance DCA", 
            "ðŸ’° HistÃ³rico de Aportes", 
            "ðŸ”® Monte Carlo", 
            "ðŸ“‹ Dados Brutos"
        ])

        # --- TAB 1: RANKING ATUAL (O QUE COMPRAR HOJE) ---
        with tab1:
            st.subheader("ðŸŽ¯ Carteira Recomendada (Baseada em Dados Hoje)")
            st.caption("Fatores: Momentum (PreÃ§o) + Valor (Brapi) + Qualidade (Brapi)")
            
            top_picks = df_master.head(top_n).copy()
            
            # Traz dados para exibiÃ§Ã£o
            latest_prices = prices.iloc[-1]
            top_picks['PreÃ§o Atual'] = latest_prices.reindex(top_picks.index)
            
            # Calcula Pesos Sugeridos (Risk Parity ou Equal)
            risk_window = prices.tail(90)
            sug_weights = construct_portfolio(top_picks, risk_window, top_n, use_vol_target)
            
            top_picks['Peso (%)'] = (sug_weights * 100)
            top_picks['AlocaÃ§Ã£o (R$)'] = (sug_weights * dca_amount)
            top_picks['Qtd Sugerida'] = (top_picks['AlocaÃ§Ã£o (R$)'] / top_picks['PreÃ§o Atual'])
            
            # Display Bonito
            cols_show = ['Sector', 'PreÃ§o Atual', 'Composite_Score', 'Peso (%)', 'AlocaÃ§Ã£o (R$)', 'Qtd Sugerida']
            # Garante que colunas existem
            cols_final = [c for c in cols_show if c in top_picks.columns]
            
            display_df = top_picks[cols_final].style.format({
                'PreÃ§o Atual': 'R$ {:.2f}',
                'Composite_Score': '{:.2f}',
                'Peso (%)': '{:.1f}%',
                'AlocaÃ§Ã£o (R$)': 'R$ {:.0f}',
                'Qtd Sugerida': '{:.0f}'
            }).background_gradient(subset=['Composite_Score'], cmap='Greens')
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.plotly_chart(px.pie(values=sug_weights, names=sug_weights.index, title="AlocaÃ§Ã£o Sugerida"), use_container_width=True)
            with col_chart2:
                if 'Sector' in top_picks.columns:
                    st.plotly_chart(px.pie(top_picks, names='Sector', values='Peso (%)', title="ExposiÃ§Ã£o Setorial"), use_container_width=True)

        # --- TAB 2: PERFORMANCE BACKTEST (DCA) ---
        with tab2:
            st.subheader("SimulaÃ§Ã£o de AcumulaÃ§Ã£o de Capital (DCA)")
            st.warning("âš ï¸ Nota MetodolÃ³gica: O Backtest utiliza apenas **Momentum e Volatilidade** para selecionar ativos no passado. Fatores de Valor/Qualidade sÃ£o usados apenas no Ranking Atual para evitar viÃ©s de antecipaÃ§Ã£o (Lookahead Bias).")
            
            if not dca_curve.empty:
                # MÃ©tricas
                start_val = dca_curve.iloc[0,0]
                end_val = dca_curve.iloc[-1,0]
                total_invested = len(dca_transactions) * dca_amount if not dca_transactions.empty else 0 # Aprox
                # CorreÃ§Ã£o cÃ¡lculo total investido exato baseado nos meses Ãºnicos
                unique_months = pd.to_datetime(dca_transactions['Date']).dt.to_period('M').nunique()
                total_invested_real = unique_months * dca_amount
                
                profit = end_val - total_invested_real
                roi = (profit / total_invested_real) if total_invested_real > 0 else 0
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("PatrimÃ´nio Final", f"R$ {end_val:,.2f}")
                m2.metric("Total Investido", f"R$ {total_invested_real:,.2f}")
                m3.metric("Lucro LÃ­quido", f"R$ {profit:,.2f}", delta=f"{roi:.1%}")
                
                # GrÃ¡fico
                fig = px.line(dca_curve, title="Curva de PatrimÃ´nio (Aportes Mensais)")
                st.plotly_chart(fig, use_container_width=True)
                
                # MÃ©tricas AvanÃ§adas
                st.markdown("### AnÃ¡lise de Risco")
                metrics = calculate_advanced_metrics(dca_curve['Strategy_DCA'])
                st.json(metrics)

        # --- TAB 3: HISTÃ“RICO APORTES ---
        with tab3:
            st.subheader("DiÃ¡rio de TransaÃ§Ãµes Simulado")
            if not dca_transactions.empty:
                df_trans = pd.DataFrame(dca_transactions)
                df_trans['Date'] = pd.to_datetime(df_trans['Date']).dt.date
                st.dataframe(df_trans.sort_values('Date', ascending=False), use_container_width=True)

        # --- TAB 4: MONTE CARLO ---
        with tab4:
            st.subheader("ProjeÃ§Ã£o ProbabilÃ­stica")
            if not dca_curve.empty:
                # Pega stats do backtest
                daily_rets = dca_curve['Strategy_DCA'].pct_change().dropna()
                mu = daily_rets.mean() * 252
                sigma = daily_rets.std() * np.sqrt(252)
                
                st.write(f"ParÃ¢metros estimados do Backtest: Retorno Anual ~{mu:.1%} | Volatilidade ~{sigma:.1%}")
                
                sim_df = run_monte_carlo(
                    initial_balance=dca_curve.iloc[-1,0], # ComeÃ§a de onde parou
                    monthly_contrib=dca_amount,
                    mu_annual=mu,
                    sigma_annual=sigma,
                    years=mc_years
                )
                
                last_row = sim_df.iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("CenÃ¡rio Conservador (5%)", f"R$ {last_row['Pessimista (5%)']:,.0f}")
                c2.metric("CenÃ¡rio Base (50%)", f"R$ {last_row['Base (50%)']:,.0f}")
                c3.metric("CenÃ¡rio Otimista (95%)", f"R$ {last_row['Otimista (95%)']:,.0f}")
                
                st.plotly_chart(px.line(sim_df, title=f"Cone de Probabilidade - PrÃ³ximos {mc_years} Anos"), use_container_width=True)

        # --- TAB 5: DADOS BRUTOS ---
        with tab5:
            st.subheader("Dados Fundamentais (Brapi Snapshot)")
            if not fundamentals.empty:
                st.dataframe(fundamentals)
            else:
                st.info("Nenhum dado fundamental carregado.")

if __name__ == "__main__":
    main()