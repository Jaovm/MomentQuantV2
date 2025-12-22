import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
from datetime import datetime, timedelta
import time
import fundamentus  # pip install fundamentus

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro (Fundamentus)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING (Busca de Dados Robusta)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Busca histórico de preços ajustados com tratamento de erros de API e Cache.
    Usa threads=False para evitar 'database is locked'.
    """
    t_list = list(tickers)
    # Garante sufixo .SA para YFinance
    t_list_sa = [t if t.endswith('.SA') else f"{t}.SA" for t in t_list]
    
    if 'BOVA11.SA' not in t_list_sa:
        t_list_sa.append('BOVA11.SA')
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # threads=False é CRÍTICO para evitar erros de SQLite e Rate Limit
            data = yf.download(
                t_list_sa, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=False,
                threads=False 
            )['Adj Close']
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Limpeza básica
            df_clean = data.dropna(how='all')
            
            if not df_clean.empty:
                return df_clean
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Erro ao baixar preços após {max_retries} tentativas: {e}")
                return pd.DataFrame()
            time.sleep(2 * (attempt + 1)) # Backoff exponencial
            
    return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals_fundamentus(tickers: list) -> pd.DataFrame:
    try:
        df_raw = fundamentus.get_resultado()
        
        # Padronização: Removemos pontos e espaços dos nomes das colunas para facilitar a busca
        df_raw.columns = [c.strip().lower().replace('.', '').replace(' ', '_') for c in df_raw.columns]
        
        clean_tickers_input = [t.replace('.SA', '') for t in tickers]
        valid_tickers = [t for t in clean_tickers_input if t in df_raw.index]
        
        if not valid_tickers:
            return pd.DataFrame()

        df = df_raw.loc[valid_tickers].copy()
        df_final = pd.DataFrame(index=df.index)
        
        # Mapeamento Flexível: Tentamos encontrar a coluna por variações de nome
        def get_col(options):
            for opt in options:
                if opt in df.columns: return pd.to_numeric(df[opt], errors='coerce')
            return np.nan

        # Valuation
        df_final['pl'] = get_col(['pl'])
        df_final['pvp'] = get_col(['pvp'])
        df_final['evebitda'] = get_col(['evebitda'])
        
        # Qualidade
        df_final['roe'] = get_col(['roe'])
        df_final['mrgliq'] = get_col(['mrg_liq', 'marg_liquida', 'mrgliq'])
        
        # Alavancagem (Onde estava dando erro)
        # A lib costuma retornar 'divb/patr' ou 'div_bruta/patrim'
        df_final['div_bruta_patrim'] = get_col(['div_bruta/patrim', 'divb/patr', 'div_br/patr'])
        
        # Crescimento
        df_final['cresc_rec5'] = get_col(['cres_rec5', 'cresc_rec_5a'])

        # Restaura o sufixo .SA
        df_final.index = [f"{x}.SA" for x in df_final.index]
        return df_final

    except Exception as e:
        st.error(f"Erro ao processar colunas do Fundamentus: {e}")
        # Debug para você ver o que está vindo da API no console
        if 'df_raw' in locals(): st.write("Colunas disponíveis:", df_raw.columns.tolist())
        return pd.DataFrame()

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
    """Z-Score de crescimento de Receita (5 anos)."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'cresc_rec5' in fund_df:
        s = fund_df['cresc_rec5'].fillna(fund_df['cresc_rec5'].median())
        if s.std() != 0:
            scores['Growth'] = (s - s.mean()) / s.std()
        else:
            scores['Growth'] = 0.0
    else:
        scores['Growth'] = 0.0
        
    return scores.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E (P/L) e P/B (P/VP)."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'pl' in fund_df: 
        scores['EP'] = np.where(fund_df['pl'] > 0, 1/fund_df['pl'], 0)
    if 'pvp' in fund_df: 
        scores['BP'] = np.where(fund_df['pvp'] > 0, 1/fund_df['pvp'], 0)
    
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem Liq e Baixa Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'roe' in fund_df: scores['ROE'] = fund_df['roe']
    if 'mrgliq' in fund_df: scores['PM'] = fund_df['mrgliq']
    if 'div_bruta_patrim' in fund_df: scores['DE_Inv'] = -1 * fund_df['div_bruta_patrim']
    
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & AUXILIARES
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

def get_implied_per_share_metrics(prices_current: pd.Series, fundamentals_current: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia Reversa para Backtest:
    Calcula LPA (EPS) e VPA (BVPS) implícitos atuais para usar com preços históricos.
    """
    metrics = pd.DataFrame(index=fundamentals_current.index)
    common_idx = prices_current.index.intersection(fundamentals_current.index)
    
    p = prices_current[common_idx]
    f = fundamentals_current.loc[common_idx]
    
    metrics.loc[common_idx, 'Implied_EPS'] = np.where(f['pl'] != 0, p / f['pl'], np.nan)
    metrics.loc[common_idx, 'Implied_BVPS'] = np.where(f['pvp'] != 0, p / f['pvp'], np.nan)
    
    return metrics

# ==============================================================================
# MÓDULO 4: PORTFOLIO & BACKTEST (CORRIGIDO)
# ==============================================================================

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

def run_dynamic_backtest(
    all_prices: pd.DataFrame, 
    all_fundamentals_snapshot: pd.DataFrame, 
    weights_config: dict, 
    top_n: int, 
    use_vol_target: bool,
    start_date_backtest: datetime
):
    """
    Executa Backtest Walk-Forward usando métricas implícitas para Valuation dinâmico.
    """
    end_date = all_prices.index[-1]
    
    # Prepara métricas fixas
    last_prices = all_prices.iloc[-1]
    implied_metrics = get_implied_per_share_metrics(last_prices, all_fundamentals_snapshot)
    
    subset_prices = all_prices.loc[start_date_backtest - timedelta(days=400):end_date]
    rebalance_dates = subset_prices.loc[start_date_backtest:end_date].resample('MS').first().index.tolist()
    
    if not rebalance_dates: return pd.DataFrame()

    strategy_daily_rets = []
    benchmark_daily_rets = []

    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
        
        # Dados Históricos
        prices_historical = subset_prices.loc[:rebal_date]
        if prices_historical.empty: continue
        
        current_prices_at_rebal = prices_historical.iloc[-1]
        
        # Fatores Dinâmicos
        mom_window = prices_historical.tail(400) 
        risk_window = prices_historical.tail(90)
        res_mom = compute_residual_momentum(mom_window)
        
        # Value Dinâmico (Reconstruído)
        df_period = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_calc = df_period.join(implied_metrics)
        df_calc['Price_Rebal'] = current_prices_at_rebal
        
        # Evita divisão por zero ou NaN
        df_calc = df_calc.fillna(0)
        
        pl_est = np.where(df_calc['Implied_EPS'] != 0, df_calc['Price_Rebal'] / df_calc['Implied_EPS'], 0)
        pvp_est = np.where(df_calc['Implied_BVPS'] != 0, df_calc['Price_Rebal'] / df_calc['Implied_BVPS'], 0)
        
        s_ep = np.where((pl_est > 0) & (pl_est < 200), 1/pl_est, 0)
        s_bp = np.where((pvp_est > 0) & (pvp_est < 20), 1/pvp_est, 0)
        val_score = (pd.Series(s_ep, index=df_period.index) + pd.Series(s_bp, index=df_period.index)) / 2
        
        # Fatores Estáticos (Quality/Growth via Snapshot Fundamentus)
        fund_mom = compute_fundamental_momentum(all_fundamentals_snapshot)
        qual_score = compute_quality_score(all_fundamentals_snapshot)
        
        # Montagem do DataFrame
        df_period['Res_Mom'] = res_mom
        df_period['Fund_Mom'] = fund_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        
        df_period.dropna(thresh=2, inplace=True)
        
        norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
        w_keys = {}
        
        for c in norm_cols:
            if c in df_period.columns:
                new_col = f"{c}_Z"
                df_period[new_col] = robust_zscore(df_period[c])
                w_keys[new_col] = weights_config.get(c, 0.0)
                    
        ranked_period = build_composite_score(df_period, w_keys)
        current_weights = construct_portfolio(ranked_period, risk_window, top_n, 0.15 if use_vol_target else None)
        
        # Cálculo de Retorno
        market_period = subset_prices.loc[rebal_date:next_date].iloc[1:] 
        period_pct = market_period.pct_change().dropna()
        if period_pct.empty: continue
            
        valid_tickers = [t for t in current_weights.index if t in period_pct.columns]
        if valid_tickers:
            strat_ret = period_pct[valid_tickers].dot(current_weights[valid_tickers])
        else:
            strat_ret = pd.Series(0.0, index=period_pct.index)
            
        if 'BOVA11.SA' in period_pct.columns:
            bench_ret = period_pct['BOVA11.SA']
        else:
            bench_ret = pd.Series(0.0, index=period_pct.index)
            
        strategy_daily_rets.append(strat_ret)
        benchmark_daily_rets.append(bench_ret)
        
    if strategy_daily_rets:
        full_strategy = pd.concat(strategy_daily_rets)
        full_benchmark = pd.concat(benchmark_daily_rets)
        full_strategy = full_strategy[~full_strategy.index.duplicated(keep='first')]
        full_benchmark = full_benchmark[~full_benchmark.index.duplicated(keep='first')]
        
        cumulative = pd.DataFrame({
            'Strategy': (1 + full_strategy).cumprod(),
            'BOVA11.SA': (1 + full_benchmark).cumprod()
        })
        return cumulative.dropna()
    return pd.DataFrame()

# ==============================================================================
# APP PRINCIPAL (STREAMLIT UI)
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Fundamentus Edition")
    st.markdown("""
    **Engine Otimizada:** Integração com biblioteca `fundamentus` e correção de Look-ahead Bias.
    *Download lento proposital (threads=False) para estabilidade.*
    """)

    st.sidebar.header("1. Universo e Dados")
    default_univ = "ITUB4, VALE3, WEGE3, PETR4, BBAS3, JBSS3, ELET3, RENT3, SUZB3, GGBR4, RAIL3, BPAC11, PRIO3, VBBR3, HYPE3, RADL3, B3SA3, CMIG4, TOTS3, VIVT3"
    ticker_input = st.sidebar.text_area("Tickers (Separados por vírgula)", default_univ, height=100)
    
    # Tratamento de input
    raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    fundamentus_tickers = [t.replace('.SA', '') for t in raw_tickers]
    yfinance_tickers = [f"{t}.SA" if not t.endswith('.SA') else t for t in fundamentus_tickers]

    st.sidebar.header("2. Pesos dos Fatores")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_fm = st.sidebar.slider("Growth (Fundamentus)", 0.0, 1.0, 0.10)
    w_val = st.sidebar.slider("Value (P/L & P/VP)", 0.0, 1.0, 0.30)
    w_qual = st.sidebar.slider("Quality (ROE & Margem)", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Configurações")
    top_n = st.sidebar.number_input("Top N Ativos", 1, 20, 5)
    use_vol_target = st.sidebar.checkbox("Vol Target (Risk Parity)", True)
    
    run_btn = st.sidebar.button("🚀 Rodar Análise", type="primary")

    if run_btn:
        if not raw_tickers:
            st.error("Insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant...", expanded=True) as status:
            end_date = datetime.now()
            start_date_total = end_date - timedelta(days=365 * 4) 
            start_date_backtest = end_date - timedelta(days=365 * 2) 

            # 1. Busca Preços
            st.write("Baixando preços históricos (Modo seguro)...")
            prices = fetch_price_data(yfinance_tickers, start_date_total, end_date)
            
            # 2. Busca Fundamentos
            st.write("Buscando dados no Fundamentus...")
            fundamentals = fetch_fundamentals_fundamentus(fundamentus_tickers)
            
            if prices.empty:
                st.error("Falha ao obter preços. Verifique sua conexão ou os tickers.")
                status.update(label="Erro!", state="error")
                return
                
            if fundamentals.empty:
                st.error("Falha ao obter fundamentos. Verifique os tickers.")
                status.update(label="Erro!", state="error")
                return
            
            # Interseção
            common_tickers = list(set(prices.columns) & set(fundamentals.index))
            if not common_tickers:
                st.error("Sem interseção entre dados de preço e fundamentos.")
                st.stop()
                
            prices = prices[common_tickers + ['BOVA11.SA']] if 'BOVA11.SA' in prices.columns else prices[common_tickers]
            fundamentals = fundamentals.loc[common_tickers]

            # 3. Scores Atuais
            st.write("Calculando fatores...")
            res_mom = compute_residual_momentum(prices)
            fund_mom = compute_fundamental_momentum(fundamentals)
            val_score = compute_value_score(fundamentals)
            qual_score = compute_quality_score(fundamentals)

            df_master = pd.DataFrame(index=common_tickers)
            df_master['Res_Mom'] = res_mom
            df_master['Fund_Mom'] = fund_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            df_master.dropna(thresh=2, inplace=True)

            norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
            weights_map = {'Res_Mom': w_rm, 'Fund_Mom': w_fm, 'Value': w_val, 'Quality': w_qual}
            weights_keys = {}

            for c in norm_cols:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    weights_keys[new_col] = weights_map.get(c, 0.0)
            
            final_df = build_composite_score(df_master, weights_keys)
            weights = construct_portfolio(final_df, prices, top_n, 0.15 if use_vol_target else None)
            
            # 4. Backtest
            st.write("Simulando performance passada...")
            backtest_curve = run_dynamic_backtest(
                prices, fundamentals, weights_map, top_n, use_vol_target, start_date_backtest
            )

            status.update(label="Concluído!", state="complete", expanded=False)

        # DASHBOARD
        tab1, tab2, tab3 = st.tabs(["🏆 Ranking Atual", "📈 Backtest", "🔍 Dados Brutos"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Melhores Oportunidades (Hoje)")
                show_cols = ['Composite_Score'] + list(weights_keys.keys())
                st.dataframe(final_df[show_cols].head(top_n).style.background_gradient(cmap='Greens'), height=400, use_container_width=True)
            with col2:
                st.subheader("Alocação Sugerida")
                if not weights.empty:
                    w_df = weights.to_frame(name="Peso")
                    w_df["Peso"] = w_df["Peso"].map("{:.2%}".format)
                    st.table(w_df)
                    st.plotly_chart(px.pie(values=weights.values, names=weights.index, hole=0.4), use_container_width=True)

        with tab2:
            st.subheader("Performance Histórica Estimada")
            if not backtest_curve.empty:
                ret_strat = backtest_curve['Strategy'].iloc[-1] - 1
                ret_bench = backtest_curve['BOVA11.SA'].iloc[-1] - 1
                
                c1, c2 = st.columns(2)
                c1.metric("Retorno Estratégia", f"{ret_strat:.2%}", delta=f"{(ret_strat-ret_bench):.2%}")
                c2.metric("Retorno BOVA11", f"{ret_bench:.2%}")
                
                st.plotly_chart(px.line(backtest_curve, title="Curva de Retorno Acumulado (Base 100)"), use_container_width=True)
            else:
                st.warning("Dados insuficientes para backtest.")

        with tab3:
            st.write("Fundamentos Baixados (Raw Data)")
            st.dataframe(fundamentals)

if __name__ == "__main__":
    main()
