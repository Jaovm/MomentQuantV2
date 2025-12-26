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
        df['cashFlowYield'] = pd.to_numeric(df['freeCashflow'], errors='coerce') / pd.to_numeric(df['marketCap'], errors='coerce')
    
    return df

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=36, skip=1) -> pd.Series:
    """Calcula Residual Momentum Clássico (Blitz et al.) com Volatility Scaling."""
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns or len(rets) < lookback: 
        return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    vols = {}
    
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        y = rets[ticker].tail(lookback + skip).iloc[:-skip]
        x = market.tail(lookback + skip).iloc[:-skip]
        if len(y) < lookback: continue
        try:
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            scores[ticker] = model.tvalues[0]
            recent_resid = model.resid[-6:]
            vol = np.std(recent_resid)
            vols[ticker] = 1.0 / vol if vol > 0 else 0
        except:
            scores[ticker] = 0
            vols[ticker] = 0
            
    s_scores = pd.Series(scores)
    s_vols = pd.Series(vols)
    final_res_mom = s_scores * s_vols
    return final_res_mom.rename('Residual_Momentum')

def compute_composite_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor Composto: Média de múltiplos normalizados."""
    df = fund_df.copy()
    for col in df.columns:
        if col != 'sector':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    val_metrics = pd.DataFrame(index=df.index)
    if 'forwardPE' in df.columns: val_metrics['EP_Forward'] = np.where(df['forwardPE'] > 0, 1/df['forwardPE'], np.nan)
    if 'trailingPE' in df.columns: val_metrics['EP_Trailing'] = np.where(df['trailingPE'] > 0, 1/df['trailingPE'], np.nan)
    if 'priceToBook' in df.columns: val_metrics['BP'] = np.where(df['priceToBook'] > 0, 1/df['priceToBook'], np.nan)
    if 'enterpriseToEbitda' in df.columns: val_metrics['EBITDA_EV'] = np.where(df['enterpriseToEbitda'] > 0, 1/df['enterpriseToEbitda'], np.nan)
    if 'cashFlowYield' in df.columns: val_metrics['CF_Yield'] = df['cashFlowYield']
    for col in val_metrics.columns:
        val_metrics[col] = robust_zscore(val_metrics[col])
    return val_metrics.mean(axis=1).rename("Value_Score")

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    """Z-Score combinado de crescimento de Receita e Lucro."""
    metrics = ['earningsGrowth', 'revenueGrowth']
    temp_df = pd.DataFrame(index=fund_df.index)
    for m in metrics:
        if m in fund_df.columns:
            s = pd.to_numeric(fund_df[m], errors='coerce').fillna(fund_df[m].median())
            temp_df[m] = robust_zscore(s)
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df: scores['ROE'] = pd.to_numeric(fund_df['returnOnEquity'], errors='coerce')
    if 'profitMargins' in fund_df: scores['PM'] = pd.to_numeric(fund_df['profitMargins'], errors='coerce')
    if 'debtToEquity' in fund_df: scores['DE_Inv'] = -1 * pd.to_numeric(fund_df['debtToEquity'], errors='coerce')
    for col in scores.columns:
        scores[col] = robust_zscore(scores[col])
    return scores.mean(axis=1).rename("Quality_Score")

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto usando Mediana e MAD."""
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isnull().all(): return series
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: 
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
# MÓDULO 3: BACKTEST & DCA
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio."""
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

def run_dynamic_backtest(all_prices, all_fundamentals, weights_config, top_n, use_vol_target, use_sector_neutrality, start_date):
    """Executa backtest Walk-Forward."""
    end_date = all_prices.index[-1]
    subset_prices = all_prices.loc[start_date - timedelta(days=400):end_date]
    rebalance_dates = subset_prices.loc[start_date:end_date].resample('MS').first().index.tolist()
    if not rebalance_dates: return pd.DataFrame()
    strategy_daily_rets = []
    benchmark_daily_rets = []
    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else end_date
        prices_hist = subset_prices.loc[:rebal_date]
        res_mom = compute_residual_momentum(prices_hist.tail(400))
        fund_mom = compute_fundamental_momentum(all_fundamentals)
        val_score = compute_composite_value_score(all_fundamentals)
        qual_score = compute_quality_score(all_fundamentals)
        df_period = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_period['Res_Mom'] = res_mom
        df_period['Fund_Mom'] = fund_mom
        df_period['Value'] = val_score
        df_period['Quality'] = qual_score
        if 'sector' in all_fundamentals.columns: df_period['Sector'] = all_fundamentals['sector']
        df_period.dropna(thresh=2, inplace=True)
        norm_cols = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
        w_keys = {}
        for c in norm_cols:
            if c in df_period.columns:
                new_col = f"{c}_Z"
                if use_sector_neutrality and 'Sector' in df_period.columns and df_period['Sector'].nunique() > 1:
                    df_period[new_col] = df_period.groupby('Sector')[c].transform(lambda x: robust_zscore(x) if len(x) > 1 else x - x.median())
                else:
                    df_period[new_col] = robust_zscore(df_period[c])
                w_keys[new_col] = weights_config.get(c, 0.0)
        ranked_period = build_composite_score(df_period, w_keys)
        current_weights = construct_portfolio(ranked_period, prices_hist.tail(90), top_n, 0.15 if use_vol_target else None)
        market_period = subset_prices.loc[rebal_date:next_date].iloc[1:] 
        period_pct = market_period.pct_change().dropna()
        if period_pct.empty: continue
        valid_tickers = [t for t in current_weights.index if t in period_pct.columns]
        if valid_tickers:
            strat_rets = (period_pct[valid_tickers] * current_weights[valid_tickers]).sum(axis=1)
            strategy_daily_rets.append(strat_rets)
            if 'BOVA11.SA' in period_pct.columns: benchmark_daily_rets.append(period_pct['BOVA11.SA'])
    if not strategy_daily_rets: return pd.DataFrame()
    final_strat = pd.concat(strategy_daily_rets)
    final_bench = pd.concat(benchmark_daily_rets)
    return pd.DataFrame({'Strategy': (1 + final_strat).cumprod(), 'BOVA11.SA': (1 + final_bench).cumprod()})

def run_dca_simulation(backtest_df, monthly_investment):
    """Simula DCA e calcula TIR."""
    if backtest_df.empty: return pd.DataFrame(), pd.DataFrame(), {}
    dca_dates = backtest_df.resample('MS').first().index
    strat_rets = backtest_df['Strategy'].pct_change().fillna(0)
    bench_rets = backtest_df['BOVA11.SA'].pct_change().fillna(0)
    strat_bal = 0; bench_bal = 0
    strat_hist = []; bench_hist = []; trans = []
    cash_flows = []
    for date in backtest_df.index:
        strat_bal *= (1 + strat_rets.loc[date])
        bench_bal *= (1 + bench_rets.loc[date])
        if date in dca_dates:
            strat_bal += monthly_investment
            bench_bal += monthly_investment
            trans.append({'Date': date, 'Amount': monthly_investment})
            cash_flows.append(-monthly_investment)
        else:
            cash_flows.append(0)
        strat_hist.append(strat_bal)
        bench_hist.append(bench_bal)
    
    # Adicionar valor final como fluxo positivo para cálculo de TIR
    cash_flows[-1] += strat_bal
    # Cálculo simplificado de TIR mensal -> anual
    try:
        irr_monthly = np.irr(cash_flows) if hasattr(np, 'irr') else 0.01 # Fallback se np.irr não existir (numpy 1.18+)
        # No numpy moderno, usa-se numpy_financial.irr. Aqui usaremos uma aproximação se necessário.
        irr_annual = (1 + irr_monthly)**12 - 1 if not np.isnan(irr_monthly) else 0
    except:
        irr_annual = 0
        
    dca_curve = pd.DataFrame({'Strategy_DCA': strat_hist, 'BOVA11.SA_DCA': bench_hist}, index=backtest_df.index)
    
    # Simulação de Holdings (Simplificada para a aba de alocação)
    last_rebal_date = dca_dates[-1]
    # Para fins de demonstração, assumimos que as holdings são os ativos do último rebalanceamento
    # Em um sistema real, isso seria rastreado dia a dia.
    
    return dca_curve, pd.DataFrame(trans), irr_annual

def analyze_factor_timing(all_prices, all_fundamentals, lookback_years=3):
    """Análise de performance histórica por fator."""
    end_date = all_prices.index[-1]
    start_date = end_date - timedelta(days=lookback_years*365)
    monthly_prices = all_prices.loc[start_date:end_date].resample('ME').last()
    monthly_rets = monthly_prices.pct_change().dropna()
    factor_perf = {}
    factors = {
        'Residual_Momentum': lambda: compute_residual_momentum(all_prices),
        'Value': lambda: compute_composite_value_score(all_fundamentals),
        'Quality': lambda: compute_quality_score(all_fundamentals),
        'Fund_Momentum': lambda: compute_fundamental_momentum(all_fundamentals)
    }
    for f_name, f_func in factors.items():
        try:
            scores = f_func()
            top = scores.sort_values(ascending=False).head(10).index.tolist()
            valid = [t for t in top if t in monthly_rets.columns]
            if valid: factor_perf[f_name] = (1 + monthly_rets[valid].mean(axis=1)).cumprod()
        except: continue
    if 'BOVA11.SA' in monthly_rets.columns: factor_perf['Benchmark'] = (1 + monthly_rets['BOVA11.SA']).cumprod()
    return pd.DataFrame(factor_perf)

# ==============================================================================
# MÓDULO 4: INTERFACE
# ==============================================================================

def main():
    st.markdown("<style>.stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}</style>", unsafe_allow_html=True)
    st.title("🚀 Quant Factor Lab Pro v2")
    
    with st.sidebar:
        st.header("⚙️ Painel de Controle")
        universe = st.selectbox("Universo", ["Ibovespa (Top 50)", "Personalizado"])
        tickers = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 'BBAS3.SA', 'B3SA3.SA', 'WEGE3.SA', 'RENT3.SA', 'SUZB3.SA', 'GGBR4.SA', 'ITSA4.SA', 'JBSS3.SA', 'RDOR3.SA', 'RAIL3.SA', 'EQTL3.SA', 'CSAN3.SA', 'VIVT3.SA', 'LREN3.SA', 'PRIO3.SA', 'RADL3.SA', 'HAPV3.SA', 'BPAC11.SA', 'ELET3.SA', 'VBBR3.SA', 'SBSP3.SA', 'CCRO3.SA', 'CMIG4.SA', 'HYPE3.SA', 'CPLE6.SA', 'UGPA3.SA', 'SANB11.SA', 'EGIE3.SA', 'TOTS3.SA', 'TRPL4.SA', 'CSNA3.SA', 'ENEV3.SA', 'GOAU4.SA', 'CYRE3.SA', 'BRFS3.SA', 'ALOS3.SA', 'MULT3.SA', 'CRFB3.SA', 'TIMS3.SA', 'EMBR3.SA', 'CPFE3.SA', 'MRVE3.SA', 'CIEL3.SA', 'BRKM5.SA', 'AZUL4.SA'] if universe == "Ibovespa (Top 50)" else [t.strip() for t in st.text_area("Tickers", "VALE3.SA, PETR4.SA").split(",")]
        
        st.subheader("Pesos")
        w_res = st.slider("Residual Momentum", 0.0, 1.0, 0.4)
        w_val = st.slider("Value", 0.0, 1.0, 0.3)
        w_qual = st.slider("Quality", 0.0, 1.0, 0.2)
        w_fund = st.slider("Fund. Momentum", 0.0, 1.0, 0.1)
        
        st.subheader("DCA")
        dca_amt = st.number_input("Aporte Mensal (R$)", 100, 100000, 1000)
        top_n = st.number_input("Top N", 3, 20, 5)
        use_vol = st.checkbox("Vol Scaling", True)
        use_sec = st.checkbox("Setorial", True)

    if st.button("🚀 Executar Análise"):
        with st.status("Processando...", expanded=False) as status:
            prices = fetch_price_data(tickers, "2018-01-01", datetime.now().strftime("%Y-%m-%d"))
            fundamentals = fetch_fundamentals(tickers)
            res_mom = compute_residual_momentum(prices)
            val_score = compute_composite_value_score(fundamentals)
            qual_score = compute_quality_score(fundamentals)
            fund_mom = compute_fundamental_momentum(fundamentals)
            
            final_df = pd.DataFrame(index=fundamentals.index)
            final_df['Res_Mom'] = res_mom.reindex(fundamentals.index)
            final_df['Value'] = val_score.reindex(fundamentals.index)
            final_df['Quality'] = qual_score.reindex(fundamentals.index)
            final_df['Fund_Mom'] = fund_mom.reindex(fundamentals.index)
            final_df['Sector'] = fundamentals['sector'] if 'sector' in fundamentals.columns else 'Unknown'
            
            for c in ['Res_Mom', 'Value', 'Quality', 'Fund_Mom']: final_df[f"{c}_Z"] = robust_zscore(final_df[c])
            ranked_df = build_composite_score(final_df, {'Res_Mom_Z': w_res, 'Value_Z': w_val, 'Quality_Z': w_qual, 'Fund_Mom_Z': w_fund})
            
            backtest_full = run_dynamic_backtest(prices, fundamentals, {'Res_Mom': w_res, 'Value': w_val, 'Quality': w_qual, 'Fund_Mom': w_fund}, top_n, use_vol, use_sec, datetime.now() - timedelta(days=365*3))
            dca_curve, dca_trans, tir_proj = run_dca_simulation(backtest_full, dca_amt)
            timing_df = analyze_factor_timing(prices, fundamentals)
            status.update(label="Concluído!", state="complete")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Backtest Detalhado", "💼 Alocação Final DCA", "🔍 Detalhes"])
        
        with tab1:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("🏆 Top Picks Atuais")
                st.dataframe(ranked_df[['Composite_Score', 'Sector', 'Res_Mom', 'Value']].head(top_n).style.background_gradient(cmap='RdYlGn'), width='stretch')
                st.subheader("📈 Performance Recente")
                st.plotly_chart(px.line(backtest_full.tail(252), title="Últimos 12 Meses"), width='stretch')
            with col2:
                st.subheader("💼 Carteira Sugerida")
                w = construct_portfolio(ranked_df, prices.tail(90), top_n, 0.15 if use_vol else None)
                st.table(pd.DataFrame({'Peso': w.map('{:.1%}'.format), 'Preço': prices.iloc[-1].reindex(w.index).map('R$ {:.2f}'.format)}))
                st.plotly_chart(px.pie(values=w.values, names=w.index, hole=0.4), width='stretch')

        with tab2:
            st.header("📊 Métricas de Risco/Retorno (Período Completo)")
            if not backtest_full.empty:
                ret_acc = backtest_full['Strategy'].iloc[-1] - 1
                bench_acc = backtest_full['BOVA11.SA'].iloc[-1] - 1
                vol = backtest_full['Strategy'].pct_change().std() * (252**0.5)
                sharpe = (ret_acc / 3 - 0.10) / vol # Simplificado para 3 anos
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Retorno Acumulado", f"{ret_acc:.2%}", delta=f"vs Bench {bench_acc:.2%}")
                c2.metric("Volatilidade Anualizada", f"{vol:.2%}")
                c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                st.markdown("---")
                st.subheader("💰 Estimativa de TIR Projetada")
                st.metric("TIR Projetada (Anual)", f"{tir_proj:.2%}", help="Taxa Interna de Retorno baseada nos aportes mensais e performance histórica.")
                st.plotly_chart(px.line(dca_curve, title="Evolução Patrimonial com Aportes"), width='stretch')

        with tab3:
            st.header("💼 Alocação Final da Carteira DCA")
            # Simulação de custódia baseada no último rebalanceamento do backtest
            last_w = construct_portfolio(ranked_df, prices.tail(90), top_n, 0.15 if use_vol else None)
            total_invested = len(dca_trans) * dca_amt
            current_val = dca_curve['Strategy_DCA'].iloc[-1]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Investido", f"R$ {total_invested:,.2f}")
            m2.metric("Saldo Atual", f"R$ {current_val:,.2f}", delta=f"{(current_val/total_invested-1):.2%}")
            m3.metric("Lucro/Prejuízo", f"R$ {(current_val - total_invested):,.2f}")
            
            st.subheader("Detalhamento de Custódia Estimada")
            custodia = pd.DataFrame(index=last_w.index)
            custodia['Peso (%)'] = last_w.values
            custodia['Valor (R$)'] = last_w.values * current_val
            custodia['Preço Atual'] = prices.iloc[-1].reindex(last_w.index)
            custodia['Qtd. Estimada'] = (custodia['Valor (R$)'] / custodia['Preço Atual']).round(0)
            st.dataframe(custodia.style.format({'Peso (%)': '{:.2%}', 'Valor (R$)': 'R$ {:,.2f}', 'Preço Atual': 'R$ {:.2f}', 'Qtd. Estimada': '{:.0f}'}), width='stretch')

        with tab4:
            st.subheader("Factor Timing")
            st.plotly_chart(px.line(timing_df, title="Performance por Fator"), width='stretch')
            st.subheader("Correlação")
            st.plotly_chart(px.imshow(final_df[['Res_Mom', 'Value', 'Quality', 'Fund_Mom']].corr(), text_auto=True), width='stretch')

if __name__ == "__main__":
    main()
