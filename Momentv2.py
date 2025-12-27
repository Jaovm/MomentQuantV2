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
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v3 (Brapi + DCA)",
    layout="wide",
    initial_sidebar_state="expanded"
)

BRAPI_TOKEN = "5gVedSQ928pxhFuTvBFPfr"

# ==============================================================================
# MÓDULO 1: DATA FETCHING (PREÇOS YF & FUNDAMENTOS BRAPI)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    t_list = list(tickers)
    for bench in ['BOVA11.SA', 'DIVO11.SA']:
        if bench not in t_list: t_list.append(bench)
    
    try:
        data = yf.download(t_list, start=start_date, end=end_date, progress=False)['Adj Close']
        if isinstance(data, pd.Series): data = data.to_frame()
        return data.ffill()
    except Exception as e:
        st.error(f"Erro ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*6)
def fetch_fundamentals_brapi(tickers: list, token: str) -> pd.DataFrame:
    # Remove .SA para a Brapi
    clean_tickers = [t.replace('.SA', '') for t in tickers if t not in ['BOVA11.SA', 'DIVO11.SA']]
    chunk_size = 15
    chunks = [clean_tickers[i:i + chunk_size] for i in range(0, len(clean_tickers), chunk_size)]
    
    fundamental_data = []
    for chunk in chunks:
        ticker_str = ','.join(chunk)
        url = f"https://brapi.dev/api/quote/{ticker_str}"
        params = {'token': token, 'fundamental': 'true'}
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    fundamental_data.append({
                        'ticker': item.get('symbol') + ".SA",
                        'sector': item.get('sector', 'Outros'),
                        'PE': item.get('priceEarnings', np.nan),
                        'P_VP': item.get('priceToBook', np.nan),
                        'ROE': item.get('returnOnEquity', np.nan),
                        'Net_Margin': item.get('profitMargin', np.nan),
                    })
            time.sleep(0.4)
        except: pass
    
    return pd.DataFrame(fundamental_data).set_index('ticker') if fundamental_data else pd.DataFrame()

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad < 1e-6: return series - median
    return ((series - median) / (mad * 1.4826)).clip(-3, 3)

def compute_residual_momentum(price_df: pd.DataFrame) -> pd.Series:
    if 'BOVA11.SA' not in price_df.columns: return pd.Series()
    rets = price_df.resample('ME').last().pct_change().dropna()
    market = rets['BOVA11.SA']
    scores = {}
    for ticker in rets.columns:
        if ticker in ['BOVA11.SA', 'DIVO11.SA']: continue
        try:
            y = rets[ticker].tail(36)
            x = market.tail(36)
            model = sm.OLS(y, sm.add_constant(x)).fit()
            # Momentum 12m (residual) pulando o último mês
            scores[ticker] = model.resid.iloc[-13:-1].sum() / model.resid.std()
        except: scores[ticker] = np.nan
    return pd.Series(scores)

# ==============================================================================
# MÓDULO 3: ENGINE DE BACKTEST DCA
# ==============================================================================

def run_full_comparison_backtest(all_prices, fund_df, top_n, dca_amount, start_date):
    all_prices = all_prices.ffill()
    # Correção do erro .tolist() anterior
    cal = pd.Series(all_prices.index, index=all_prices.index)
    dates = cal.loc[start_date:].resample('MS').first().dropna().tolist()
    
    portfolio_val = pd.Series(0.0, index=all_prices.index)
    bova_val = pd.Series(0.0, index=all_prices.index)
    divo_val = pd.Series(0.0, index=all_prices.index)
    
    holdings = {}
    bova_units = 0.0
    divo_units = 0.0
    transactions = []

    for i, d in enumerate(dates):
        # 1. Ranking Point-in-Time (Simplificado para Momentum)
        hist_prices = all_prices.loc[:d]
        mom = compute_residual_momentum(hist_prices)
        if mom.empty: continue
        
        selected = mom.sort_values(ascending=False).head(top_n).index.tolist()
        
        # 2. Aporte
        price_today = all_prices.loc[d]
        
        # Estratégia
        per_asset = dca_amount / len(selected)
        for t in selected:
            if t in price_today and price_today[t] > 0:
                qty = per_asset / price_today[t]
                holdings[t] = holdings.get(t, 0) + qty
                transactions.append({'Data': d, 'Ticker': t, 'Qtd': qty, 'Preço': price_today[t]})
        
        # Benchmarks
        if 'BOVA11.SA' in price_today: bova_units += dca_amount / price_today['BOVA11.SA']
        if 'DIVO11.SA' in price_today: divo_units += dca_amount / price_today['DIVO11.SA']

        # 3. Mark-to-Market
        next_d = dates[i+1] if i < len(dates)-1 else all_prices.index[-1]
        period = all_prices.loc[d:next_d].index
        for day in period:
            portfolio_val[day] = sum(holdings[t] * all_prices.at[day, t] for t in holdings)
            bova_val[day] = bova_units * all_prices.at[day, 'BOVA11.SA']
            divo_val[day] = divo_units * all_prices.at[day, 'DIVO11.SA']

    res = pd.DataFrame({'Estratégia': portfolio_val, 'BOVA11': bova_val, 'DIVO11': divo_val})
    return res[res > 0].dropna(), pd.DataFrame(transactions), holdings

# ==============================================================================
# INTERFACE
# ==============================================================================

def main():
    st.sidebar.title("Configurações")
    ticker_input = st.sidebar.text_area("Lista de Tickers", "ITUB4, VALE3, WEGE3, PETR4, BBAS3, RENT3, PRIO3, B3SA3, LREN3, ABEV3, EQTL3")
    raw_tickers = [t.strip().upper() + ".SA" if ".SA" not in t else t.strip().upper() for t in ticker_input.split(',')]
    
    top_n = st.sidebar.slider("Ativos na Carteira", 3, 15, 5)
    dca_val = st.sidebar.number_input("Aporte Mensal (R$)", 100, 10000, 1000)
    anos = st.sidebar.slider("Anos de Histórico", 1, 10, 3)

    if st.sidebar.button("Rodar Análise"):
        with st.spinner("Consultando Brapi e YFinance..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * anos)
            
            prices = fetch_price_data(raw_tickers, start_date - timedelta(days=500), end_date)
            funds = fetch_fundamentals_brapi(raw_tickers, BRAPI_TOKEN)
            
            # Ranking Atual (HOJE)
            mom_atual = compute_residual_momentum(prices)
            df_rank = pd.DataFrame(index=mom_atual.index)
            df_rank['Momentum_Z'] = robust_zscore(mom_atual)
            
            if not funds.empty:
                # Value = 1/PE e 1/P_VP (quanto menor PE, maior o score)
                funds['Value_Score'] = robust_zscore(1/funds['PE']) + robust_zscore(1/funds['P_VP'])
                df_rank = df_rank.join(funds[['Value_Score', 'sector']])
            
            df_rank['Final_Score'] = df_rank['Momentum_Z'] + df_rank.get('Value_Score', 0)
            df_rank = df_rank.sort_values('Final_Score', ascending=False)

            # Backtest
            curve, trans, final_holdings = run_full_comparison_backtest(prices, funds, top_n, dca_val, start_date)

        # DASHBOARD
        t1, t2, t3 = st.tabs(["📈 Comparativo DCA", "🏆 Ranking Atual", "📋 Alocação Final"])
        
        with t1:
            st.subheader("Evolução Patrimonial (Aportes Mensais)")
            st.plotly_chart(px.line(curve, labels={'value': 'Patrimônio (R$)'}), use_container_width=True)
            
            # Métricas
            c1, c2, c3 = st.columns(3)
            total_inv = len(trans['Data'].unique()) * dca_val
            c1.metric("Total Investido", f"R$ {total_inv:,.2f}")
            c2.metric("Estratégia", f"R$ {curve['Estratégia'].iloc[-1]:,.2f}", 
                      f"{(curve['Estratégia'].iloc[-1]/total_inv - 1):.1%}")
            c3.metric("Benchmark (BOVA11)", f"R$ {curve['BOVA11'].iloc[-1]:,.2f}",
                      f"{(curve['BOVA11'].iloc[-1]/total_inv - 1):.1%}")

        with t2:
            st.subheader("Top Picks para Rebalanceamento")
            st.dataframe(df_rank.head(top_n).style.background_gradient(cmap='RdYlGn'), use_container_width=True)

        with t3:
            st.subheader("Composição da Carteira Acumulada")
            if final_holdings:
                last_p = prices.iloc[-1]
                data_h = []
                for t, q in final_holdings.items():
                    val = q * last_p[t]
                    data_h.append({'Ticker': t, 'Valor': val, 'Setor': funds.at[t, 'sector'] if t in funds.index else 'N/A'})
                
                df_h = pd.DataFrame(data_h)
                col_a, col_b = st.columns(2)
                col_a.plotly_chart(px.pie(df_h, values='Valor', names='Ticker', title="Por Ativo"), use_container_width=True)
                col_b.plotly_chart(px.pie(df_h, values='Valor', names='Setor', title="Por Setor"), use_container_width=True)
                
                st.markdown("**Tabela de Alocação**")
                st.table(df_h.sort_values('Valor', ascending=False))

if __name__ == "__main__":
    main()
