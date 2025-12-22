import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
from datetime import datetime, timedelta
import time
import fundamentus

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro (Fundamentus + DCA)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING & PRE-PROCESSING
# ==============================================================================
@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date, end_date):
    t_list_sa = [t if t.endswith('.SA') else f"{t}.SA" for t in list(tickers)]
    if 'BOVA11.SA' not in t_list_sa: 
        t_list_sa.append('BOVA11.SA')
  
    for attempt in range(3):
        try:
            data = yf.download(t_list_sa, start=start_date, end=end_date, progress=False, 
                               auto_adjust=False, threads=False)['Adj Close']
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = data.columns.get_level_values(0)
            return data.ffill().dropna(how='all')
        except Exception as e:
            if attempt == 2:
                st.error(f"Erro ao baixar preços: {e}")
            time.sleep(2)
    return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals_fundamentus(tickers: list):
    try:
        df_raw = fundamentus.get_resultado()
        df_raw.columns = [c.strip().lower().replace('.', '').replace(' ', '_').replace('/', '_') 
                          for c in df_raw.columns]
        valid_tickers = [t.replace('.SA', '') for t in tickers 
                         if t.replace('.SA', '') in df_raw.index]
        if not valid_tickers: 
            return pd.DataFrame()
       
        df = df_raw.loc[valid_tickers].copy()
        df_final = pd.DataFrame(index=df.index)
       
        def get_col(opts):
            for o in opts:
                if o in df.columns: 
                    return pd.to_numeric(df[o], errors='coerce')
            return np.nan
        df_final['pl'] = get_col(['pl'])
        df_final['pvp'] = get_col(['pvp'])
        df_final['roe'] = get_col(['roe'])
        df_final['mrgliq'] = get_col(['mrg_liq', 'marg_liquida'])
        df_final['div_bruta_patrim'] = get_col(['div_bruta_patrim', 'divb_patr'])
        df_final['cresc_rec5'] = get_col(['cres_rec5', 'cresc_rec_5a'])
       
        df_final.index = [f"{x}.SA" for x in df_final.index]
        return df_final
    except Exception as e:
        st.error(f"Erro ao buscar fundamentos: {e}")
        return pd.DataFrame()

# ==============================================================================
# MÓDULO 2: CÁLCULOS QUANTITATIVOS
# ==============================================================================
def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1):
    monthly = price_df.resample('ME').last().ffill()
    rets = monthly.pct_change(fill_method=None).dropna()
    if 'BOVA11.SA' not in rets.columns or len(rets) < lookback: 
        return pd.Series(dtype=float)
   
    market = rets['BOVA11.SA']
    scores = {}
    for t in rets.columns:
        if t == 'BOVA11.SA': 
            continue
        y, x = rets[t].tail(lookback+skip), market.tail(lookback+skip)
        try:
            res = sm.OLS(y.values, sm.add_constant(x.values)).fit().resid[:-skip]
            scores[t] = np.sum(res) / np.std(res) if np.std(res) > 0 else 0
        except: 
            scores[t] = 0
    return pd.Series(scores)

def robust_zscore(series):
    s = series.replace([np.inf, -np.inf], np.nan)
    mad = (s - s.median()).abs().median()
    if mad == 0 or np.isnan(mad): 
        return s.fillna(0) - s.median()
    return ((s - s.median()) / (mad * 1.4826)).clip(-3, 3).fillna(0)

def construct_portfolio(ranked_df, prices_subset, top_n, vol_target=None):
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: 
        return pd.Series()
    if vol_target:
        vols = prices_subset[selected].pct_change(fill_method=None).std() * (252**0.5)
        vols = vols.replace(0, 0.001).fillna(0.20)
        w = (1/vols) / (1/vols).sum()
        return w
    return pd.Series(1/len(selected), index=selected)

# ==============================================================================
# MÓDULO 3: MOTOR DCA
# ==============================================================================
def run_dca_simulation(all_prices, all_fund, weights_config, top_n, dca_val, use_vol, start_date):
    end_date = all_prices.index[-1]
    dca_dates = all_prices.loc[start_date:end_date].resample('MS').first().index.tolist()
   
    portfolio = {}
    bench_shares = 0.0
    daily_equity = pd.DataFrame(index=all_prices.loc[start_date:end_date].index)
    daily_equity[['Strategy', 'Benchmark', 'Invested']] = 0.0
    history = []
    total_invested = 0.0
    
    for i, rebal_date in enumerate(dca_dates):
        hist_p = all_prices.loc[:rebal_date]
        if hist_p.empty: 
            continue
        curr_p = hist_p.iloc[-1]
       
        # Fatores
        res_mom = compute_residual_momentum(hist_p.tail(400))
        df_step = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_step['Res_Mom'] = res_mom
        df_step['Quality'] = (robust_zscore(all_fund['roe']) + robust_zscore(all_fund['mrgliq'])) / 2
        df_step['Value'] = (robust_zscore(1/all_fund['pl'].replace(0, np.nan)) + 
                            robust_zscore(1/all_fund['pvp'].replace(0, np.nan))) / 2
       
        # Z-scores e composite
        weights_keys = {f"{k}_Z": weights_config.get(k, 0) for k in ['Res_Mom', 'Value', 'Quality']}
        for k in ['Res_Mom', 'Value', 'Quality']: 
            if k in df_step.columns:
                df_step[f"{k}_Z"] = robust_zscore(df_step[k])
       
        final_scores = (df_step[list(weights_keys.keys())] * pd.Series(weights_keys)).sum(axis=1)
        ranked = pd.DataFrame({'score': final_scores}).sort_values('score', ascending=False)
        w = construct_portfolio(ranked, hist_p.tail(90), top_n, 0.15 if use_vol else None)
       
        # Aporte
        total_invested += dca_val
        if 'BOVA11.SA' in curr_p.index and pd.notna(curr_p['BOVA11.SA']):
            bench_shares += dca_val / curr_p['BOVA11.SA']
       
        for t, weight in w.items():
            if t in curr_p.index and pd.notna(curr_p[t]):
                qty = (dca_val * weight) / curr_p[t]
                portfolio[t] = portfolio.get(t, 0.0) + qty
                history.append({'Data': rebal_date, 'Ativo': t.replace('.SA',''), 'Qtd': qty, 
                                'Preço': curr_p[t], 'Investido': dca_val * weight})
        
        # Atualiza patrimônio diário
        next_date = dca_dates[i+1] if i < len(dca_dates)-1 else end_date + timedelta(days=1)
        period_idx = all_prices.loc[rebal_date:next_date].index
        for d in period_idx:
            if d > end_date: 
                break
            strat_val = sum(all_prices.loc[d, t] * q for t, q in portfolio.items() if t in all_prices.columns)
            bench_val = all_prices.loc[d, 'BOVA11.SA'] * bench_shares if 'BOVA11.SA' in all_prices.columns else 0
            daily_equity.at[d, 'Strategy'] = strat_val
            daily_equity.at[d, 'Benchmark'] = bench_val
            daily_equity.at[d, 'Invested'] = total_invested
    
    daily_equity = daily_equity.ffill()
    return daily_equity, history, portfolio

# ==============================================================================
# MÓDULO 4: INTERFACE STREAMLIT
# ==============================================================================
def main():
    st.title("🧪 Quant Factor Lab Pro – Versão Simplificada & Otimizada")
    st.markdown("**Multi-Factor (Momentum + Value + Quality) com Simulação DCA Realista**")

    st.sidebar.header("⚙️ Configurações")
    univ = st.sidebar.text_area("Universo de Ativos (vírgula)", 
                                "ITUB3, TOTS3, MDIA3, TAEE3, BBSE3, WEGE3, PSSA3, EGIE3, B3SA3, VIVT3, AGRO3, PRIO3, BBAS3, BPAC11, SBSP3, SAPR4, CMIG3, UNIP6, FRAS3", 
                                height=100)
   
    st.sidebar.subheader("Pesos dos Fatores (soma não precisa ser 1)")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.50)
    w_val = st.sidebar.slider("Value (1/P-L + 1/P-VP)", 0.0, 1.0, 0.30)
    w_qual = st.sidebar.slider("Quality (ROE + Margem Líq.)", 0.0, 1.0, 0.20)
   
    st.sidebar.subheader("Portfólio & DCA")
    top_n = st.sidebar.number_input("Top N Ativos", 3, 10, 5)
    use_vol = st.sidebar.checkbox("Risk Parity (Vol Target 15%)", True)
    dca_val = st.sidebar.number_input("Aporte Mensal (R$)", 100, 20000, 1000, step=100)
    dca_years = st.sidebar.slider("Período de Simulação (anos)", 1, 5, 3)

    if st.sidebar.button("🚀 Iniciar Análise", type="primary"):
        raw_tickers = [t.strip().upper() for t in univ.split(',') if t.strip()]
        if not raw_tickers:
            st.error("Insira ao menos um ticker.")
            return
            
        end_date = datetime.now()
        start_total = end_date - timedelta(days=365 * (dca_years + 2))  # buffer extra
        
        with st.status("Baixando dados e simulando...", expanded=True) as status:
            st.write("⬇️ Preços históricos...")
            prices = fetch_price_data(raw_tickers, start_total, end_date)
            st.write("⬇️ Fundamentos (Fundamentus)...")
            funds = fetch_fundamentals_fundamentus(raw_tickers + ['BOVA11'])
            
            if prices.empty or funds.empty:
                st.error("Falha ao obter dados.")
                return
                
            common = set(prices.columns) & set(funds.index)
            if not common:
                st.error("Nenhum ativo em comum.")
                return
                
            prices = prices[list(common) + ['BOVA11.SA']] if 'BOVA11.SA' in prices.columns else prices[list(common)]
            funds = funds.loc[list(common)]
            
            weights = {'Res_Mom': w_rm, 'Value': w_val, 'Quality': w_qual}
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v/total_w for k,v in weights.items()}
            
            start_dca = end_date - timedelta(days=365*dca_years)
            status.write("💰 Simulando DCA...")
            dca_curve, dca_hist, final_holdings = run_dca_simulation(
                prices, funds, weights, top_n, dca_val, use_vol, start_dca)
            
            status.update(label="Concluído!", state="complete")
        
        # ==============================================================================
        # DASHBOARD
        # ==============================================================================
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance DCA", "📜 Histórico de Aportes", "💼 Custódia Final", "🔍 Fundamentos"])

        with tab1:
            st.subheader(f"Simulação DCA – R$ {dca_val:,} mensais por {dca_years} anos")
            
            if not dca_curve.empty:
                invested = dca_curve['Invested'].iloc[-1]
                ret_strat = (dca_curve['Strategy'].iloc[-1] / invested) - 1
                ret_bench = (dca_curve['Benchmark'].iloc[-1] / invested) - 1
                rets_d = dca_curve['Strategy'].pct_change().dropna()
                vol_ann = rets_d.std() * np.sqrt(252)
                sharpe = ((ret_strat / dca_years) - 0.10) / vol_ann if vol_ann > 0 else 0
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno sobre Aportado", f"{ret_strat:.2%}", f"{ret_strat - ret_bench:+.2%} vs BOVA11")
                c2.metric("Volatilidade Anual", f"{vol_ann:.2%}")
                c3.metric("Sharpe Aprox. (RF 10%)", f"{sharpe:.2f}")
                c4.metric("Total Aportado", f"R$ {invested:,.0f}")
                
                fig = px.line(dca_curve[['Strategy', 'Benchmark', 'Invested']], 
                              title="Evolução do Patrimônio")
                fig.update_layout(yaxis_title="Patrimônio (R$)")
                st.plotly_chart(fig, use_container_width=True)
                
                colf1, colf2 = st.columns(2)
                colf1.metric("Patrimônio Final Estratégia", f"R$ {dca_curve['Strategy'].iloc[-1]:,.0f}")
                colf2.metric("Patrimônio Final BOVA11", f"R$ {dca_curve['Benchmark'].iloc[-1]:,.0f}")

        with tab2:
            st.subheader("Log de Aportes Mensais")
            if dca_hist:
                df_hist = pd.DataFrame(dca_hist)
                df_hist['Data'] = pd.to_datetime(df_hist['Data']).dt.date
                st.dataframe(df_hist.style.format({'Preço': 'R$ {:.2f}', 
                                                   'Investido': 'R$ {:.2f}', 
                                                   'Qtd': '{:.4f}'}), 
                             use_container_width=True)
            else:
                st.info("Nenhum aporte registrado.")

        with tab3:
            st.subheader("Custódia Acumulada Final")
            if final_holdings:
                last_p = prices.iloc[-1]
                alloc = []
                for t, q in final_holdings.items():
                    if t in last_p.index:
                        val = q * last_p[t]
                        alloc.append({'Ativo': t.replace('.SA',''), 'Qtd': round(q,4), 
                                      'Valor': val})
                df_a = pd.DataFrame(alloc).sort_values('Valor', ascending=False)
                df_a['Peso %'] = (df_a['Valor'] / df_a['Valor'].sum() * 100).round(2)
                total = df_a['Valor'].sum()
                
                st.write(f"**Patrimônio Total: R$ {total:,.0f}**")
                c1, c2 = st.columns([1.5, 1])
                with c1:
                    st.dataframe(df_a.style.format({'Valor': 'R$ {:,.2f}', 'Peso %': '{:.2f}%'}), 
                                 use_container_width=True)
                with c2:
                    st.plotly_chart(px.pie(df_a, values='Valor', names='Ativo', hole=0.4, 
                                           title="Alocação"), use_container_width=True)
            else:
                st.info("Carteira vazia.")

        with tab4:
            st.subheader("Dados de Fundamentos (Fundamentus)")
            st.dataframe(funds.style.background_gradient(cmap='Blues'), use_container_width=True)

if __name__ == "__main__":
    main()
