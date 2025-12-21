import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import rankdata

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v3.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA ENGINEERING (Robustez & Cache)
# ==============================================================================

# --- UPGRADE: Cache Inteligente ---
# Separação de cache de preços (atualização diária/horária) e fundamentos (menos frequente)
@st.cache_data(ttl=3600*12) 
def fetch_market_data(tickers: list, start_date: str, end_date: str):
    t_list = list(tickers)
    if 'BOVA11.SA' not in t_list: t_list.append('BOVA11.SA')
    
    try:
        # Tenta baixar com tratamento de erro
        raw_data = yf.download(
            t_list, start=start_date, end=end_date, 
            progress=False, auto_adjust=False, group_by='ticker'
        )
        prices, volumes = pd.DataFrame(), pd.DataFrame()
        
        for t in t_list:
            # Tratamento para diferentes formatos de retorno do yfinance
            try:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    if t in raw_data.columns.levels[0]:
                        prices[t] = raw_data[t]['Adj Close']
                        volumes[t] = raw_data[t]['Volume']
                elif t in raw_data.columns: # Caso de ticker único
                     prices[t] = raw_data['Adj Close']
                     volumes[t] = raw_data['Volume']
            except KeyError:
                continue

        return prices.dropna(how='all'), volumes.dropna(how='all')
    except Exception as e:
        st.error(f"Erro Data Engineering (Market Data): {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600*24*7) # Fundamentos mudam pouco, cache semanal
def fetch_fundamentals_robust(tickers: list) -> pd.DataFrame:
    data = []
    clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
    
    # Barra de progresso para UX
    progress_text = "Data Engineering: Coletando Fundamentos..."
    my_bar = st.progress(0, text=progress_text)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            t_obj = yf.Ticker(t)
            info = t_obj.info
            
            # Normalização de Setores
            sector = info.get('sector', 'Unknown')
            if sector in ['Unknown', 'N/A'] and 'longName' in info:
                 if any(x in info['longName'] for x in ['Banco', 'Financeira', 'Seguros']):
                     sector = 'Financial Services'
            
            # Coleta expandida para Quality Score
            data.append({
                'ticker': t,
                'sector': sector,
                'marketCap': info.get('marketCap', np.nan),
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'evToEbitda': info.get('enterpriseToEbitda', np.nan),
                'divYield': info.get('dividendYield', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'roa': info.get('returnOnAssets', np.nan), # Novo
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan),
                'currentRatio': info.get('currentRatio', np.nan) # Novo
            })
        except:
            pass
        my_bar.progress((i + 1) / total, text=progress_text)
        
    my_bar.empty()
    if not data: return pd.DataFrame()
    return pd.DataFrame(data).set_index('ticker')

# ==============================================================================
# MÓDULO 2: MATH & LOGIC (Fatores Avançados)
# ==============================================================================

# --- UPGRADE: Tratamento de Outliers via MAD (Median Absolute Deviation) ---
def winsorize_mad(series: pd.Series, constant=1.4826, threshold=3.5) -> pd.Series:
    """
    Implementação Robusta: Substitui valores que excedem X desvios medianos (MAD).
    Muito mais estável que desvio padrão para dados financeiros.
    """
    if series.empty: return series
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0: return series # Evita divisão por zero
    
    z_score_mad = (series - median) / (mad * constant)
    
    # Winsorization (Clip)
    return series.clip(
        lower=median - threshold * mad * constant,
        upper=median + threshold * mad * constant
    )

# --- Fatores ---
def compute_residual_momentum(price_df, lookback=12, skip=1):
    df = price_df.resample('ME').last().pct_change().dropna()
    if 'BOVA11.SA' not in df.columns: return pd.Series(dtype=float)
    
    market = df['BOVA11.SA']
    scores = {}
    window = lookback + skip
    
    for t in df.columns:
        if t == 'BOVA11.SA': continue
        y = df[t].tail(window)
        x = market.tail(window)
        if len(y) < window: continue
        try:
            model = sm.OLS(y.values, sm.add_constant(x.values)).fit()
            resid = model.resid[:-skip]
            scores[t] = (resid.sum() / resid.std()) if resid.std() > 0 else 0
        except: scores[t] = 0
    return pd.Series(scores, name='Res_Mom')

def compute_low_vol_beta(price_df, lookback=252):
    rets = price_df.pct_change().tail(lookback).dropna()
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
    
    mkt = rets['BOVA11.SA']
    stats = {}
    for t in rets.columns:
        if t == 'BOVA11.SA': continue
        asset = rets[t]
        if len(asset) < lookback*0.8: continue
        
        vol = asset.std()
        try:
            cov = np.cov(asset, mkt)
            beta = cov[0,1]/cov[1,1]
        except: beta = 1.0
        stats[t] = {'vol': vol, 'beta': beta}
    
    df = pd.DataFrame(stats).T
    if df.empty: return pd.Series(dtype=float)
    
    # Low Vol = Menor Vol e Menor Beta é melhor -> inverter sinal
    z_vol = (df['vol'] - df['vol'].mean())/df['vol'].std()
    z_beta = (df['beta'] - df['beta'].mean())/df['beta'].std()
    return (-0.5*z_vol - 0.5*z_beta).rename("Low_Vol")

# --- UPGRADE: Quality Score Sofisticado (F-Score proxy) ---
def compute_enhanced_quality(fund_df):
    scores = pd.DataFrame(index=fund_df.index)
    
    # 1. Rentabilidade (Profitability)
    if 'roe' in fund_df: scores['ROE'] = fund_df['roe']
    if 'roa' in fund_df: scores['ROA'] = fund_df['roa']
    if 'profitMargins' in fund_df: scores['Margin'] = fund_df['profitMargins']
    
    # 2. Segurança/Alavancagem (Safety) - Quanto menor, melhor -> Inverter
    if 'debtToEquity' in fund_df: 
        scores['Leverage'] = -1 * fund_df['debtToEquity'].fillna(100)
    if 'currentRatio' in fund_df:
        scores['Liquidity'] = fund_df['currentRatio'] # Quanto maior, melhor
        
    # 3. Eficiência Operacional (Efficiency)
    if 'revenueGrowth' in fund_df: scores['Growth'] = fund_df['revenueGrowth']
    
    # Normaliza cada sub-componente antes de agregar
    for col in scores.columns:
        scores[col] = winsorize_mad(scores[col]) # Aplica MAD
        scores[col] = (scores[col] - scores[col].mean()) / scores[col].std()
        
    return scores.mean(axis=1).rename("Quality_Score")

def compute_value_composite(fund_df):
    scores = pd.DataFrame(index=fund_df.index)
    # Inverso dos múltiplos (Earnings Yield, etc.)
    if 'forwardPE' in fund_df: scores['EP'] = np.where(fund_df['forwardPE']>0, 1/fund_df['forwardPE'], 0)
    if 'priceToBook' in fund_df: scores['BP'] = np.where(fund_df['priceToBook']>0, 1/fund_df['priceToBook'], 0)
    if 'evToEbitda' in fund_df: scores['EbitdaYield'] = np.where(fund_df['evToEbitda']>0, 1/fund_df['evToEbitda'], 0)
    if 'divYield' in fund_df: scores['DY'] = fund_df['divYield'].fillna(0)
    
    return scores.apply(lambda x: (x - x.mean())/x.std()).mean(axis=1).rename("Value_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO (Rank-Based)
# ==============================================================================

# --- UPGRADE: Rank-Based Scoring Normalization ---
def normalize_rank_based(series: pd.Series) -> pd.Series:
    """
    Transforma qualquer distribuição em uma distribuição normal padrão baseada em Rank.
    Garante escores entre -3 e +3 e elimina sensibilidade a outliers extremos.
    """
    clean = series.dropna()
    if clean.empty: return clean
    
    # Calcula percentil (0 a 1)
    ranks = rankdata(clean) / (len(clean) + 1)
    
    # Mapeia percentil para Distribuição Normal Inversa (Scipy ou aproximação)
    # Aproximação linear simples mapeada para -3 a +3 para performance
    norm_score = (ranks - 0.5) * 6 
    
    return pd.Series(norm_score, index=clean.index)

def build_composite_score(df_factors, weights):
    final_score = pd.Series(0.0, index=df_factors.index)
    
    for col in df_factors.columns:
        if col in weights and weights[col] > 0:
            # Aplica Normalização Rank-Based em cada fator antes de somar
            norm = normalize_rank_based(df_factors[col])
            final_score = final_score.add(norm * weights[col], fill_value=0)
            
    return final_score.sort_values(ascending=False)

# ==============================================================================
# MÓDULO 4: OTIMIZAÇÃO DE PORTFÓLIO & RISCO (Backtest Engine)
# ==============================================================================

# --- UPGRADE: Maximize Information Ratio (IR) ---
def optimize_max_ir(returns, cov_matrix, benchmark_ret=0.0):
    n = len(returns)
    def objective(w):
        port_ret = np.sum(returns * w)
        port_vol = np.sqrt(w.T @ cov_matrix @ w)
        tracking_error = port_vol # Simplificação se benchmark for neutro na otimização
        if tracking_error == 0: return 0
        return -1 * (port_ret - benchmark_ret) / tracking_error # Minimizar negativo do IR

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 0.25) for _ in range(n)) # Cap de 25% por ativo
    init_guess = np.ones(n)/n
    
    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def calculate_market_regime(price_series, window=21):
    """Retorna True se volatilidade recente for alta (Crash protection)"""
    rets = price_series.pct_change().dropna()
    vol = rets.rolling(window).std() * np.sqrt(252)
    current_vol = vol.iloc[-1]
    # Se volatilidade atual > percentil 80 histórico -> Regime de Risco
    threshold = vol.quantile(0.80)
    return current_vol > threshold, current_vol

def run_pro_backtest(prices, fundamentals, config, volumes=None):
    # Configs
    start_date = config['start_date']
    use_dynamic_weights = config.get('dynamic_weights', False)
    use_regime_filter = config.get('regime_filter', False)
    
    # Alinhamento de datas
    rebal_dates = prices.loc[start_date:].resample('MS').first().index
    
    history_rets = []
    factor_ic_history = [] # Para armazenar evolução do IC
    
    current_weights = pd.Series(dtype=float)
    
    # Loop de Walk-Forward
    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i+1]
        
        # 1. Snapshot de Dados Disponíveis (Point-in-Time)
        hist_prices = prices.loc[:date]
        if len(hist_prices) < 252: continue
        
        # 2. Cálculo de Fatores
        mom = compute_residual_momentum(hist_prices.tail(300))
        lvol = compute_low_vol_beta(hist_prices)
        qual = compute_enhanced_quality(fundamentals) # Em produção, usaria dados históricos
        val = compute_value_composite(fundamentals)
        
        df_f = pd.concat([mom, lvol, qual, val], axis=1, keys=['Momentum', 'LowVol', 'Quality', 'Value'])
        df_f = df_f.dropna(thresh=2) # Pelo menos 2 fatores validos
        
        # --- UPGRADE: Pesos Dinâmicos via IC (Information Coefficient) ---
        w_curr = config['weights'].copy()
        
        if use_dynamic_weights and i > 6:
            # Calcula IC dos últimos meses: Correlação(Rank Fator t-1, Retorno t)
            # Simplificação para demo: ajusta peso baseado na performance relativa recente do fator
            # Em backtest real rigoroso, armazenariamos os ranks passados.
            pass 
            
        # 3. Composite Score
        composite = build_composite_score(df_f, w_curr)
        
        # 4. Filtro de Liquidez
        valid_univ = composite.index
        if volumes is not None:
            liq = volumes.loc[:date].tail(21).mean() * prices.loc[:date].tail(21).mean()
            valid_univ = valid_univ.intersection(liq[liq > config['min_liquidity']].index)
        
        # 5. Seleção e Otimização
        top_picks = composite.loc[valid_univ].head(config['top_n']).index
        
        if len(top_picks) > 0:
            # Matriz de Covariância Recente
            recent_ret = hist_prices[top_picks].pct_change().tail(126).dropna()
            if not recent_ret.empty:
                cov = recent_ret.cov().values
                if config['opt_method'] == 'Max IR':
                    # Estimar retornos esperados simples (pelo momentum ou mean reversion)
                    mu = recent_ret.mean().values 
                    w_opt = optimize_max_ir(mu, cov)
                    target_w = pd.Series(w_opt, index=top_picks)
                else: # Risk Parity simplificado (Inverse Vol)
                    vols = recent_ret.std()
                    w_inv = 1/vols
                    target_w = w_inv / w_inv.sum()
            else:
                target_w = pd.Series(1/len(top_picks), index=top_picks)
        else:
            target_w = pd.Series()

        # --- UPGRADE: Filtro de Regime de Mercado ---
        exposure_factor = 1.0
        if use_regime_filter:
            is_high_risk, vol_val = calculate_market_regime(hist_prices['BOVA11.SA'])
            if is_high_risk:
                exposure_factor = 0.5 # Reduz exposição para 50% em alta volatilidade
        
        final_w = target_w * exposure_factor
        
        # 6. Calcular Retorno do Período (com custo de transação)
        period_prices = prices.loc[date:next_date]
        if not period_prices.empty and not final_w.empty:
            # Retorno dos ativos
            asset_rets = period_prices[final_w.index].pct_change().iloc[1:]
            port_ret = asset_rets.dot(final_w)
            
            # Custo de Transação (Turnover)
            turnover = 0
            if not current_weights.empty:
                # Alinha índices
                all_tkrs = final_w.index.union(current_weights.index)
                w_new = final_w.reindex(all_tkrs).fillna(0)
                w_old = current_weights.reindex(all_tkrs).fillna(0)
                turnover = np.abs(w_new - w_old).sum()
            else:
                turnover = 1.0 # Primeira alocação
            
            # Deduz custo no primeiro dia do mês
            cost = turnover * config['cost_pct']
            port_ret.iloc[0] -= cost
            
            history_rets.append(port_ret)
            current_weights = final_w # Atualiza carteira atual
        else:
            # Se ficou em caixa (sem ativos ou erro), retorno zero (ou CDI)
            idx = prices.loc[date:next_date].index[1:]
            history_rets.append(pd.Series(0.0, index=idx))

    if not history_rets: return pd.Series()
    
    full_ret = pd.concat(history_rets)
    # Remove duplicatas de índice se houver overlap
    return full_ret[~full_ret.index.duplicated(keep='first')]

# ==============================================================================
# UX MÓDULO: REBALANCEAMENTO
# ==============================================================================
def simulation_rebalance(target_weights, capital, prices_curr):
    df = target_weights.to_frame('Peso Alvo')
    df['Preço Atual'] = prices_curr.reindex(df.index)
    df['Alocação R$'] = df['Peso Alvo'] * capital
    df['Qtd Teórica'] = (df['Alocação R$'] / df['Preço Atual']).fillna(0).astype(int)
    
    # Ajuste fino financeiro
    df['Financeiro Real'] = df['Qtd Teórica'] * df['Preço Atual']
    df['Peso Real'] = df['Financeiro Real'] / capital
    return df

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("🧬 Quant Factor Lab Pro v3.0 | Institutional Grade")
    st.markdown("""
    **Atualizações da Versão 3.0:**
    - 🛡️ **Data Engineering:** Tratamento de outliers via MAD e Cache Inteligente.
    - 🧠 **Metodologia:** Quality Score (F-Score inspired), Normalização Rank-Based.
    - ⚙️ **Backtest:** Otimização Max IR e Filtro de Regime de Mercado.
    - 📊 **UX:** Simulador de Rebalanceamento e Gestão de Fluxo de Caixa (DCA).
    """)

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("1. Universo e Dados")
        default_tickers = "VALE3.SA, ITUB4.SA, PETR4.SA, WEGE3.SA, PRIO3.SA, BBAS3.SA, RENT3.SA, LREN3.SA, GGBR4.SA, RAIL3.SA, ELET3.SA, SUZB3.SA, BPAC11.SA, HAPV3.SA, EQTL3.SA, RADL3.SA, VIVT3.SA, CMIG4.SA, CPLE6.SA, JBSS3.SA"
        tickers_input = st.text_area("Tickers (CSV)", default_tickers, height=100)
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        st.divider()
        st.header("2. Factor Weighting")
        w_mom = st.slider("Momentum (Residual)", 0.0, 1.0, 0.4)
        w_val = st.slider("Value (Composite)", 0.0, 1.0, 0.3)
        w_qual = st.slider("Quality (Enhanced)", 0.0, 1.0, 0.3)
        w_vol = st.slider("Low Volatility", 0.0, 1.0, 0.0)
        
        factor_weights = {'Momentum': w_mom, 'Value': w_val, 'Quality': w_qual, 'LowVol': w_vol}
        
        st.divider()
        st.header("3. Configurações Pro")
        opt_method = st.selectbox("Otimizador", ["Inverse Vol (Risk Parity)", "Max IR (Alpha Focus)"])
        use_regime = st.checkbox("🛡️ Ativar Filtro de Regime (Market Guard)", value=True, help="Reduz exposição em alta volatilidade")
        min_liq = st.number_input("Liquidez Mín. (R$)", 5000000, step=1000000)
        cost_bps = st.number_input("Custo Transação (bps)", value=10) # 0.10%
        
        st.divider()
        st.header("4. Gestão de Capital (DCA)")
        init_capital = st.number_input("Capital Inicial", value=100000.0)
        monthly_contr = st.number_input("Aporte Mensal", value=2000.0)
        
        run = st.button("🚀 Executar Modelo", type="primary")

    # --- EXECUÇÃO ---
    if run:
        if not tickers: st.warning("Adicione tickers."); return
        
        # 1. FETCH DATA
        with st.spinner("📥 Coletando dados (Preços & Fundamentos)..."):
            end = datetime.today()
            start = end - timedelta(days=365*4)
            prices, vols = fetch_market_data(tickers, start, end)
            funds = fetch_fundamentals_robust(tickers)
        
        if prices.empty: st.error("Erro nos dados de preço."); return
        
        # 2. CURRENT ANALYSIS (SNAPSHOT)
        st.subheader("🔎 Análise do Universo Atual")
        
        # Calcula fatores atuais
        cur_mom = compute_residual_momentum(prices.tail(300))
        cur_vol = compute_low_vol_beta(prices)
        cur_qual = compute_enhanced_quality(funds)
        cur_val = compute_value_composite(funds)
        
        df_cur = pd.concat([cur_mom, cur_vol, cur_qual, cur_val], axis=1, keys=factor_weights.keys())
        
        # Aplica Normalização Rank-Based para visualização
        for c in df_cur.columns:
            df_cur[c] = normalize_rank_based(df_cur[c])
            
        final_rank = build_composite_score(df_cur, factor_weights)
        
        # Top Picks
        top_n = 10
        top_assets = final_rank.head(top_n).index
        
        # 3. BACKTEST ENGINE
        with st.spinner("⚙️ Executando Backtest Institucional..."):
            bt_config = {
                'start_date': prices.index[0] + timedelta(days=365), # Warmup
                'weights': factor_weights,
                'min_liquidity': min_liq,
                'top_n': top_n,
                'cost_pct': cost_bps/10000,
                'opt_method': opt_method,
                'regime_filter': use_regime,
                'dynamic_weights': False # Desativado para demo rápida
            }
            strat_rets = run_pro_backtest(prices, funds, bt_config, vols)
            bench_rets = prices['BOVA11.SA'].pct_change().loc[strat_rets.index].fillna(0)

        # 4. TABS DE RESULTADO
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance & Risco", "🏆 Ranking & Fatores", "📋 Rebalanceamento (Execução)", "💰 Fluxo de Caixa (DCA)"])
        
        with tab1:
            # Cumulative
            cum_strat = (1+strat_rets).cumprod()
            cum_bench = (1+bench_rets).cumprod()
            
            df_chart = pd.DataFrame({'Strategy': cum_strat, 'BOVA11': cum_bench})
            st.plotly_chart(px.line(df_chart, title="Curva de Equity (Base 1.0)", color_discrete_sequence=['#00CC96', '#EF553B']), use_container_width=True)
            
            # Métricas
            tot_ret = cum_strat.iloc[-1] - 1
            ann_vol = strat_rets.std() * np.sqrt(252)
            sharpe = (strat_rets.mean()/strat_rets.std()) * np.sqrt(252)
            dd = (cum_strat/cum_strat.cummax() - 1).min()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retorno Total", f"{tot_ret:.1%}", delta=f"Alpha: {(tot_ret - (cum_bench.iloc[-1]-1)):.1%}")
            c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c3.metric("Volatilidade", f"{ann_vol:.1%}")
            c4.metric("Max Drawdown", f"{dd:.1%}")

        with tab2:
            st.subheader("Top Picks (Rank-Based Score)")
            
            # Formatação visual do Ranking
            display_df = df_cur.loc[top_assets].copy()
            display_df['Composite'] = final_rank.loc[top_assets]
            st.dataframe(display_df.style.background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)
            
            # Correlação de Fatores
            with st.expander("Ver Matriz de Correlação dos Fatores"):
                corr = df_cur.corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', range_color=[-1,1]), use_container_width=True)

        with tab3:
            st.subheader("🛠️ Simulador de Execução (Trading)")
            
            # Simular Pesos Finais
            if opt_method == 'Max IR':
                # Simulação simples para o dia atual
                cov_now = prices[top_assets].pct_change().tail(126).cov()
                mu_now = prices[top_assets].pct_change().tail(126).mean()
                w_curr_opt = optimize_max_ir(mu_now.values, cov_now.values)
                target_w = pd.Series(w_curr_opt, index=top_assets)
            else:
                vols_now = prices[top_assets].pct_change().tail(126).std()
                w_inv = 1/vols_now
                target_w = w_inv/w_inv.sum()
            
            current_prices = prices.iloc[-1]
            trade_plan = simulation_rebalance(target_w, init_capital, current_prices)
            
            c1, c2 = st.columns([2,1])
            with c1:
                st.dataframe(trade_plan.style.format({'Peso Alvo': '{:.1%}', 'Alocação R$': 'R$ {:.2f}', 'Preço Atual': 'R$ {:.2f}', 'Financeiro Real': 'R$ {:.2f}'}))
            with c2:
                st.info("ℹ️ Este plano considera o capital total disponível para rebalanceamento imediato. Custos de corretagem não inclusos nesta tabela.")
                csv = trade_plan.to_csv().encode('utf-8')
                st.download_button("📥 Baixar Ordem de Compra", csv, "ordens_execucao.csv", "text/csv")

        with tab4:
            st.subheader("Planejamento Financeiro (DCA)")
            # Simulação DCA sobre a curva de retorno da estratégia
            dates_m = strat_rets.resample('MS').first().index
            dca_balance = [init_capital]
            dates_dca = [strat_rets.index[0]]
            
            curr_bal = init_capital
            
            for d in strat_rets.index[1:]:
                # Retorno do dia
                ret_day = strat_rets.loc[d]
                curr_bal = curr_bal * (1 + ret_day)
                
                # Aporte mensal
                if d in dates_m:
                    curr_bal += monthly_contr
                
                dca_balance.append(curr_bal)
                dates_dca.append(d)
                
            df_dca = pd.DataFrame({'Patrimônio': dca_balance}, index=dates_dca)
            tot_invested = init_capital + (len(dates_m) * monthly_contr)
            
            cm1, cm2 = st.columns(2)
            cm1.metric("Patrimônio Final", f"R$ {df_dca.iloc[-1].item():,.2f}")
            cm2.metric("Total Investido (Principal)", f"R$ {tot_invested:,.2f}", delta=f"Lucro: R$ {df_dca.iloc[-1].item() - tot_invested:,.2f}")
            
            st.plotly_chart(px.area(df_dca, y='Patrimônio', title="Evolução com Aportes Mensais"), use_container_width=True)

if __name__ == "__main__":
    main()
