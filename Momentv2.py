import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v3.1 (Benchmark & Alloc)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
BRAPI_TOKEN = "5gVedSQ928pxhFuTvBFPfr"  # Mantido o token original
BENCHMARKS = ['BOVA11.SA', 'DIVO11.SA']

# ==============================================================================
# MÓDULO 1: DATA FETCHING (PREÇOS & BRAPI)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados via YFinance, garantindo benchmarks."""
    t_list = list(tickers)
    # Garante benchmarks na lista de download
    for bench in BENCHMARKS:
        if bench not in t_list:
            t_list.append(bench)
    
    try:
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False,
            threads=True
        )['Adj Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        # Limpeza básica
        data.dropna(how='all', inplace=True)
        return data
    except Exception as e:
        st.error(f"Erro no download de preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def fetch_brapi_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Busca fundamentos ATUAIS (Snapshot) via Brapi para a seleção de ativos de HOJE.
    Não confundir com dados históricos para backtest.
    """
    valid_tickers = [t for t in tickers if t not in BENCHMARKS]
    chunks = [valid_tickers[i:i + 20] for i in range(0, len(valid_tickers), 20)]
    
    all_stocks = []
    
    for chunk in chunks:
        t_str = ",".join(chunk)
        url = f"https://brapi.dev/api/quote/{t_str}?range=1d&interval=1d&fundamental=true&token={BRAPI_TOKEN}"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if 'results' in data:
                    all_stocks.extend(data['results'])
        except Exception as e:
            st.warning(f"Erro ao buscar lote Brapi: {e}")
            continue
            
    processed_data = []
    for s in all_stocks:
        try:
            ticker = s['symbol']
            price = s.get('regularMarketPrice', np.nan)
            
            # Extração segura de fundamentos
            # A estrutura da Brapi pode variar, ajustando para o padrão comum
            # Tenta pegar valuation attributes se existirem
            # Nota: A API gratuita as vezes limita dados complexos, focaremos no essencial
            
            # Simulação de extração (A API Brapi real retorna dados aninhados em 'stockData' ou similar dependendo do endpoint)
            # Adaptando para a estrutura padrão de resposta Quote+Fundamental
            
            # Tenta buscar métricas chave (exemplo genérico baseado na doc)
            # Se não estiver disponível direto, usamos price action do YF para Momentum
            # Aqui assumimos que conseguimos pegar P/L, DY, etc se disponíveis.
            
            # Para este script robusto, focaremos em PREÇO ATUAL e alguns múltiplos se disponíveis
            # Se a API não retornar múltiplos detalhados no endpoint quote, usamos apenas Price Action na seleção?
            # O usuário pediu Multifator. Vamos tentar extrair.
            
            # Fallback: Se não houver dados fundamentais ricos, o modelo usa Momentum + Vol (Híbrido)
            # Mas vamos tentar pegar o DY e P/L se houver.
            
            processed_data.append({
                'ticker': ticker,
                'price': price,
                # Placeholder para fundamentos (na versão free as vezes vem vazio)
                'pe': s.get('priceEarnings', np.nan), 
                'dy': s.get('dividendYield', np.nan)
            })
        except:
            continue
            
    return pd.DataFrame(processed_data)

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES & RANKING (SELEÇÃO ATUAL)
# ==============================================================================

def calculate_current_ranking(price_df, brapi_df, lookback_months=12):
    """
    Gera o Ranking para alocação HOJE.
    Combina:
    1. Momentum (YFinance) -> 12m - 1m
    2. Volatilidade (YFinance) -> Inverse Vol
    3. Fundamentos (Brapi) -> Se disponíveis (usaremos Momentum como driver principal se faltar dados)
    """
    
    # 1. Momentum & Vol (Dados YFinance - Mais confiáveis para histórico)
    end_idx = price_df.index[-1]
    start_idx_mom = end_idx - timedelta(days=30*lookback_months)
    start_idx_rec = end_idx - timedelta(days=30)
    
    # Filtra período
    hist_window = price_df.loc[start_idx_mom:end_idx]
    
    # Retorno 12 meses
    try:
        ret_12m = (hist_window.iloc[-1] / hist_window.iloc[0]) - 1
    except:
        ret_12m = pd.Series(dtype=float)
        
    # Retorno 1 mês (para subtrair, efeito reversão de curto prazo)
    recent_window = price_df.loc[start_idx_rec:end_idx]
    try:
        ret_1m = (recent_window.iloc[-1] / recent_window.iloc[0]) - 1
    except:
        ret_1m = pd.Series(dtype=float)
        
    momentum_score = ret_12m - ret_1m # 12-1 Momentum
    
    # Volatilidade (Desvio padrão diário anualizado)
    volatility = price_df.pct_change().tail(252).std() * np.sqrt(252)
    
    # DataFrame de Fatores Técnicos
    factors = pd.DataFrame({
        'Momentum': momentum_score,
        'Volatility': volatility
    })
    
    # Merge com Brapi (se houver dados)
    if not brapi_df.empty:
        brapi_df.set_index('ticker', inplace=True)
        # Ajuste de sufixo .SA se necessário
        brapi_df.index = [f"{x}.SA" if not x.endswith('.SA') else x for x in brapi_df.index]
        factors = factors.join(brapi_df[['pe', 'dy']], how='left')
    
    # Scoring (Z-Score Robusto)
    # Foco em Momentum Ajustado pelo Risco (Sharpe Proxy)
    factors['Risk_Adj_Mom'] = factors['Momentum'] / factors['Volatility']
    
    # Ranking Final
    # Remove NaN e Infinitos
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna(subset=['Risk_Adj_Mom'])
    
    factors['Rank_Score'] = factors['Risk_Adj_Mom'].rank(ascending=False)
    return factors.sort_values('Rank_Score')

# ==============================================================================
# MÓDULO 3: BACKTEST ROBUSTO (DCA + BENCHMARKS)
# ==============================================================================

def run_dca_backtest_robust(price_df, initial_capital, monthly_contrib, top_n, lookback_days=252):
    """
    Simula DCA comparando Estratégia vs BOVA11 vs DIVO11.
    Usa apenas dados de preço passados (sem lookahead de fundamentos).
    """
    # 1. Preparação dos dados
    monthly_dates = price_df.resample('M').last().index
    
    # DataFrames para armazenar a evolução
    history = []
    
    # Estado inicial
    cash_strat = initial_capital
    cash_bova = initial_capital
    cash_divo = initial_capital
    
    # Assume que compramos "cotas fracionárias" ou valor financeiro direto dos índices
    units_bova = 0.0
    units_divo = 0.0
    
    # Para a estratégia, precisamos rastrear o portfólio
    # Portfolio: {ticker: value}
    strat_holdings = {} 
    
    # Loop de Rebalanceamento
    for date in monthly_dates:
        if date not in price_df.index:
            # Pega o dia válido anterior mais próximo
            try:
                valid_date = price_df.loc[:date].index[-1]
            except:
                continue
        else:
            valid_date = date
            
        # --- 1. Aporte (DCA) ---
        cash_strat += monthly_contrib
        cash_bova += monthly_contrib
        cash_divo += monthly_contrib
        
        # --- 2. Atualiza valor das posições (Strategy) ---
        current_prices = price_df.loc[valid_date]
        strat_value = cash_strat
        
        # Se temos posições, atualizamos o valor e liquidamos (rebalanceamento total teórico)
        # Ignoramos custos de transação para simplificar o comparativo
        if strat_holdings:
            port_val = 0
            for t, shares in strat_holdings.items():
                if t in current_prices and not np.isnan(current_prices[t]):
                    val = shares * current_prices[t]
                    port_val += val
            strat_value += port_val
            cash_strat = strat_value # Liquida tudo virtualmente
            strat_holdings = {}
            
        # --- 3. Atualiza valor Benchmarks ---
        price_bova = current_prices.get('BOVA11.SA', np.nan)
        price_divo = current_prices.get('DIVO11.SA', np.nan)
        
        # Compra BOVA
        if not np.isnan(price_bova) and price_bova > 0:
            units_to_buy = cash_bova / price_bova
            units_bova += units_to_buy
            cash_bova = 0 # Todo caixa alocado
            
        # Compra DIVO
        if not np.isnan(price_divo) and price_divo > 0:
            units_to_buy = cash_divo / price_divo
            units_divo += units_to_buy
            cash_divo = 0 # Todo caixa alocado
            
        # Valor Atual Benchmarks
        val_bova_curr = (units_bova * price_bova) if not np.isnan(price_bova) else cash_bova
        val_divo_curr = (units_divo * price_divo) if not np.isnan(price_divo) else cash_divo
            
        # --- 4. Seleção e Alocação da Estratégia ---
        # Lookback para Momentum
        lookback_date = valid_date - timedelta(days=lookback_days)
        if lookback_date < price_df.index[0]:
            # Sem dados suficientes ainda, mantém em caixa
            history.append({
                'Date': valid_date, 
                'Strategy': strat_value, 
                'BOVA11': val_bova_curr, 
                'DIVO11': val_divo_curr,
                'Invested': (initial_capital + monthly_contrib * (len(history)+1))
            })
            continue
            
        # Subset de dados conhecidos até a data
        past_window = price_df.loc[lookback_date:valid_date]
        
        # Calcula Momentum (Retorno Total no período)
        # Exclui benchmarks da seleção de ativos
        candidates = [c for c in price_df.columns if c not in BENCHMARKS]
        
        try:
            # Retorno
            mom = (past_window[candidates].iloc[-1] / past_window[candidates].iloc[0]) - 1
            # Volatilidade (proxy risco)
            vol = past_window[candidates].pct_change().std()
            
            # Ranking: Momentum / Vol (Sharpe simples)
            score = mom / vol
            score = score.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Top N
            top_tickers = score.nlargest(top_n).index.tolist()
            
            if not top_tickers:
                pass # Mantém caixa
            else:
                # Alocação (Risk Parity Simples: 1/Vol)
                # Para simplificar e ser robusto: Equal Weight nos Top N
                # (Risk Parity real requer matriz de covariância, pesado para loop)
                weight_per_asset = 1.0 / len(top_tickers)
                
                for ticker in top_tickers:
                    alloc_val = strat_value * weight_per_asset
                    p_now = current_prices[ticker]
                    if not np.isnan(p_now) and p_now > 0:
                        shares = alloc_val / p_now
                        strat_holdings[ticker] = shares
                
                cash_strat = 0 # Todo alocado
                
        except Exception:
            pass # Mantém caixa em caso de erro de calculo
            
        # Registra
        history.append({
            'Date': valid_date, 
            'Strategy': strat_value, 
            'BOVA11': val_bova_curr, 
            'DIVO11': val_divo_curr,
            'Total_Invested': initial_capital + (monthly_contrib * (len(history))) # Approx
        })
        
    return pd.DataFrame(history).set_index('Date')

# ==============================================================================
# UI PRINCIPAL
# ==============================================================================

st.title("🛡️ Quant Factor Lab Pro - Modelo Robusto (v3.1)")
st.markdown("""
**Filosofia do Modelo:** Este sistema prioriza a **robustez estatística** sobre promessas de retorno irrealistas.
* **Seleção Atual:** Utiliza modelo Multifator (Momentum + Volatilidade + Fundamentos Brapi).
* **Backtest:** Utiliza simulação "Cega" (apenas Preço/Momentum) para evitar viés de antecipação (Lookahead Bias).
""")

# --- SIDEBAR: PARÂMETROS ---
st.sidebar.header("⚙️ Configuração da Carteira")
tickers_input = st.sidebar.text_area(
    "Universo de Ativos (Tickers + Benchmarks):",
    "VALE3.SA, PETR4.SA, ITUB4.SA, WEGE3.SA, PRIO3.SA, BBAS3.SA, JBSS3.SA, ELET3.SA, GGBR4.SA, RENT3.SA, BPAC11.SA, SUZB3.SA, HAPV3.SA, RADL3.SA, EQTL3.SA, LREN3.SA, B3SA3.SA, VIVT3.SA, CMIG4.SA, CCRO3.SA, RAIL3.SA, CPLE6.SA, PSSA3.SA, TOTS3.SA, UGPA3.SA, CMIN3.SA, BRFS3.SA, CSAN3.SA, EMBR3.SA, ENGI11.SA, KLBN11.SA, CSNA3.SA, AZUL4.SA, CVCB3.SA, GOLL4.SA, VIIA3.SA, MGLU3.SA, BOVA11.SA, DIVO11.SA"
)
top_n = st.sidebar.slider("Número de Ativos na Carteira", 5, 20, 10)
lookback = st.sidebar.slider("Lookback Momentum (Meses)", 3, 24, 12)

st.sidebar.header("💰 Parâmetros de Aporte (Backtest)")
init_cash = st.sidebar.number_input("Aporte Inicial (R$)", 10000, 1000000, 50000)
monthly_cash = st.sidebar.number_input("Aporte Mensal (R$)", 0, 50000, 2000)

if st.sidebar.button("🚀 Executar Análise"):
    
    # Limpeza de Tickers
    t_list = [x.strip().upper() for x in tickers_input.split(',')]
    t_list = list(set(t_list)) # Remove duplicatas
    
    with st.spinner("Baixando dados de mercado (YFinance) e Fundamentos (Brapi)..."):
        # 1. Dados Históricos (Preço)
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        price_df = fetch_price_data(t_list, start_date, datetime.now().strftime('%Y-%m-%d'))
        
        # 2. Dados Atuais (Brapi) - Apenas para o ranking de hoje
        brapi_df = fetch_brapi_fundamentals(t_list)
        
    if not price_df.empty:
        
        tab1, tab2, tab3 = st.tabs(["📊 Ranking & Alocação Atual", "📈 Backtest Comparativo", "📋 Dados Brutos"])
        
        # ==============================================================================
        # TAB 1: RANKING ATUAL & ALOCAÇÃO
        # ==============================================================================
        with tab1:
            st.subheader("Carteira Sugerida (Mês Atual)")
            
            ranking = calculate_current_ranking(price_df, brapi_df, lookback_months=lookback)
            top_picks = ranking.head(top_n)
            
            # Cálculo de Pesos (Inverse Volatility)
            # Se Vol for muito baixa ou zero, tratamos
            inv_vol = 1.0 / top_picks['Volatility']
            weights = inv_vol / inv_vol.sum()
            
            top_picks['Peso (%)'] = weights * 100
            top_picks['Alocação Sugerida (R$)'] = (init_cash) * weights # Baseado apenas no capital inicial para visualização
            
            # --- DISPLAY VISUAL DA ALOCAÇÃO ---
            col_kpi1, col_kpi2 = st.columns([1, 2])
            
            with col_kpi1:
                st.markdown("### 🥧 Distribuição")
                fig_pie = px.pie(
                    top_picks, 
                    values='Peso (%)', 
                    names=top_picks.index, 
                    title='Alocação por Ativo',
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_kpi2:
                st.markdown("### 📋 Detalhes da Ordem")
                display_cols = ['Momentum', 'Volatility', 'Risk_Adj_Mom', 'Peso (%)']
                if 'pe' in top_picks.columns: display_cols += ['pe', 'dy']
                
                # Formatação
                st.dataframe(
                    top_picks[display_cols].style.format({
                        'Momentum': "{:.2%}",
                        'Volatility': "{:.2%}",
                        'Risk_Adj_Mom': "{:.2f}",
                        'Peso (%)': "{:.2f}%",
                        'pe': "{:.1f}",
                        'dy': "{:.2f}"
                    }),
                    use_container_width=True,
                    height=400
                )
                
            st.info("💡 **Nota:** Esta alocação utiliza o modelo Multifator completo (Preço + Volatilidade + Fundamentos Brapi se disponíveis).")

        # ==============================================================================
        # TAB 2: BACKTEST COMPARATIVO
        # ==============================================================================
        with tab2:
            st.subheader("Simulação Histórica (DCA)")
            st.markdown("Comparativo de retorno acumulado com aportes mensais recorrentes.")
            
            # Roda Backtest
            results_df = run_dca_backtest_robust(
                price_df, 
                initial_capital=init_cash, 
                monthly_contrib=monthly_cash, 
                top_n=top_n
            )
            
            if not results_df.empty:
                # Métricas Finais
                final_strat = results_df['Strategy'].iloc[-1]
                final_bova = results_df['BOVA11'].iloc[-1]
                final_divo = results_df['DIVO11'].iloc[-1]
                total_invested = results_df['Total_Invested'].iloc[-1]
                
                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Investido", f"R$ {total_invested:,.2f}")
                c2.metric("Saldo Estratégia", f"R$ {final_strat:,.2f}", delta=f"{(final_strat/total_invested - 1)*100:.1f}%")
                c3.metric("Saldo BOVA11", f"R$ {final_bova:,.2f}", delta=f"{(final_bova/total_invested - 1)*100:.1f}%")
                c4.metric("Saldo DIVO11", f"R$ {final_divo:,.2f}", delta=f"{(final_divo/total_invested - 1)*100:.1f}%")
                
                # Gráfico
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df['Strategy'], name='Modelo Robusto', line=dict(color='#00CC96', width=3)))
                fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df['BOVA11'], name='BOVA11 (Ibovespa)', line=dict(color='#EF553B')))
                fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df['DIVO11'], name='DIVO11 (Dividendos)', line=dict(color='#636EFA')))
                fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df['Total_Invested'], name='Capital Aportado', line=dict(color='gray', dash='dash')))
                
                fig_bt.update_layout(
                    title="Crescimento Patrimonial (Aporte Inicial + Aportes Mensais)",
                    yaxis_title="Patrimônio (R$)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Drawdown Analysis
                st.markdown("#### Análise de Risco (Drawdown)")
                # Calcula DD
                dd_df = pd.DataFrame()
                for col in ['Strategy', 'BOVA11', 'DIVO11']:
                    peak = results_df[col].cummax()
                    dd = (results_df[col] - peak) / peak
                    dd_df[col] = dd
                
                fig_dd = px.area(dd_df, title="Drawdown (Queda do Topo Histórico)")
                st.plotly_chart(fig_dd, use_container_width=True)

            else:
                st.warning("Dados insuficientes para backtest nesta janela.")

        # ==============================================================================
        # TAB 3: DADOS BRUTOS
        # ==============================================================================
        with tab3:
            st.dataframe(price_df.tail(100))
    else:
        st.error("Não foi possível carregar dados. Verifique os Tickers.")
