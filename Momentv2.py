import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

# Configuração da Página
st.set_page_config(page_title="Painel Quantitativo de Ações BR", layout="wide")

# --- 1. Definição do Universo e Inputs ---
DEFAULT_TICKERS = [
    'ITUB3.SA', 'TOTS3.SA', 'MDIA3.SA', 'TAEE3.SA', 'BBSE3.SA', 'WEGE3.SA', 
    'PSSA3.SA', 'EGIE3.SA', 'B3SA3.SA', 'VIVT3.SA', 'AGRO3.SA', 'PRIO3.SA', 
    'BBAS3.SA', 'BPAC11.SA', 'SBSP3.SA', 'SAPR4.SA', 'CMIG3.SA', 'UNIP6.SA', 'FRAS3.SA'
]
BENCHMARK = "^BVSP" # Ibovespa

# --- 2. Funções de Coleta e Cálculo ---

@st.cache_data
def get_stock_data(tickers, benchmark, period="5y"):
    """Coleta dados de preço ajustado e dados fundamentais básicos."""
    data = yf.download(tickers + [benchmark], period=period, group_by='ticker', auto_adjust=True)
    
    # Estrutura para preços de fechamento
    close_prices = pd.DataFrame()
    for t in tickers:
        try:
            close_prices[t] = data[t]['Close']
        except KeyError:
            pass
    
    try:
        close_prices['BENCHMARK'] = data[benchmark]['Close']
    except KeyError:
         close_prices['BENCHMARK'] = data[benchmark]['Close'] # Retry logic usually handled internally
            
    return close_prices.dropna(how='all')

@st.cache_data
def get_fundamentals(tickers):
    """Coleta indicadores fundamentalistas (Snapshot Atual)."""
    # Nota: Em produção, usar API paga (Economatica/Bloomberg) para evitar lentidão e dados faltantes.
    metrics = []
    
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            
            # Tratamento de erros para dados faltantes
            def get_val(key):
                return info.get(key, np.nan)

            metrics.append({
                'Ticker': t,
                'ROE': get_val('returnOnEquity'),
                'Margem Líquida': get_val('profitMargins'),
                'Dívida/EBITDA': get_val('debtToEquity'), # Proxy simples
                'Crescimento Lucro': get_val('earningsGrowth'),
                'P/L': get_val('trailingPE'),
                'P/VP': get_val('priceToBook'),
                'EV/EBITDA': get_val('enterpriseToEbitda'), # Usando EBITDA como proxy de EBIT comum na API free
                'Dividend Yield': get_val('dividendYield')
            })
        except:
            metrics.append({'Ticker': t})
            
    df = pd.DataFrame(metrics)
    return df.set_index('Ticker')

def calculate_residual_momentum(prices_df, lookback_months=12, exclude_recent_months=1):
    """
    Calcula o Momentum Residual.
    Regressão do retorno da ação contra o benchmark. O resíduo é o alpha ajustado.
    """
    momentum_scores = {}
    
    # Definir janela (aprox 21 dias uteis por mês)
    window = lookback_months * 21
    skip = exclude_recent_months * 21
    
    if len(prices_df) < window + skip:
        return pd.Series()

    # Preços defasados (para evitar momentum de curtíssimo prazo de reversão)
    recent_prices = prices_df.iloc[:-skip]
    
    for col in prices_df.columns:
        if col == 'BENCHMARK': continue
        
        # Série temporal dos últimos 'window' dias
        asset_ret = recent_prices[col].pct_change().tail(window).dropna()
        bench_ret = recent_prices['BENCHMARK'].pct_change().tail(window).dropna()
        
        # Alinhar dados
        common_idx = asset_ret.index.intersection(bench_ret.index)
        if len(common_idx) < window * 0.8:
            momentum_scores[col] = np.nan
            continue
            
        y = asset_ret.loc[common_idx]
        x = bench_ret.loc[common_idx]
        
        # Regressão Linear (Slope e Intercept)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        # Calcular Resíduos (Retorno anormal)
        expected_ret = intercept + slope * x
        residuals = y - expected_ret
        
        # Score = Média do resíduo / Desvio Padrão do resíduo (Information Ratio do Momentum)
        score = residuals.mean() / residuals.std()
        momentum_scores[col] = score
        
    return pd.Series(momentum_scores, name='Residual Momentum')

# --- 3. Interface do Streamlit ---

st.sidebar.header("⚙️ Configurações do Painel")
selected_tickers = st.sidebar.multiselect("Universo de Ações", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
period_input = st.sidebar.selectbox("Período de Análise", ["1y", "3y", "5y"], index=1)

if st.sidebar.button("Gerar Painel"):
    with st.spinner('Baixando dados de mercado e calculando fatores...'):
        
        # 1. Dados
        prices = get_stock_data(selected_tickers, BENCHMARK, period="5y") # Baixa 5y para ter histórico suficiente
        df_funds = get_fundamentals(selected_tickers)
        
        # 2. Cálculos de Fatores
        # Normalização e Tratamento
        df_funds['ROE'] = df_funds['ROE'].fillna(0) * 100
        df_funds['Margem Líquida'] = df_funds['Margem Líquida'].fillna(0) * 100
        df_funds['Dividend Yield'] = df_funds['Dividend Yield'].fillna(0) * 100
        
        # Inverter métricas onde menor é melhor
        df_calc = df_funds.copy()
        
        # Calcular Scores (Rank Percentil) - 0 a 1
        # Quality
        q_score = (df_calc['ROE'].rank(pct=True) + 
                   df_calc['Margem Líquida'].rank(pct=True) + 
                   df_calc['Crescimento Lucro'].rank(pct=True) - 
                   df_calc['Dívida/EBITDA'].rank(pct=True)) # Menor dívida é melhor
        
        # Value
        # Para P/L, EV/EBITDA e P/VP, menor é melhor, então invertemos o rank
        v_score = (df_calc['Dividend Yield'].rank(pct=True) + 
                   (1 - df_calc['P/L'].rank(pct=True)) + 
                   (1 - df_calc['P/VP'].rank(pct=True)) + 
                   (1 - df_calc['EV/EBITDA'].rank(pct=True)))
        
        # Momentum Residual (12 meses, excluindo último mês)
        mom_resid = calculate_residual_momentum(prices, lookback_months=12, exclude_recent_months=1)
        
        # Consolidar Tabela Mestra
        master_df = df_funds.copy()
        master_df['Residual Momentum'] = mom_resid
        master_df['Quality Score'] = q_score
        master_df['Value Score'] = v_score
        
        # Normalizar Scores finais (0 a 100)
        master_df['Quality Rank'] = (master_df['Quality Score'].rank(pct=True) * 100).round(1)
        master_df['Value Rank'] = (master_df['Value Score'].rank(pct=True) * 100).round(1)
        master_df['Momentum Rank'] = (master_df['Residual Momentum'].rank(pct=True) * 100).round(1)
        
        master_df['Score Geral'] = ((master_df['Quality Rank'] + master_df['Value Rank'] + master_df['Momentum Rank'])/3).round(1)

        # --- Layout do Dashboard ---
        
        st.title(f"📊 Painel de Decisão Quantitativa: {len(selected_tickers)} Ativos")
        st.markdown("---")

        # TAB 1: Rankings e Fatores
        tab1, tab2, tab3 = st.tabs(["🏆 Rankings e Fatores", "📈 Backtest e Performance", "🧠 Insights e Correlações"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Picks (Score Combinado)")
                top_picks = master_df.sort_values(by='Score Geral', ascending=False).head(10)
                st.dataframe(top_picks[['Quality Rank', 'Value Rank', 'Momentum Rank', 'Score Geral']], use_container_width=True)
            
            with col2:
                st.subheader("Múltiplos e Fundamentos")
                st.dataframe(master_df[['P/L', 'ROE', 'Dividend Yield', 'Residual Momentum']].style.format("{:.2f}"), use_container_width=True)
                
            st.markdown("### Detalhamento por Fator")
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Top Quality:** {master_df.sort_values(by='Quality Rank', ascending=False).index[0]}")
            c2.success(f"**Top Value:** {master_df.sort_values(by='Value Rank', ascending=False).index[0]}")
            c3.warning(f"**Top Momentum:** {master_df.sort_values(by='Momentum Rank', ascending=False).index[0]}")

        # TAB 2: Backtest
        with tab2:
            st.markdown(f"#### Backtest Comparativo ({period_input}) vs Ibovespa")
            
            # Definir data de corte
            days_map = {"1y": 252, "3y": 252*3, "5y": 252*5}
            lookback = days_map.get(period_input, 252)
            
            bt_prices = prices.iloc[-lookback:].copy()
            bt_ret = bt_prices.pct_change().dropna()
            
            # Estratégias Simplificadas (Equally Weighted nos Top Tier)
            # Nota: Isso é um backtest estático (seleção atual mantida para trás) para fins de demonstração
            top_q = master_df.sort_values('Quality Rank', ascending=False).head(int(len(master_df)*0.3)).index
            top_v = master_df.sort_values('Value Rank', ascending=False).head(int(len(master_df)*0.3)).index
            top_m = master_df.sort_values('Momentum Rank', ascending=False).head(int(len(master_df)*0.4)).index
            top_combo = master_df.sort_values('Score Geral', ascending=False).head(5).index
            
            # Calcular equity curves
            cum_ret = (1 + bt_ret).cumprod()
            
            strategies = {
                'Ibovespa': cum_ret['BENCHMARK'],
                'Top Quality (30%)': cum_ret[top_q].mean(axis=1),
                'Top Value (30%)': cum_ret[top_v].mean(axis=1),
                'Top Momentum (40%)': cum_ret[top_m].mean(axis=1),
                'Top Combo (Top 5)': cum_ret[top_combo].mean(axis=1)
            }
            
            df_equity = pd.DataFrame(strategies)
            df_equity = df_equity / df_equity.iloc[0] * 100 # Base 100
            
            # Gráfico
            fig_bt = px.line(df_equity, title="Curva de Retorno Acumulado (Base 100)")
            st.plotly_chart(fig_bt, use_container_width=True)
            
            # Tabela de Métricas
            metrics_bt = []
            for name, series in strategies.items():
                total_ret = (series.iloc[-1] / series.iloc[0]) - 1
                vol = series.pct_change().std() * (252**0.5)
                sharpe = (total_ret) / vol if vol > 0 else 0
                
                # Drawdown
                roll_max = series.cummax()
                drawdown = (series - roll_max) / roll_max
                max_dd = drawdown.min()
                
                metrics_bt.append({
                    'Estratégia': name,
                    'Retorno Total': f"{total_ret*100:.2f}%",
                    'Volatilidade (aa)': f"{vol*100:.2f}%",
                    'Sharpe Ratio': f"{sharpe:.2f}",
                    'Max Drawdown': f"{max_dd*100:.2f}%"
                })
            
            st.table(pd.DataFrame(metrics_bt).set_index('Estratégia'))
            st.caption("*Nota: O backtest de Quality e Value assume a manutenção da carteira atual (viés de look-ahead) para demonstrar a qualidade dos ativos selecionados hoje. O Momentum é calculado historicamente.*")

        # TAB 3: Visualizações Avançadas
        with tab3:
            col_hm1, col_hm2 = st.columns([2, 1])
            
            with col_hm1:
                st.subheader("Mapa de Correlação de Retornos")
                corr = bt_ret.drop(columns=['BENCHMARK'], errors='ignore').corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col_hm2:
                st.subheader("Dispersão: Valor vs Qualidade")
                fig_scat = px.scatter(master_df, x='Value Rank', y='Quality Rank', hover_name=master_df.index, 
                                      color='Score Geral', size='Momentum Rank', 
                                      title="Quadrantes de Seleção")
                fig_scat.add_hline(y=50, line_dash="dash", line_color="gray")
                fig_scat.add_vline(x=50, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_scat, use_container_width=True)

else:
    st.info("👋 Olá! Clique no botão na barra lateral para carregar os dados e gerar o painel.")
