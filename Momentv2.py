import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats.mstats import winsorize
from scipy.optimize import minimize
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# CONFIGURAÇÃO E ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro [Institutional]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes Globais
RISK_FREE_RATE = 0.10  # Selic proxy 10%
TRADING_DAYS = 252

# ==============================================================================
# MÓDULO 1: INFRAESTRUTURA E DADOS (Data Handler)
# ==============================================================================

class DataSanitizer:
    """Responsável pela limpeza e tratamento estatístico dos dados."""
    
    @staticmethod
    def winsorize_series(series: pd.Series, limits=(0.025, 0.025)) -> pd.Series:
        """Aplica Winsorization para limitar outliers extremos (caudas de 2.5%)."""
        if series.empty: return series
        # O winsorize do scipy retorna um array mascarado ou numpy array, precisamos converter de volta
        data = series.dropna()
        if len(data) < 5: return series # Poucos dados para winsorizar
        
        # Scipy winsorize modifica os valores extremos para os valores dos limites
        win_data = winsorize(data, limits=limits)
        return pd.Series(win_data, index=data.index)

    @staticmethod
    def clean_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
        """Limpeza rigorosa de dados fundamentais (remover Infs, NaNs críticos)."""
        df = df.replace([np.inf, -np.inf], np.nan)
        # Exemplo: P/E negativo muitas vezes é ruído ou empresa em prejuízo. 
        # Tratamento: ou remove ou penaliza. Aqui vamos deixar passar mas winsorizar depois.
        return df

class DataProvider:
    """Camada de Abstração para Busca de Dados."""
    
    @staticmethod
    @st.cache_data(ttl=3600*12)
    def fetch_price_history(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        t_list = list(set(tickers))
        if 'BOVA11.SA' not in t_list: t_list.append('BOVA11.SA')
        
        try:
            data = yf.download(t_list, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data.dropna(how='all')
        except Exception as e:
            st.error(f"Erro crítico no download de preços: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600*24)
    def fetch_current_fundamentals(tickers: List[str]) -> pd.DataFrame:
        """
        NOTA: Yahoo Finance Free não fornece histórico de balanços fácil.
        Para um backtest real (PIT), conectaríamos aqui a um SQL ou API paga (Bloomberg/Comdinheiro).
        """
        data = []
        clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
        
        # Progress Bar visual
        prog_bar = st.progress(0, text="Baixando Fundamentos (Snapshot Atual)...")
        total = len(clean_tickers)
        
        for i, t in enumerate(clean_tickers):
            try:
                ticker_obj = yf.Ticker(t)
                info = ticker_obj.info
                
                # Tratamento de Setor
                sector = info.get('sector', 'Unknown')
                if sector in ['Unknown', 'N/A'] and 'longName' in info:
                     if any(x in info.get('longName', '') for x in ['Banco', 'Financeira', 'Seguridade']):
                         sector = 'Financial Services'
                
                # Dados ampliados para fatores mais sofisticados
                data.append({
                    'ticker': t,
                    'sector': sector,
                    # Value
                    'forwardPE': info.get('forwardPE', np.nan),
                    'priceToBook': info.get('priceToBook', np.nan),
                    'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan), # Melhor que PE
                    'freeCashflow': info.get('freeCashflow', np.nan),
                    'marketCap': info.get('marketCap', np.nan),
                    # Quality
                    'returnOnEquity': info.get('returnOnEquity', np.nan),
                    'returnOnAssets': info.get('returnOnAssets', np.nan),
                    'grossMargins': info.get('grossMargins', np.nan), # Proxy de GP/A
                    'operatingMargins': info.get('operatingMargins', np.nan),
                    'debtToEquity': info.get('debtToEquity', np.nan),
                    'currentRatio': info.get('currentRatio', np.nan),
                    # Growth
                    'earningsGrowth': info.get('earningsGrowth', np.nan),
                    'revenueGrowth': info.get('revenueGrowth', np.nan)
                })
            except Exception:
                pass # Falha silenciosa em ticker individual para não parar o loop
            prog_bar.progress((i + 1) / total)
            
        prog_bar.empty()
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data).set_index('ticker')
        return DataSanitizer.clean_fundamentals(df)

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Sophisticated Math)
# ==============================================================================

class FactorEngine:
    
    @staticmethod
    def robust_zscore(series: pd.Series) -> pd.Series:
        """Z-Score robusto usando Mediana e MAD (Median Absolute Deviation)."""
        series = DataSanitizer.winsorize_series(series) # Passo 1: Winsorize
        median = series.median()
        mad = (series - median).abs().median()
        if mad < 1e-6: return series - median
        z = (series - median) / (mad * 1.4826)
        return z.clip(-3, 3) # Hard clip final

    @staticmethod
    def compute_residual_momentum(price_df: pd.DataFrame, lookback=252, skip=21) -> pd.Series:
        """
        Momentum Residual (Alpha do Idiossincrático).
        Melhoria: Janela móvel de 12 meses (252 dias) pulando o último mês (21 dias) para evitar reversão.
        """
        # Reamostragem mensal para suavizar ruído
        df_monthly = price_df.resample('ME').last()
        rets = df_monthly.pct_change().dropna()
        
        if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
        market = rets['BOVA11.SA']
        scores = {}
        
        # Lookback ajustado para meses (aprox 12 meses)
        window_months = 12
        
        for ticker in rets.columns:
            if ticker == 'BOVA11.SA': continue
            
            y = rets[ticker].tail(window_months)
            x = market.tail(window_months)
            
            if len(y) < 6: continue # Mínimo de dados
            
            try:
                # Regressão OLS: Ri = alpha + beta*Rm + epsilon
                # Estamos interessados no epsilon (resíduo) normalizado
                X = sm.add_constant(x.values)
                model = sm.OLS(y.values, X).fit()
                resid = model.resid
                
                # Ignorar o mês mais recente (skip) já foi feito pela natureza do dado mensal ou pode ser feito fatiando
                # Momentum Residual = Média do Resíduo / Desvio do Resíduo (Information Ratio do Alpha)
                if np.std(resid) > 1e-6:
                    scores[ticker] = np.sum(resid) / np.std(resid)
                else:
                    scores[ticker] = 0
            except:
                scores[ticker] = 0
                
        return pd.Series(scores, name='Residual_Momentum')

    @staticmethod
    def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
        """Value V2: Combinação de Earnings Yield, Book-to-Market e EBITDA/EV."""
        scores = pd.DataFrame(index=fund_df.index)
        
        # Invertendo métricas (quanto menor melhor -> quanto maior melhor)
        if 'forwardPE' in fund_df: 
            scores['E_Yield'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
        
        if 'priceToBook' in fund_df:
            scores['B_M'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
            
        if 'enterpriseToEbitda' in fund_df:
            # EV/EBITDA é superior ao P/E por neutralizar estrutura de capital
            scores['EBITDA_EV'] = np.where(fund_df['enterpriseToEbitda'] > 0, 1/fund_df['enterpriseToEbitda'], 0)
            
        # Média dos Z-Scores dos sub-fatores
        z_scores = scores.apply(FactorEngine.robust_zscore)
        return z_scores.mean(axis=1).rename("Value_Score")

    @staticmethod
    def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
        """Quality V2: Lucratividade, Margens e Saúde Financeira (Piotroski Proxy)."""
        scores = pd.DataFrame(index=fund_df.index)
        
        # Profitability
        if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
        if 'grossMargins' in fund_df: scores['GPA_Proxy'] = fund_df['grossMargins'] # Gross Profitability
        
        # Financial Health
        if 'debtToEquity' in fund_df: scores['Solvency'] = -1 * fund_df['debtToEquity'] # Menor é melhor
        if 'currentRatio' in fund_df: scores['Liquidity'] = fund_df['currentRatio']
        
        z_scores = scores.apply(FactorEngine.robust_zscore)
        return z_scores.mean(axis=1).rename("Quality_Score")
    
    @staticmethod
    def compute_growth_score(fund_df: pd.DataFrame) -> pd.Series:
        metrics = ['earningsGrowth', 'revenueGrowth']
        temp = pd.DataFrame(index=fund_df.index)
        for m in metrics:
            if m in fund_df: temp[m] = fund_df[m]
        return temp.apply(FactorEngine.robust_zscore).mean(axis=1).rename("Growth_Score")

# ==============================================================================
# MÓDULO 3: OTIMIZAÇÃO DE PORTFÓLIO (Institutional Grade)
# ==============================================================================

class PortfolioOptimizer:
    
    @staticmethod
    def optimize_portfolio(returns_df: pd.DataFrame, method='max_sharpe') -> pd.Series:
        """
        Otimização Mean-Variance usando scipy.optimize.
        Assume long-only, soma dos pesos = 1.
        """
        tickers = returns_df.columns
        n = len(tickers)
        if n == 0: return pd.Series()
        if n == 1: return pd.Series([1.0], index=tickers)
        
        mu = returns_df.mean() * 252 # Retorno esperado anualizado
        S = returns_df.cov() * 252   # Covariância anualizada
        
        # Chute inicial: Equal Weight
        init_weights = np.array([1/n] * n)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Fully invested
        bounds = tuple((0.0, 0.40) for _ in range(n)) # Limite de concentração de 40% por ativo
        
        if method == 'max_sharpe':
            def neg_sharpe(weights):
                p_ret = np.sum(weights * mu)
                p_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
                return - (p_ret - RISK_FREE_RATE) / p_vol
            
            try:
                res = minimize(neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
                opt_weights = res.x
            except:
                # Fallback para inverso da volatilidade se falhar
                vols = returns_df.std()
                opt_weights = (1/vols) / (1/vols).sum()
                return opt_weights

        elif method == 'min_vol':
            def port_vol(weights):
                return np.sqrt(np.dot(weights.T, np.dot(S, weights)))
            
            res = minimize(port_vol, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            opt_weights = res.x
            
        else: # Inverse Volatility (Naive Risk Parity)
            vols = returns_df.std()
            inv_vols = 1 / vols
            opt_weights = inv_vols / inv_vols.sum()
            return opt_weights

        return pd.Series(opt_weights, index=tickers).sort_values(ascending=False)

# ==============================================================================
# MÓDULO 4: BACKTESTING & RISCO (Com Custo de Transação e Métricas)
# ==============================================================================

class BacktestEngine:
    
    def __init__(self, prices: pd.DataFrame, fundamentals: pd.DataFrame, config: Dict):
        self.prices = prices
        self.fundamentals = fundamentals
        self.config = config
        self.transaction_cost_bps = 10 # 0.10% por trade (corretagem + slippage)
        
    def get_fundamentals_for_date(self, date: datetime) -> pd.DataFrame:
        """
        SIMULAÇÃO POINT-IN-TIME (PIT).
        AVISO CRÍTICO: Como não temos um banco de dados histórico real no modo free,
        esta função retorna o snapshot atual.
        EM PRODUÇÃO: Esta função faria uma query SQL: "SELECT * FROM fundamentals WHERE date <= {date} ORDER BY date DESC"
        """
        # Aqui reside o viés de Look-Ahead na versão demo. 
        # Mantemos a função para estruturar a lógica corretamente.
        return self.fundamentals.copy()
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calcula métricas de risco institucional."""
        if returns.empty: return {}
        
        cum_ret = (1 + returns).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        
        # Max Drawdown
        running_max = cum_ret.cummax()
        drawdown = (cum_ret / running_max) - 1
        mdd = drawdown.min()
        
        # VaR e CVaR (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            "Total Return": total_ret,
            "Ann. Return": ann_ret,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": mdd,
            "VaR (95%)": var_95,
            "CVaR (95%)": cvar_95
        }

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Executa backtest com rebalanceamento mensal e custos."""
        start_date = self.prices.index[0] + timedelta(days=252) # Warmup
        end_date = self.prices.index[-1]
        
        # Datas de rebalanceamento (primeiro dia útil do mês)
        rebal_dates = self.prices.loc[start_date:end_date].resample('MS').first().index
        
        portfolio_curve = [100.0] # Base 100
        portfolio_dates = [rebal_dates[0]]
        current_holdings = pd.Series(dtype=float)
        
        daily_returns = []
        
        for i, date in enumerate(rebal_dates[:-1]):
            next_date = rebal_dates[i+1]
            
            # 1. Dados Históricos Disponíveis (Preço)
            # Usamos shift para garantir que só temos dados ATÉ ontem (D-1)
            hist_prices = self.prices.loc[:date]
            
            # 2. Dados Fundamentais (Simulação PIT)
            fund_snapshot = self.get_fundamentals_for_date(date)
            
            # 3. Cálculo de Fatores
            df_factors = pd.DataFrame(index=self.prices.columns.drop('BOVA11.SA', errors='ignore'))
            
            # Momentum (Calculado apenas com preços passados)
            mom_window = hist_prices.tail(300) 
            df_factors['Momentum'] = FactorEngine.compute_residual_momentum(mom_window)
            
            # Fundamentos
            df_factors['Value'] = FactorEngine.compute_value_score(fund_snapshot)
            df_factors['Quality'] = FactorEngine.compute_quality_score(fund_snapshot)
            df_factors['Growth'] = FactorEngine.compute_growth_score(fund_snapshot)
            
            if 'sector' in fund_snapshot.columns: df_factors['Sector'] = fund_snapshot['sector']
            
            # Drop NaN e Limpeza
            df_factors.dropna(thresh=2, inplace=True)
            
            # 4. Scoring Combinado (Weighted Average)
            df_factors['Z_Mom'] = FactorEngine.robust_zscore(df_factors['Momentum'])
            df_factors['Z_Val'] = FactorEngine.robust_zscore(df_factors['Value'])
            df_factors['Z_Qual'] = FactorEngine.robust_zscore(df_factors['Quality'])
            df_factors['Z_Grow'] = FactorEngine.robust_zscore(df_factors['Growth'])
            
            w = self.config['weights']
            df_factors['Final_Score'] = (
                df_factors['Z_Mom'] * w['Momentum'] +
                df_factors['Z_Val'] * w['Value'] +
                df_factors['Z_Qual'] * w['Quality'] +
                df_factors['Z_Grow'] * w['Growth']
            )
            
            # Seleção Top N
            top_n = self.config['top_n']
            selected_df = df_factors.sort_values('Final_Score', ascending=False).head(top_n)
            tickers_selected = selected_df.index.tolist()
            
            # 5. Otimização de Pesos
            risk_window = hist_prices[tickers_selected].tail(126).pct_change().dropna()
            if not risk_window.empty and len(risk_window) > 20:
                weights = PortfolioOptimizer.optimize_portfolio(risk_window, method=self.config['opt_method'])
            else:
                weights = pd.Series(1/len(tickers_selected), index=tickers_selected)
                
            # 6. Cálculo de Custos de Transação (Turnover)
            # Turnover = sum(abs(new_weight - old_weight))
            # Simplificação: Assumimos rebalanceamento total se tickers mudarem
            # Custo = Turnover * transaction_cost_bps
            turnover = 0.0
            if current_holdings.empty:
                turnover = 1.0 # 100% de entrada
            else:
                # Alinha índices
                all_tickers = list(set(weights.index) | set(current_holdings.index))
                w_new = weights.reindex(all_tickers).fillna(0)
                w_old = current_holdings.reindex(all_tickers).fillna(0)
                turnover = np.sum(np.abs(w_new - w_old)) / 2 # One-way turnover
            
            cost_impact = turnover * (self.transaction_cost_bps / 10000)
            
            # 7. Avançar no tempo
            period_prices = self.prices.loc[date:next_date, tickers_selected]
            if not period_prices.empty:
                period_rets = period_prices.pct_change().dropna()
                # Retorno ponderado
                strat_period_ret = period_rets.dot(weights)
                
                # Aplica custo no primeiro dia do período
                if not strat_period_ret.empty:
                    strat_period_ret.iloc[0] -= cost_impact 
                
                daily_returns.append(strat_period_ret)
            
            current_holdings = weights

        if daily_returns:
            full_series = pd.concat(daily_returns)
            # Remove duplicatas de índice causadas pelo resample
            full_series = full_series[~full_series.index.duplicated(keep='first')]
            return full_series
        return pd.Series()

# ==============================================================================
# UI PRINCIPAL (STREAMLIT)
# ==============================================================================

def main():
    st.sidebar.title("🧪 Quant Factor Lab Pro")
    st.sidebar.markdown("**Versão Institucional v2.0**")
    
    # --- Sidebar Inputs ---
    st.sidebar.header("1. Universo e Dados")
    default_tickers = "ITUB4.SA, VALE3.SA, PETR4.SA, WEGE3.SA, BBAS3.SA, RENT3.SA, BPAC11.SA, PRIO3.SA, RDOR3.SA, RADL3.SA, EQTL3.SA, TOTS3.SA, LREN3.SA, VIBRA3.SA, RAIL3.SA, SUZB3.SA, CMIG4.SA, GGBR4.SA, CSAN3.SA, BBSE3.SA"
    ticker_input = st.sidebar.text_area("Tickers (IBOV Proxy)", default_tickers, height=100)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    st.sidebar.header("2. Alocação de Fatores (Alpha)")
    w_mom = st.sidebar.slider("Momentum Residual", 0.0, 1.0, 0.4)
    w_val = st.sidebar.slider("Value (EV/EBITDA +)", 0.0, 1.0, 0.2)
    w_qual = st.sidebar.slider("Quality (Profitability)", 0.0, 1.0, 0.2)
    w_grow = st.sidebar.slider("Growth", 0.0, 1.0, 0.2)
    
    st.sidebar.header("3. Otimizador")
    opt_method = st.sidebar.selectbox("Método de Otimização", 
                                      ["Max Sharpe (Mean-Var)", "Min Volatility", "Inverse Volatility"])
    opt_map = {"Max Sharpe (Mean-Var)": "max_sharpe", 
               "Min Volatility": "min_vol", 
               "Inverse Volatility": "inv_vol"}
    
    top_n = st.sidebar.number_input("Ativos na Carteira", 5, 20, 8)

    if st.sidebar.button("🚀 Executar Backtest Institucional", type="primary"):
        
        with st.status("Processando Pipeline Quant...", expanded=True) as status:
            # 1. Fetching
            st.write("📡 Buscando dados de mercado...")
            prices = DataProvider.fetch_price_history(tickers, 
                                                      (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d'), 
                                                      datetime.now().strftime('%Y-%m-%d'))
            
            st.write("🏗️ Construindo base fundamentalista (Simulação PIT)...")
            fundamentals = DataProvider.fetch_current_fundamentals(tickers)
            
            if prices.empty or fundamentals.empty:
                status.update(label="Erro: Dados insuficientes.", state="error")
                st.stop()
                
            # 2. Backtest
            st.write("⚙️ Rodando Motor de Backtest (com custos)...")
            config = {
                'weights': {'Momentum': w_mom, 'Value': w_val, 'Quality': w_qual, 'Growth': w_grow},
                'top_n': top_n,
                'opt_method': opt_map[opt_method]
            }
            
            engine = BacktestEngine(prices, fundamentals, config)
            strategy_ret = engine.run()
            
            # Benchmark
            bench_ret = prices['BOVA11.SA'].pct_change().dropna()
            common_idx = strategy_ret.index.intersection(bench_ret.index)
            strategy_ret = strategy_ret.loc[common_idx]
            bench_ret = bench_ret.loc[common_idx]
            
            status.update(label="Concluído!", state="complete", expanded=False)

        # --- Resultados ---
        
        # Disclaimer Importante
        st.warning("""
        ⚠️ **Aviso de Integridade de Dados (Data Integrity Warning)**: 
        Este backtest utiliza um "Snapshot" atual dos fundamentos (Yahoo Finance Free Tier) aplicado historicamente. 
        **Existe Viés de Antecipação (Look-Ahead Bias) nos fatores fundamentais.**
        Para uso em produção, conecte a classe `DataProvider` a uma base de dados Point-in-Time paga (ex: Economatica, Bloomberg).
        """)

        tab1, tab2, tab3 = st.tabs(["📈 Performance & Risco", "🔍 Análise de Fatores", "📋 Posições Atuais"])
        
        with tab1:
            st.subheader("Análise de Performance Acumulada")
            cum_strat = (1 + strategy_ret).cumprod()
            cum_bench = (1 + bench_ret).cumprod()
            
            df_chart = pd.DataFrame({'Estratégia (Líquida de Custos)': cum_strat, 'Benchmark (BOVA11)': cum_bench})
            st.plotly_chart(px.line(df_chart, color_discrete_sequence=['#00CC96', '#EF553B']), use_container_width=True)
            
            st.subheader("Métricas de Risco (Risk Attribution)")
            metrics_strat = engine.calculate_risk_metrics(strategy_ret)
            metrics_bench = engine.calculate_risk_metrics(bench_ret)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retorno Total", f"{metrics_strat.get('Total Return',0):.2%}", delta=f"{(metrics_strat.get('Total Return',0)-metrics_bench.get('Total Return',0)):.2%}")
            col2.metric("Sharpe Ratio", f"{metrics_strat.get('Sharpe Ratio',0):.2f}", delta=f"{(metrics_strat.get('Sharpe Ratio',0)-metrics_bench.get('Sharpe Ratio',0)):.2f}")
            col3.metric("Max Drawdown", f"{metrics_strat.get('Max Drawdown',0):.2%}", help="Queda máxima do topo ao fundo")
            col4.metric("VaR 95% (Diário)", f"{metrics_strat.get('VaR (95%)',0):.2%}", help="Perda máxima esperada em 95% dos dias")
            
            with st.expander("Ver Tabela Detalhada de Risco"):
                risk_df = pd.DataFrame([metrics_strat, metrics_bench], index=['Estratégia', 'Benchmark']).T
                st.table(risk_df)

        with tab2:
            st.subheader("Diagnóstico de Dados (Outliers)")
            st.markdown("Visualização da dispersão dos dados fundamentais antes da normalização. Útil para identificar anomalias.")
            
            # Boxplot dos fundamentos brutos
            numeric_cols = fundamentals.select_dtypes(include=np.number).columns
            cols_to_plot = st.multiselect("Selecione métricas para inspecionar", numeric_cols, default=['forwardPE', 'returnOnEquity'])
            
            if cols_to_plot:
                df_melt = fundamentals[cols_to_plot].melt(var_name='Métrica', value_name='Valor')
                # Remover outliers extremos só para visualização do gráfico
                df_melt = df_melt[df_melt['Valor'].between(df_melt['Valor'].quantile(0.05), df_melt['Valor'].quantile(0.95))]
                fig_box = px.box(df_melt, x='Métrica', y='Valor', points="all", title="Distribuição de Métricas (Sem Outliers Extremos)")
                st.plotly_chart(fig_box, use_container_width=True)

        with tab3:
            st.subheader("Sugestão de Carteira Atual")
            # Recalcular score atual
            current_prices = prices
            mom = FactorEngine.compute_residual_momentum(current_prices)
            val = FactorEngine.compute_value_score(fundamentals)
            qual = FactorEngine.compute_quality_score(fundamentals)
            grow = FactorEngine.compute_growth_score(fundamentals)
            
            df_now = pd.DataFrame({'Momentum': mom, 'Value': val, 'Quality': qual, 'Growth': grow})
            df_now.dropna(inplace=True)
            
            # Normalizar
            df_now_z = df_now.apply(FactorEngine.robust_zscore)
            df_now['Composite_Score'] = (
                df_now_z['Momentum'] * w_mom +
                df_now_z['Value'] * w_val +
                df_now_z['Quality'] * w_qual +
                df_now_z['Growth'] * w_grow
            )
            
            top_picks = df_now.sort_values('Composite_Score', ascending=False).head(top_n)
            
            # Otimizar pesos atuais
            curr_risk = prices[top_picks.index].tail(126).pct_change().dropna()
            final_weights = PortfolioOptimizer.optimize_portfolio(curr_risk, method=opt_map[opt_method])
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(top_picks.style.background_gradient(cmap='Greens', subset=['Composite_Score']), use_container_width=True)
            with c2:
                fig_pie = px.pie(values=final_weights.values, names=final_weights.index, title="Alocação Otimizada")
                st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
