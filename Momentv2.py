import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import warnings

# Configurações iniciais
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# --- CONFIGURAÇÃO DA API ---
FMP_API_KEY = "rd6uBzkLLSPG68s9GcSx3folN76IxRhV"  # Sua chave aqui

# ==============================================================================
# 1. AQUISIÇÃO DE DADOS (PREÇOS E FUNDAMENTOS HISTÓRICOS REAIS)
# ==============================================================================

def fetch_price_data(tickers, start_date, end_date):
    """Baixa dados de preço ajustado do Yahoo Finance."""
    print("Baixando dados de preço (yfinance)...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def fetch_fmp_fundamentals(tickers, api_key, limit=80):
    """
    Busca dados históricos trimestrais (Key Metrics) via Financial Modeling Prep.
    limit=80 busca aprox. 20 anos de dados trimestrais.
    """
    print("Baixando dados fundamentais históricos (Financial Modeling Prep)...")
    
    all_data = []
    
    # Mapeamento: Nome na FMP -> Nome esperado pelo seu script
    column_mapping = {
        'date': 'date',
        'symbol': 'Ticker',
        'peRatio': 'trailingPE',
        'pbRatio': 'priceToBook',
        'roe': 'returnOnEquity',
        'netProfitMargin': 'profitMargins',
        'debtToEquity': 'debtToEquity',
        'enterpriseValueOverEBITDA': 'enterpriseValue' # Aprox. para EV/EBITDA
    }

    session = requests.Session()

    for ticker in tickers:
        try:
            # Endpoint: Key Metrics (Trimestral)
            url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=quarter&limit={limit}&apikey={api_key}"
            response = session.get(url)
            data = response.json()
            
            if not data or isinstance(data, dict) and 'Error' in data:
                print(f"  Aviso: Sem dados para {ticker}")
                continue
                
            df = pd.DataFrame(data)
            
            # Filtra e renomeia colunas
            cols_to_keep = [c for c in column_mapping.keys() if c in df.columns]
            df = df[cols_to_keep].rename(columns=column_mapping)
            
            # Garante que a data seja datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Adiciona colunas faltantes com NaN se a API não retornar
            expected_metrics = ['trailingPE', 'priceToBook', 'returnOnEquity', 'profitMargins', 'debtToEquity']
            for m in expected_metrics:
                if m not in df.columns:
                    df[m] = np.nan
            
            all_data.append(df)
            
        except Exception as e:
            print(f"  Erro ao baixar {ticker}: {e}")

    if not all_data:
        raise ValueError("Nenhum dado fundamental foi baixado. Verifique a API Key ou os Tickers.")

    # Concatena todos os tickers
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

def align_fundamentals_to_prices(fund_df, price_df):
    """
    Transforma os dados trimestrais em diários usando Forward Fill (ffill).
    Isso garante que em qualquer data do backtest, temos o dado do último balanço publicado.
    Retorna um DataFrame MultiIndex (Date, Ticker).
    """
    print("Alinhando dados fundamentais ao calendário de preços (Point-in-Time)...")
    
    # 1. Pivotar para ter datas no índice e (Ticker, Métrica) nas colunas
    # Removemos duplicatas de datas se houver
    fund_df = fund_df.drop_duplicates(subset=['date', 'Ticker'])
    
    pivot_df = fund_df.pivot(index='date', columns='Ticker')
    
    # 2. Reindexar para cobrir todas as datas de preço (Diário)
    # O método 'ffill' propaga o último valor válido para frente.
    # Ex: Balanço sai 31/03. De 01/04 até 29/06 (próximo balanço), usa-se o valor de 31/03.
    aligned_df = pivot_df.reindex(price_df.index, method='ffill')
    
    # 3. Empilhar (Stack) para voltar ao formato (Date, Ticker) -> Métricas
    # dropna=False mantém dias onde temos preço mas talvez ainda não tenha saído o primeiro balanço
    stacked_df = aligned_df.stack(level='Ticker', future_stack=True)
    
    # Ajuste fino: converter colunas para float
    stacked_df = stacked_df.apply(pd.to_numeric, errors='coerce')
    
    return stacked_df

# ==============================================================================
# 2. CÁLCULO DOS FATORES (SCORES)
# ==============================================================================

def compute_residual_momentum(price_df, window=252, lookback_months=12):
    """Calcula Momentum Residual (Retorno não explicado pelo mercado - Simples)."""
    # Simplificação: Usamos retorno total ajustado pela volatilidade como proxy robusto
    # para evitar regressão linear pesada dentro do loop.
    # Momentum Clássico: Retorno 12 meses excluindo o último mês
    
    log_ret = np.log(price_df / price_df.shift(1))
    
    # Retorno acumulado 12 meses (aprox 252 dias)
    cum_ret = price_df.pct_change(window).iloc[-1] if window < len(price_df) else price_df.pct_change().iloc[-1]
    
    # Excluir último mês (Reversal effect) - aprox 21 dias
    recent_ret = price_df.pct_change(21).iloc[-1] if 21 < len(price_df) else 0
    
    momentum_score = cum_ret - recent_ret
    return momentum_score

def compute_fundamental_momentum(fundamentals_df):
    """
    Variação do ROE ou Lucros em relação ao período anterior.
    Como fundamentals_df é o snapshot da data, precisamos calcular a variação
    em relação ao dado que estava disponível X dias atrás. 
    Aqui, usamos o nível atual de ROE como proxy de força, pois o diff no loop é complexo.
    """
    # Uma abordagem melhor num snapshot: Score é alto se ROE e Margens são altos
    # Para variação real, precisaríamos do lag dentro do DF.
    # Vamos usar um Score Misto de Crescimento Implícito (High ROE)
    score = fundamentals_df['returnOnEquity'].fillna(0)
    return score

def compute_value_score(fundamentals_df):
    """Value: P/E baixo, P/B baixo."""
    # Inverter P/E e P/B (quanto menor melhor)
    # Tratamento para P/E negativo ou zero: penalizar
    pe = fundamentals_df['trailingPE']
    pb = fundamentals_df['priceToBook']
    
    # Inverso do PE (Earnings Yield) - Lida melhor com negativos
    ey = 1 / pe
    ey = ey.replace([np.inf, -np.inf], 0)
    
    # Inverso do PB (Book Yield)
    by = 1 / pb
    by = by.replace([np.inf, -np.inf], 0)
    
    # Score combinado
    score = (ey + by) / 2
    return score

def compute_quality_score(fundamentals_df):
    """Quality: Margem Alta, Dívida Baixa, ROE Alto."""
    roe = fundamentals_df['returnOnEquity'].fillna(0)
    margin = fundamentals_df['profitMargins'].fillna(0)
    debt = fundamentals_df['debtToEquity'].fillna(100) # Preenche NaN com dívida alta para penalizar
    
    # Inverso da dívida (quanto menor melhor)
    inv_debt = 1 / (debt + 0.1) 
    
    # Normalizar simples antes de somar não é ideal sem z-score cross-sectional,
    # mas aqui somamos os valores brutos pois serão normalizados depois no loop.
    score = roe + margin + inv_debt
    return score

# ==============================================================================
# 3. ENGINE DE BACKTEST
# ==============================================================================

def run_dynamic_backtest(price_df, all_fundamentals_aligned, momentum_lookback=252, rebal_freq=63, top_n=5):
    """
    Executa o backtest usando dados Point-in-Time.
    all_fundamentals_aligned: DataFrame MultiIndex (Date, Ticker) já alinhado.
    """
    rebalance_dates = price_df.index[::rebal_freq]
    portfolio_history = []
    dates_history = []
    
    print(f"Iniciando Backtest: {len(rebalance_dates)} rebalanceamentos...")

    for i, rebal_date in enumerate(rebalance_dates):
        # Pular início se não tiver histórico suficiente
        if i * rebal_freq < 252: 
            continue
            
        # 1. Dados de Preço Históricos (até a data)
        hist_prices = price_df.loc[:rebal_date]
        if len(hist_prices) < momentum_lookback: continue
        
        # 2. Dados Fundamentais DA DATA (Point-in-Time)
        try:
            # Seleciona a fatia transversal (cross-section) do dia
            current_fundamentals = all_fundamentals_aligned.loc[rebal_date]
        except KeyError:
            # Se cair num feriado, pega o dia anterior disponível
            idx = all_fundamentals_aligned.index.get_level_values(0)
            prev_date = idx[idx <= rebal_date].max()
            current_fundamentals = all_fundamentals_aligned.loc[prev_date]

        # 3. Filtro de Liquidez/Existência
        # Garante que temos preço E fundamento
        valid_tickers = current_fundamentals.dropna(how='all').index.intersection(hist_prices.columns)
        
        if len(valid_tickers) < top_n:
            continue
            
        curr_fund_subset = current_fundamentals.loc[valid_tickers]
        hist_price_subset = hist_prices[valid_tickers]

        # 4. Cálculo dos Fatores
        # Momentum (Price Action)
        mom_score = hist_price_subset.pct_change(momentum_lookback).iloc[-1] # Simplificado 12m
        
        # Fundamentais
        fund_mom = compute_fundamental_momentum(curr_fund_subset)
        val_score = compute_value_score(curr_fund_subset)
        qual_score = compute_quality_score(curr_fund_subset)
        
        # 5. Combinação e Ranking
        factors = pd.DataFrame({
            'Momentum': mom_score,
            'Fund_Mom': fund_mom,
            'Value': val_score,
            'Quality': qual_score
        })
        
        # Limpeza e Normalização (Z-Score)
        factors = factors.dropna()
        if factors.empty: continue
        
        z_scores = (factors - factors.mean()) / (factors.std() + 1e-6)
        z_scores['Total'] = z_scores.mean(axis=1)
        
        # Seleção
        top_picks = z_scores.sort_values(by='Total', ascending=False).head(top_n).index.tolist()
        
        # 6. Calcular Retorno para o PRÓXIMO Período
        if i < len(rebalance_dates) - 1:
            next_date = rebalance_dates[i+1]
            # Retorno diário dos ativos selecionados nesse período
            period_data = price_df.loc[rebal_date:next_date, top_picks]
            period_returns = period_data.pct_change().mean(axis=1).dropna()
            
            if not period_returns.empty:
                portfolio_history.extend(period_returns.values)
                dates_history.extend(period_returns.index)

    # Criação da Série de Retorno Acumulado
    if not portfolio_history:
        return pd.Series(dtype=float)
        
    strategy_series = pd.Series(portfolio_history, index=dates_history)
    strategy_equity = (1 + strategy_series).cumprod()
    
    return strategy_equity

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Definir Universo ---
    # Lista de Tickers (Exemplo: Big Tech + Financeiras + Varejo)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "BAC", "WMT", "TGT", "PG", "KO", "PEP", "TSLA"]
    
    start_date = "2018-01-01"
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    try:
        # --- 2. Baixar Preços ---
        price_df = fetch_price_data(tickers, start_date, end_date)
        
        # --- 3. Baixar Fundamentais Históricos (FMP API) ---
        raw_fundamentals = fetch_fmp_fundamentals(tickers, FMP_API_KEY)
        
        # --- 4. Alinhar Dados (Criar Base Point-in-Time) ---
        # Este passo é crucial: transforma trimestral em diário (ffill)
        aligned_fundamentals = align_fundamentals_to_prices(raw_fundamentals, price_df)
        
        # --- 5. Rodar Backtest ---
        equity_curve = run_dynamic_backtest(price_df, aligned_fundamentals, top_n=3)
        
        # --- 6. Resultados ---
        if not equity_curve.empty:
            print("\n" + "="*40)
            print(f"Retorno Acumulado: {equity_curve.iloc[-1]:.2f}x")
            print("="*40)
            
            # Plot simples (se estiver no Jupyter ou local)
            try:
                import matplotlib.pyplot as plt
                equity_curve.plot(title="Estratégia Momentum + Fundamental (Dados Históricos Reais FMP)")
                plt.show()
            except ImportError:
                print("Matplotlib não instalado, gráfico pulado.")
        else:
            print("Backtest não gerou trades (verifique datas ou dados).")
            
    except Exception as e:
        print(f"\nErro Crítico na Execução: {e}")
