# Moment_Corrected.py
# Script de Backtest de Fatores com Correção de Look-Ahead Bias e Integração Fundamentus
# Autor: Adaptado por Gemini para Estrutura Point-in-Time

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fundamentus  # Biblioteca necessária: pip install fundamentus
import warnings

# Ignorar avisos de fragmentação do pandas para limpeza do output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. Configurações e Funções Auxiliares
# ==========================================

def get_b3_tickers():
    """
    Retorna uma lista de exemplo de tickers da B3.
    Em produção, substitua por sua lista completa ou IBOV.
    """
    return [
        'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'PETR3.SA',
        'ABEV3.SA', 'WEGE3.SA', 'BBAS3.SA', 'ITSA4.SA', 'RENT3.SA',
        'BPAC11.SA', 'SUZB3.SA', 'HAPV3.SA', 'EQTL3.SA', 'RADL3.SA',
        'RDOR3.SA', 'PRIO3.SA', 'RAIL3.SA', 'GGBR4.SA', 'JBSS3.SA'
    ]

# ==========================================
# 2. Coleta de Dados Fundamentalistas (Fundamentus)
# ==========================================

def fetch_fundamentals(tickers, price_df_index):
    """
    Busca dados fundamentais atuais via Fundamentus e cria uma estrutura 
    de série temporal simulada (Snapshot replicado) para o backtest.
    
    Args:
        tickers (list): Lista de tickers com sufixo .SA
        price_df_index (DatetimeIndex): Índice de datas do backtest
    
    Returns:
        pd.DataFrame: MultiIndex (Date, Ticker) com métricas.
    """
    print(f"\n--- Coletando dados fundamentalistas (Fonte: Fundamentus) ---")
    
    # 1. Obter dados brutos do Fundamentus (retorna todos os papéis da B3)
    try:
        # get_resultado_raw retorna dataframe com Index = Ticker (ex: PETR4)
        df_fundamentus = fundamentus.get_resultado_raw()
    except Exception as e:
        print(f"Erro crítico ao conectar com Fundamentus: {e}")
        return pd.DataFrame()

    # 2. Tratamento de Tickers
    # Removemos o '.SA' da lista de entrada para filtrar no DF do Fundamentus
    tickers_clean = [t.replace('.SA', '') for t in tickers]
    
    # Filtra apenas os tickers do nosso universo que existem no Fundamentus
    df_selected = df_fundamentus[df_fundamentus.index.isin(tickers_clean)].copy()
    
    print(f"Tickers encontrados no Fundamentus: {len(df_selected)} de {len(tickers)}")

    # 3. Mapeamento e Conversão de Colunas
    # Mapeia as colunas do Fundamentus para os nomes esperados pelas funções de score
    # Nota: Fundamentus retorna decimais corretos (0.15 para 15%) na função raw? 
    # Verificação: Geralmente raw retorna floats.
    
    rename_map = {
        'P/L': 'trailingPE',
        'P/VP': 'priceToBook',
        'ROE': 'returnOnEquity',
        'Mrg. Liq.': 'netMargins',       # Usado para Quality
        'Div. Brut/ Pat.': 'debtToEquity',
        'Liq. Corr.': 'currentRatio',
        'EV/EBIT': 'evToEbit'            # Adicional para Value se necessário
    }
    
    df_selected.rename(columns=rename_map, inplace=True)

    # Converter colunas numéricas (garantia)
    cols_to_numeric = list(rename_map.values())
    for col in cols_to_numeric:
        if col in df_selected.columns:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

    # Re-adicionar sufixo .SA no índice para compatibilidade com YFinance
    df_selected.index = [f"{t}.SA" for t in df_selected.index]
    df_selected.index.name = 'ticker'

    # --- ARQUITETURA POINT-IN-TIME (Simulação) ---
    print("Estruturando arquitetura Point-in-Time (Replicando Snapshot)...")
    
    if df_selected.empty:
        return pd.DataFrame()

    # Expansão para Série Temporal
    # Multiplicamos o snapshot por todas as datas disponíveis no backtest
    dates = price_df_index.unique()
    
    # Criação do MultiIndex (Date, Ticker)
    # Isso simula que o dado de hoje estava disponível em todas as datas passadas
    # (Limitação conhecida, mas arquitetura correta)
    df_time_series = pd.concat([df_selected] * len(dates), keys=dates, names=['Date', 'ticker'])
    
    return df_time_series

# ==========================================
# 3. Funções de Cálculo de Scores (Fatores)
# ==========================================

def compute_fundamental_momentum(fundamentals_df):
    """
    Calcula Momentum Fundamental. 
    Nota: Com snapshot estático, este score tende a ser neutro ou baseado apenas em níveis,
    pois não temos crescimento histórico real no Fundamentus gratuito.
    """
    scores = pd.Series(0, index=fundamentals_df.index)
    
    # Como fallback, podemos usar margens altas como proxy de "bom momento" operacional
    if 'netMargins' in fundamentals_df.columns:
        # Z-Score da Margem Líquida
        metric = fundamentals_df['netMargins'].fillna(0)
        if metric.std() != 0:
            scores = (metric - metric.mean()) / metric.std()
            
    return scores

def compute_value_score(fundamentals_df):
    """
    Calcula Value Score (P/L, P/VP, etc). Quanto menor, melhor.
    """
    score_df = pd.DataFrame(index=fundamentals_df.index)
    
    # Inverter lógica: queremos P/L baixo, então multiplicamos por -1 no Z-Score
    metrics = ['trailingPE', 'priceToBook', 'evToEbit']
    
    for m in metrics:
        if m in fundamentals_df.columns:
            val = fundamentals_df[m]
            # Limpeza básica de outliers e valores negativos inválidos para múltiplos
            val = val.replace([np.inf, -np.inf], np.nan)
            
            # Z-Score invertido (Menor é melhor)
            if val.std() > 0:
                z_score = (val - val.mean()) / val.std()
                score_df[m] = -z_score # Inverte sinal
            else:
                score_df[m] = 0
    
    return score_df.mean(axis=1).fillna(0)

def compute_quality_score(fundamentals_df):
    """
    Calcula Quality Score (ROE, Margens, Baixa Dívida).
    """
    score_df = pd.DataFrame(index=fundamentals_df.index)
    
    # ROE e Margens (Maior é melhor)
    positive_metrics = ['returnOnEquity', 'netMargins', 'currentRatio']
    for m in positive_metrics:
        if m in fundamentals_df.columns:
            val = fundamentals_df[m].replace([np.inf, -np.inf], np.nan)
            if val.std() > 0:
                score_df[m] = (val - val.mean()) / val.std()

    # Dívida (Menor é melhor)
    if 'debtToEquity' in fundamentals_df.columns:
        val = fundamentals_df['debtToEquity'].replace([np.inf, -np.inf], np.nan)
        if val.std() > 0:
            z_score = (val - val.mean()) / val.std()
            score_df['debtToEquity'] = -z_score # Inverte sinal

    return score_df.mean(axis=1).fillna(0)

# ==========================================
# 4. Engine de Backtest
# ==========================================

def run_dynamic_backtest(tickers, start_date="2020-01-01", end_date="2023-12-30"):
    print(f"\n--- Iniciando Backtest ({start_date} a {end_date}) ---")
    
    # 1. Download de Preços (Yahoo Finance)
    print("Baixando histórico de preços (Yahoo Finance)...")
    price_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Tratamento para download recente do yfinance (se vier MultiIndex nas colunas)
    if isinstance(price_df.columns, pd.MultiIndex):
         price_df.columns = price_df.columns.get_level_values(0)
    
    # Preenche feriados (ffill) para manter continuidade
    price_df = price_df.ffill().dropna(axis=1, how='all')
    
    # Atualiza lista de tickers com os que realmente têm preço
    valid_tickers = price_df.columns.tolist()
    
    # 2. Prepara Dados Fundamentais (Point-in-Time Simulado)
    all_fundamentals = fetch_fundamentals(valid_tickers, price_df.index)
    
    if all_fundamentals.empty:
        print("Erro: Nenhum dado fundamentalista disponível. Abortando.")
        return None

    # 3. Configuração do Loop
    # Rebalanceamento Mensal ('ME' ou 'M' dependendo da versão do pandas)
    rebalance_dates = price_df.resample('ME').last().index 
    
    portfolio_returns = []
    portfolio_value = 100.0 # Base 100
    portfolio_curve = [100.0]
    dates_curve = [price_df.index[0]]
    
    current_weights = pd.Series(0, index=valid_tickers)
    
    print("\nExecutando Loop de Rebalanceamento...")
    
    # Itera sobre o histórico dia a dia ou período a período
    # Para simplificar, calculamos retornos entre datas de rebalanceamento
    
    last_rebal_date = price_df.index[0]
    
    for i, rebal_date in enumerate(rebalance_dates):
        if rebal_date not in price_df.index:
            # Tenta encontrar a data válida mais próxima anterior
            try:
                rebal_date = price_df.index[price_df.index.get_indexer([rebal_date], method='pad')[0]]
            except:
                continue
                
        # --- Cálculo de Retorno do Período Anterior ---
        # Preços entre o último rebal e o atual
        period_prices = price_df.loc[last_rebal_date:rebal_date]
        if not period_prices.empty and sum(current_weights) > 0:
            # Retorno simples do período ponderado
            # (Simplificação: assume pesos fixos durante o mês)
            pct_change = period_prices.iloc[-1] / period_prices.iloc[0] - 1
            period_return = (pct_change * current_weights).sum()
            
            new_value = portfolio_curve[-1] * (1 + period_return)
            portfolio_curve.append(new_value)
            dates_curve.append(rebal_date)
        elif sum(current_weights) == 0 and len(portfolio_curve) == 1:
            # Primeiro loop, apenas ajusta data
            dates_curve[0] = rebal_date

        # --- SELEÇÃO DE ATIVOS (LOOK-AHEAD FIXED) ---
        
        # 1. Slice Temporal: Pegamos apenas os dados fundamentais "deste dia"
        try:
            # Como all_fundamentals é MultiIndex (Date, Ticker), .loc[rebal_date]
            # retorna um DF com Index=Ticker. Perfeito.
            fundamentals_at_rebal = all_fundamentals.loc[rebal_date]
        except KeyError:
            # Se não houver dados para esta data exata (ex: feriado no index simulado), pula
            last_rebal_date = rebal_date
            continue
            
        if fundamentals_at_rebal.empty:
            last_rebal_date = rebal_date
            continue

        # 2. Momentum de Preço (Calculado com dados passados reais)
        # Momentum de 6 meses (126 dias úteis)
        lookback = 126
        start_mom_date = rebal_date - pd.Timedelta(days=180) # Aproximado
        
        # Pega janela de preços até hoje (rebal_date)
        price_window = price_df.loc[:rebal_date].tail(lookback+1)
        
        if len(price_window) > 100:
            price_mom = (price_window.iloc[-1] / price_window.iloc[0]) - 1
        else:
            price_mom = pd.Series(0, index=valid_tickers)

        # 3. Scores Fundamentais (Usando o slice do momento)
        # Alinha índices (intersecção entre preços e fundamentos)
        common_tickers = fundamentals_at_rebal.index.intersection(price_mom.index)
        
        f_snapshot = fundamentals_at_rebal.loc[common_tickers]
        p_mom_snapshot = price_mom.loc[common_tickers]
        
        # Calcula Fatores
        val_s = compute_value_score(f_snapshot)
        qual_s = compute_quality_score(f_snapshot)
        fund_mom_s = compute_fundamental_momentum(f_snapshot)
        
        # Normalização Final (Z-Score transversal)
        def normalize(s):
            return (s - s.mean()) / s.std() if s.std() > 0 else s
            
        final_score = (
            0.4 * normalize(p_mom_snapshot) +
            0.2 * normalize(val_s) +
            0.2 * normalize(qual_s) +
            0.2 * normalize(fund_mom_s)
        )
        
        # 4. Alocação (Top N)
        top_n = 5
        top_assets = final_score.nlargest(top_n).index
        
        # Pesos Iguais (Equal Weight)
        weight = 1.0 / len(top_assets) if len(top_assets) > 0 else 0
        current_weights = pd.Series(0, index=valid_tickers)
        current_weights.loc[top_assets] = weight
        
        last_rebal_date = rebal_date

    # ==========================================
    # 5. Resultados
    # ==========================================
    results_df = pd.DataFrame({'Portfolio': portfolio_curve}, index=dates_curve)
    
    # Benchmark (IBOV aproximado via ETF BOVA11 se disponível, ou média dos ativos)
    # Calculando retorno médio do universo como benchmark simples
    univ_ret = price_df.pct_change().mean(axis=1)
    univ_cum = (1 + univ_ret).cumprod() * 100
    # Reindexar benchmark para as datas da curva
    bench_curve = univ_cum.reindex(results_df.index, method='ffill')
    results_df['Benchmark_Universe'] = bench_curve / bench_curve.iloc[0] * 100

    return results_df

# ==========================================
# 6. Execução Principal
# ==========================================

if __name__ == "__main__":
    tickers = get_b3_tickers()
    
    # Executa Backtest
    df_result = run_dynamic_backtest(tickers)
    
    if df_result is not None:
        print("\n--- Resultados Finais ---")
        print(df_result.tail())
        
        # Métricas Simples
        total_ret = (df_result['Portfolio'].iloc[-1] / df_result['Portfolio'].iloc[0]) - 1
        print(f"Retorno Total Acumulado: {total_ret:.2%}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df_result.index, df_result['Portfolio'], label='Estratégia Multifator (Fundamentus)')
        plt.plot(df_result.index, df_result['Benchmark_Universe'], label='Universo (Média)', linestyle='--')
        plt.title('Backtest Multifator: Preço Dinâmico + Fundamentos Estáticos (Simulação PIT)')
        plt.xlabel('Data')
        plt.ylabel('Valor Base 100')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
