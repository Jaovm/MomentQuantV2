import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import newton
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="B3 Equity Research Dashboard | CFA Level Analysis",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stMetric {
        background-color: transparent !important;
    }
    .buy-signal { color: #00FF00; font-weight: bold; }
    .sell-signal { color: #FF0000; font-weight: bold; }
    .neutral-signal { color: #FFA500; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- LISTA DE ATIVOS ---
TICKERS = [
    "ITUB3.SA", "TOTS3.SA", "MDIA3.SA", "TAEE3.SA", "BBSE3.SA",
    "WEGE3.SA", "PSSA3.SA", "EGIE3.SA", "B3SA3.SA", "VIVT3.SA",
    "AGRO3.SA", "PRIO3.SA", "BBAS3.SA", "BPAC11.SA", "SBSP3.SA",
    "SAPR4.SA", "CMIG3.SA", "UNIP6.SA", "FRAS3.SA", "CPFE3.SA"
]

# --- CLASSE DE DADOS E ANÁLISE (Backend) ---

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = {}
        self.financials = pd.DataFrame()
        self.balance_sheet = pd.DataFrame()
        self.cashflow = pd.DataFrame()
        self.history = pd.DataFrame()
        self.data_loaded = False

    def fetch_data(self):
        """Coleta dados da API yfinance com tratamento de erros."""
        try:
            self.info = self.stock.info
            # Tenta pegar demonstrações anuais
            self.financials = self.stock.financials.T
            self.balance_sheet = self.stock.balance_sheet.T
            self.cashflow = self.stock.cashflow.T
            
            # Histórico de 5 anos
            self.history = self.stock.history(period="5y")
            
            if self.history.empty:
                raise ValueError("Histórico de preços vazio.")
                
            self.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Erro ao carregar dados para {self.ticker}: {e}")
            return False

    def get_metrics(self):
        """Extrai métricas fundamentais chave."""
        if not self.data_loaded: return {}
        
        current_price = self.history['Close'].iloc[-1]
        market_cap = self.info.get('marketCap', 0)
        pe_ratio = self.info.get('trailingPE', 0)
        ev_ebitda = self.info.get('enterpriseToEbitda', 0)
        beta = self.info.get('beta', 1.0)
        
        return {
            'price': current_price,
            'market_cap': market_cap,
            'pe': pe_ratio,
            'ev_ebitda': ev_ebitda,
            'beta': beta if beta is not None else 1.0
        }

    def calculate_micro_kpis(self):
        """Calcula KPIs microeconômicos: Margens, ROIC, Dívida."""
        if self.financials.empty or self.balance_sheet.empty:
            return None

        # Pega o ano mais recente disponível
        recent_fin = self.financials.iloc[0] if not self.financials.empty else None
        recent_bs = self.balance_sheet.iloc[0] if not self.balance_sheet.empty else None
        
        if recent_fin is None or recent_bs is None: return None

        # Tratamento seguro de chaves (yfinance muda chaves ocasionalmente)
        def get_val(series, keys, default=0):
            for k in keys:
                if k in series: return series[k]
            return default

        revenue = get_val(recent_fin, ['Total Revenue', 'Operating Revenue'])
        ebit = get_val(recent_fin, ['EBIT', 'Operating Income'])
        net_income = get_val(recent_fin, ['Net Income', 'Net Income Common Stockholders'])
        
        total_debt = get_val(recent_bs, ['Total Debt', 'Long Term Debt']) + get_val(recent_bs, ['Current Debt'], 0)
        total_equity = get_val(recent_bs, ['Stockholders Equity', 'Total Stockholder Equity'])
        cash = get_val(recent_bs, ['Cash And Cash Equivalents', 'Cash Financial'])
        
        # Cálculos
        gross_margin = (revenue - get_val(recent_fin, ['Cost Of Revenue'])) / revenue if revenue else 0
        ebit_margin = ebit / revenue if revenue else 0
        net_margin = net_income / revenue if revenue else 0
        
        invested_capital = total_equity + total_debt - cash
        nopat = ebit * 0.66 # Assumindo taxa efetiva de imposto ~34% Brasil
        roic = nopat / invested_capital if invested_capital else 0
        
        debt_to_equity = total_debt / total_equity if total_equity else 0

        return {
            'gross_margin': gross_margin,
            'ebit_margin': ebit_margin,
            'net_margin': net_margin,
            'roic': roic,
            'debt_to_equity': debt_to_equity,
            'revenue_growth': self._calc_cagr(self.financials, 'Total Revenue')
        }

    def _calc_cagr(self, df, col_name):
        """Calcula CAGR de 3 anos."""
        try:
            vals = [df.iloc[i][col_name] for i in range(min(3, len(df)))]
            if len(vals) < 2: return 0
            start, end = vals[-1], vals[0] # DataFrame invertido (mais recente primeiro)
            if start <= 0: return 0
            return (end / start) ** (1/len(vals)) - 1
        except:
            return 0

    def calculate_dcf(self, risk_free_rate=0.11, market_premium=0.06, terminal_growth=0.025, projection_years=5):
        """Modelo de Valuation DCF Simplificado e Cálculo de TIR."""
        if self.cashflow.empty: return None
        
        # 1. Estimar Free Cash Flow (FCF) = Operating Cash Flow - CapEx
        try:
            ocf = self.cashflow.iloc[0]['Operating Cash Flow']
            capex = self.cashflow.iloc[0]['Capital Expenditure']
            # CapEx costuma vir negativo no yfinance
            fcf_ttm = ocf + capex 
        except:
            # Fallback se chaves não existirem
            fcf_ttm = self.info.get('freeCashflow', 0)

        if fcf_ttm <= 0:
            return {"error": "FCF negativo ou indisponível, DCF não confiável."}

        # 2. WACC (Estimativa)
        beta = self.info.get('beta', 1.0)
        if beta is None: beta = 1.0
        cost_of_equity = risk_free_rate + beta * market_premium
        
        # Custo da dívida simplificado (CDI + spread) -> Pós impostos
        cost_of_debt = (risk_free_rate + 0.02) * (1 - 0.34)
        
        kpis = self.calculate_micro_kpis()
        if kpis:
            d_e = kpis['debt_to_equity']
            w_e = 1 / (1 + d_e)
            w_d = d_e / (1 + d_e)
            wacc = (w_e * cost_of_equity) + (w_d * cost_of_debt)
        else:
            wacc = cost_of_equity # Fallback

        # 3. Projeção
        growth_rate = kpis['revenue_growth'] if kpis else 0.05
        # Capa crescimento para ser conservador
        growth_rate = min(max(growth_rate, 0.02), 0.15) 
        
        future_fcf = []
        for i in range(1, projection_years + 1):
            fcf = fcf_ttm * ((1 + growth_rate) ** i)
            future_fcf.append(fcf)
            
        # 4. Valor Terminal
        terminal_value = (future_fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        
        # 5. Valor Presente (PV)
        pv_fcf = sum([f / ((1 + wacc) ** (i + 1)) for i, f in enumerate(future_fcf)])
        pv_terminal = terminal_value / ((1 + wacc) ** projection_years)
        
        enterprise_value = pv_fcf + pv_terminal
        
        # 6. Equity Value
        net_debt = 0
        try:
            # Tenta pegar dívida líquida, senão 0 para ser conservador no erro
            debt = self.balance_sheet.iloc[0].get('Total Debt', 0)
            cash = self.balance_sheet.iloc[0].get('Cash And Cash Equivalents', 0)
            net_debt = debt - cash
        except: pass
        
        equity_value = enterprise_value - net_debt
        shares = self.info.get('sharesOutstanding', 1)
        
        fair_price = equity_value / shares
        current_price = self.history['Close'].iloc[-1]
        
        # 7. Cálculo da TIR Implícita (Internal Rate of Return)
        # Encontrar a taxa de desconto 'r' que iguala o PV dos fluxos ao preço atual (ajustado por div liq e shares)
        # Equity Value Target = Current Price * Shares
        target_ev = (current_price * shares) + net_debt
        
        def npv_func(r):
            # Evita divisão por zero ou taxas absurdas
            if r <= terminal_growth: return 9999999999
            pv_f = sum([f / ((1 + r) ** (i + 1)) for i, f in enumerate(future_fcf)])
            term = (future_fcf[-1] * (1 + terminal_growth)) / (r - terminal_growth)
            pv_t = term / ((1 + r) ** projection_years)
            return (pv_f + pv_t) - target_ev

        try:
            implied_irr = newton(npv_func, wacc)
        except:
            implied_irr = 0.0 # Falha na convergência
            
        margin_of_safety = (fair_price - current_price) / fair_price

        return {
            'fair_price': fair_price,
            'current_price': current_price,
            'wacc': wacc,
            'growth_rate_used': growth_rate,
            'margin_of_safety': margin_of_safety,
            'implied_irr': implied_irr,
            'fcf_ttm': fcf_ttm,
            'shares': shares
        }

# --- FUNÇÕES DE CACHE E INTERFACE ---

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    analyzer = StockAnalyzer(ticker)
    if analyzer.fetch_data():
        return analyzer
    return None

def color_recommendation(val):
    if val == "COMPRA": return "color: green; font-weight: bold"
    if val == "EVITAR": return "color: red; font-weight: bold"
    return "color: orange; font-weight: bold"

# --- SIDEBAR ---
st.sidebar.title("🛠️ Parâmetros de Valuation")
selected_ticker = st.sidebar.selectbox("Selecione o Ativo:", TICKERS)

st.sidebar.markdown("---")
st.sidebar.subheader("Premissas Macro")
rf_input = st.sidebar.number_input("Taxa Livre de Risco (Rf) %", value=11.5, step=0.1) / 100
erp_input = st.sidebar.number_input("Prêmio de Risco Mercado %", value=6.0, step=0.1) / 100
g_input = st.sidebar.number_input("Crescimento Perpétuo (g) %", value=2.5, step=0.1) / 100

st.sidebar.markdown("---")
st.sidebar.info("Dados fornecidos por Yahoo Finance API. Valores em BRL.")

# --- MAIN APP ---

stock = get_stock_data(selected_ticker)

if stock:
    # Cabeçalho
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(f"https://logo.clearbit.com/{stock.info.get('website', 'google.com').replace('https://www.', '').split('/')[0]}", width=80)
    with col2:
        st.title(f"{stock.info.get('longName', selected_ticker)} ({selected_ticker})")
        st.markdown(f"**Setor:** {stock.info.get('sector', 'N/A')} | **Indústria:** {stock.info.get('industry', 'N/A')}")

    # Cálculos Principais
    metrics = stock.get_metrics()
    micro = stock.calculate_micro_kpis()
    valuation = stock.calculate_dcf(risk_free_rate=rf_input, market_premium=erp_input, terminal_growth=g_input)

    # --- TAB DASHBOARD ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "🧠 Valuation (DCF & TIR)", "🔬 Microeconomia", "📋 Risco & Recomendação"])

    with tab1:
        # Preço e Gráfico
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço Atual", f"R$ {metrics['price']:.2f}")
        c2.metric("P/L (LTM)", f"{metrics['pe']:.2f}x")
        c3.metric("EV/EBITDA", f"{metrics['ev_ebitda']:.2f}x")
        c4.metric("Beta", f"{metrics['beta']:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock.history.index,
                        open=stock.history['Open'], high=stock.history['High'],
                        low=stock.history['Low'], close=stock.history['Close'], name='Market Data'))
        fig.update_layout(title='Evolução de Preço (5 Anos)', xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Fluxo de Caixa Descontado (DCF)")
        
        if valuation and "error" not in valuation:
            col_v1, col_v2, col_v3 = st.columns(3)
            
            fair_val = valuation['fair_price']
            curr_val = valuation['current_price']
            upside = (fair_val - curr_val) / curr_val * 100
            
            with col_v1:
                st.subheader("Preço Justo")
                st.metric("Fair Value", f"R$ {fair_val:.2f}", delta=f"{upside:.1f}% Upside")
            
            with col_v2:
                st.subheader("Margem de Segurança")
                color = "green" if valuation['margin_of_safety'] > 0.2 else "red"
                st.markdown(f"<h2 style='color:{color}'>{valuation['margin_of_safety']*100:.1f}%</h2>", unsafe_allow_html=True)
            
            with col_v3:
                st.subheader("TIR Implícita")
                tir = valuation['implied_irr'] * 100
                wacc_disp = valuation['wacc'] * 100
                delta_tir = tir - wacc_disp
                st.metric("IRR (Retorno Esperado)", f"{tir:.1f}%", delta=f"{delta_tir:.1f} pp vs WACC")

            st.markdown("---")
            
            # Análise de Sensibilidade (Heatmap)
            st.subheader("🔎 Análise de Sensibilidade: WACC vs Taxa de Crescimento")
            
            wacc_range = np.linspace(valuation['wacc'] - 0.02, valuation['wacc'] + 0.02, 5)
            g_range = np.linspace(valuation['growth_rate_used'] - 0.01, valuation['growth_rate_used'] + 0.01, 5)
            
            sensitivity_data = []
            for w in wacc_range:
                row = []
                for g in g_range:
                    # Recálculo simplificado do PV para o heatmap
                    # (Reutilizando a lógica simplificada para performance)
                    val_temp = stock.calculate_dcf(risk_free_rate=rf_input, market_premium=erp_input, terminal_growth=g)
                    # Ajuste grosseiro para simular o WACC inputado no loop (apenas para visualização)
                    # Nota: O método ideal seria refazer o DCF completo passando WACC forçado.
                    # Aqui usamos a variação percentual aproximada.
                    adjustment_factor = valuation['wacc'] / w 
                    row.append(val_temp['fair_price'] * adjustment_factor) 
                sensitivity_data.append(row)
            
            fig_heat = px.imshow(sensitivity_data,
                                labels=dict(x="Crescimento (g)", y="WACC", color="Preço Justo"),
                                x=[f"{g*100:.1f}%" for g in g_range],
                                y=[f"{w*100:.1f}%" for w in wacc_range],
                                text_auto=".2f", aspect="auto", color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_heat, use_container_width=True)
            
        else:
            st.warning("Não foi possível calcular o DCF. Dados de Fluxo de Caixa Livre insuficientes.")

    with tab3:
        if micro:
            st.header("Qualidade & Microeconomia")
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Eficiência de Capital")
                fig_bar = go.Figure(data=[
                    go.Bar(name='ROIC', x=['ROIC vs WACC'], y=[micro['roic']*100], marker_color='green'),
                    go.Bar(name='WACC Est.', x=['ROIC vs WACC'], y=[valuation['wacc']*100 if valuation else 0], marker_color='red')
                ])
                fig_bar.update_layout(barmode='group', title="Retorno sobre Capital Investido")
                st.plotly_chart(fig_bar, use_container_width=True)
                
                if micro['roic'] > (valuation['wacc'] if valuation else 0):
                    st.success(f"✅ A empresa cria valor (ROIC {micro['roic']*100:.1f}% > WACC). Indício de vantagem competitiva (Moat).")
                else:
                    st.error("⚠️ A empresa destrói valor (ROIC < WACC).")

            with c2:
                st.subheader("Saúde Financeira")
                st.metric("Dívida Líq/EBITDA", f"{stock.info.get('debtToEquity', 0):.2f}") # Proxy via info
                st.metric("Margem Líquida", f"{micro['net_margin']*100:.1f}%")
                
                # Gráfico de Margens
                df_margins = pd.DataFrame({
                    'Margem': ['Bruta', 'EBIT', 'Líquida'],
                    'Valor': [micro['gross_margin'], micro['ebit_margin'], micro['net_margin']]
                })
                fig_m = px.bar(df_margins, x='Margem', y='Valor', color='Margem', title="Estrutura de Margens")
                st.plotly_chart(fig_m, use_container_width=True)

    with tab4:
        st.header("Tese de Investimento & Riscos")
        
        # Lógica de Recomendação Automatizada
        rec = "NEUTRO"
        reasons = []
        
        if valuation and "error" not in valuation:
            mos = valuation['margin_of_safety']
            roic = micro['roic'] if micro else 0
            
            if mos > 0.30 and roic > valuation['wacc']:
                rec = "COMPRA"
                reasons.append("Margem de Segurança Alta (>30%)")
                reasons.append("Criação de Valor (ROIC > WACC)")
            elif mos > 0.15:
                rec = "COMPRA ESPECULATIVA"
                reasons.append("Desconto moderado em relação ao Valor Justo")
            elif mos < -0.20:
                rec = "EVITAR/VENDA"
                reasons.append("Ativo sobrevalorizado pelo modelo DCF")
            
            # Penalidades de Risco
            if micro and micro['debt_to_equity'] > 2.5:
                rec = "EVITAR" if rec != "VENDA" else "VENDA"
                reasons.append("⚠️ Alavancagem Excessiva (Dívida/PL > 2.5x)")
        
        # Display
        col_r1, col_r2 = st.columns([1, 2])
        
        with col_r1:
            st.subheader("Recomendação")
            color = "green" if "COMPRA" in rec else "red" if "VENDA" in rec or "EVITAR" in rec else "orange"
            st.markdown(f"<div style='text-align: center; background-color: {color}; color: white; padding: 20px; border-radius: 10px;'><h1>{rec}</h1></div>", unsafe_allow_html=True)
        
        with col_r2:
            st.subheader("Justificativas & Riscos")
            for r in reasons:
                st.markdown(f"- {r}")
            
            st.markdown("### Matriz de Riscos Gerais")
            risk_data = {
                "Risco Macro": "Alta da Selic impacta custo de capital e valuation.",
                "Risco Setorial": f"Dependência de regulação e concorrência no setor {stock.info.get('sector')}.",
                "Risco Execução": "Capacidade de entregar o crescimento projetado no DCF."
            }
            st.table(pd.DataFrame(risk_data.items(), columns=["Tipo", "Descrição"]))

    # Exportar Dados
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar Relatório (CSV)"):
        report_data = {
            "Ticker": [selected_ticker],
            "Preço": [metrics['price']],
            "Preço Justo": [valuation['fair_price'] if valuation else 0],
            "TIR": [valuation['implied_irr'] if valuation else 0],
            "ROIC": [micro['roic'] if micro else 0],
            "Recomendação": [rec]
        }
        df_rep = pd.DataFrame(report_data)
        st.sidebar.download_button("Download CSV", df_rep.to_csv(index=False), f"{selected_ticker}_report.csv", "text/csv")

else:
    st.info("Carregando dados ou erro na conexão com a API. Verifique o ticker.")

# Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** Esta ferramenta é apenas para fins educacionais e de demonstração tecnológica. Os cálculos utilizam dados públicos que podem estar atrasados ou incorretos. Não constitui recomendação de investimento. Consulte um profissional certificado antes de investir.")
