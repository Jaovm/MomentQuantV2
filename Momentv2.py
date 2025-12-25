import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# --- Configuração da Página ---
st.set_page_config(
    page_title="Terminal Valuation CFA | Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS Personalizados ---
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; border-left: 5px solid #1f77b4;}
    .big-font {font-size:24px !important; font-weight: bold;}
    .success-text {color: #28a745; font-weight: bold;}
    .warning-text {color: #ffc107; font-weight: bold;}
    .danger-text {color: #dc3545; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- Lista de Ativos Monitorados ---
TICKERS = [
    'ITUB3.SA', 'TOTS3.SA', 'MDIA3.SA', 'TAEE3.SA', 'BBSE3.SA', 
    'WEGE3.SA', 'PSSA3.SA', 'EGIE3.SA', 'B3SA3.SA', 'VIVT3.SA', 
    'AGRO3.SA', 'PRIO3.SA', 'BBAS3.SA', 'BPAC11.SA', 'SBSP3.SA', 
    'SAPR4.SA', 'CMIG3.SA', 'UNIP6.SA', 'FRAS3.SA', 'CPFE3.SA'
]

# --- Funções Auxiliares ---

def get_market_data(ticker):
    """Busca dados básicos do Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    
    # Tratamento de erro básico se dados faltarem
    current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0)
    
    data = {
        'price': current_price,
        'name': info.get('longName', ticker),
        'sector': info.get('sector', 'N/A'),
        'beta': info.get('beta', 1.0),
        'trailingPE': info.get('trailingPE', 0),
        'priceToBook': info.get('priceToBook', 0),
        'roe': info.get('returnOnEquity', 0.15), # Default 15% se faltar
        'dividendYield': info.get('dividendYield', 0.06) # Default 6%
    }
    return data, hist

def calculate_gordon_fair_value(dividend_per_share, ke, g):
    """Modelo de Gordon (Dividend Discount Model)."""
    if ke <= g:
        return 0 # Matematicamente impossível
    return dividend_per_share * (1 + g) / (ke - g)

def calculate_implied_irr(current_price, dividend_per_share, g):
    """Calcula a TIR implícita baseada em Gordon: k = D1/P0 + g"""
    if current_price <= 0: return 0
    d1 = dividend_per_share * (1 + g)
    return (d1 / current_price) + g

# --- Sidebar: Configurações do Analista ---
st.sidebar.title("⚙️ Painel de Controle CFA")
st.sidebar.header("1. Seleção do Ativo")
selected_ticker = st.sidebar.selectbox("Escolha o Ticker", TICKERS, index=0)

# Carregar Dados
data_raw, history = get_market_data(selected_ticker)

st.sidebar.header("2. Premissas Macro")
rf_rate = st.sidebar.number_input("Taxa Livre de Risco (Rf) %", value=10.5, step=0.1, help="Ex: NTN-B + Meta Inflação ou Selic Longa") / 100
erp = st.sidebar.number_input("Prêmio de Risco (ERP) %", value=5.0, step=0.1, help="Equity Risk Premium Brasil") / 100

st.sidebar.header("3. Premissas Específicas")
# Defaults inteligentes baseados nos dados baixados (se disponíveis)
default_beta = round(data_raw.get('beta', 1.0), 2)
beta_input = st.sidebar.number_input("Beta (β)", value=default_beta, step=0.05)

ke_calc = rf_rate + (beta_input * erp)
st.sidebar.info(f"🧬 Ke (Custo de Capital): {ke_calc:.2%}")

# --- Corpo Principal ---

st.title(f"📊 Análise Fundamentalista: {data_raw['name']}")
st.markdown(f"**Setor:** {data_raw['sector']} | **Preço Atual:** R$ {data_raw['price']:.2f}")

# --- 1. Análise Microeconômica (Qualitativa) ---
st.header("1️⃣ Análise Microeconômica & Moat")
col1, col2, col3 = st.columns(3)

with col1:
    moat_score = st.selectbox("Qualidade do Moat (Vantagem)", ["🟢 Largo (Wide)", "🟡 Estreito (Narrow)", "🔴 Nenhum (None)"], index=0)
with col2:
    pricing_power = st.selectbox("Poder de Precificação", ["Alto", "Médio", "Baixo"], index=1)
with col3:
    cycle_risk = st.selectbox("Sensibilidade ao Ciclo", ["Defensivo", "Cíclico Moderado", "Altamente Cíclico"], index=1)

st.markdown("---")

# --- 2. Análise Financeira (Normalizada) ---
st.header("2️⃣ KPIs Financeiros (TTM/Estimados)")

# Inputs interativos para normalizar o lucro/dividendo
col_fin1, col_fin2, col_fin3 = st.columns(3)
with col_fin1:
    roe_input = st.number_input("ROE Sustentável Estimado (%)", value=data_raw['roe']*100, step=0.5) / 100
with col_fin2:
    payout_input = st.number_input("Payout Ratio (%)", value=50.0, step=5.0) / 100
with col_fin3:
    # Estima LPA baseada no preço e P/L se disponível, ou input manual
    try:
        lpa_est = data_raw['price'] / data_raw['trailingPE'] if data_raw['trailingPE'] > 0 else 0
    except:
        lpa_est = 0
    lpa_input = st.number_input("LPA (Lucro/Ação) Normalizado R$", value=float(lpa_est), step=0.1)

dpa_implied = lpa_input * payout_input
st.write(f"ℹ️ *Dividendo/Ação Implícito nas premissas:* R$ {dpa_implied:.2f}")

st.markdown("---")

# --- 3. Valuation & Preço Justo ---
st.header("3️⃣ Valuation (Modelo Gordon / Renda Residual)")

growth_g = st.slider("Crescimento Perpétuo (g) %", 0.0, 10.0, 4.5, 0.1, help="Crescimento nominal (Inflação + PIB Real)") / 100

# Cálculo Central
fair_value = calculate_gordon_fair_value(dpa_implied, ke_calc, growth_g)
upside = (fair_value / data_raw['price']) - 1

col_val1, col_val2, col_val3 = st.columns(3)

with col_val1:
    st.markdown("### Preço Justo")
    st.markdown(f"<h2 style='color: #1f77b4;'>R$ {fair_value:.2f}</h2>", unsafe_allow_html=True)

with col_val2:
    st.markdown("### Upside / Downside")
    color = "green" if upside > 0 else "red"
    st.markdown(f"<h2 style='color: {color};'>{upside:.1%}</h2>", unsafe_allow_html=True)

with col_val3:
    st.markdown("### Margem de Segurança")
    ms_status = "🟢 Alta" if upside > 0.30 else "🟡 Moderada" if upside > 0.10 else "🔴 Baixa/Negativa"
    st.subheader(ms_status)

# Matriz de Sensibilidade
st.subheader("📐 Análise de Sensibilidade (Preço Justo)")

rates_sens = np.array([ke_calc - 0.01, ke_calc, ke_calc + 0.01])
growth_sens = np.array([growth_g - 0.01, growth_g, growth_g + 0.01])

sensitivity_data = []
for g_s in growth_sens:
    row = []
    for r_s in rates_sens:
        val = calculate_gordon_fair_value(dpa_implied, r_s, g_s)
        row.append(val)
    sensitivity_data.append(row)

df_sens = pd.DataFrame(sensitivity_data, 
                       columns=[f"Ke {(r*100):.1f}%" for r in rates_sens], 
                       index=[f"g {(g*100):.1f}%" for g in growth_sens])

st.dataframe(df_sens.style.format("R$ {:.2f}").background_gradient(cmap="RdYlGn", axis=None))

st.markdown("---")

# --- 4. TIR Implícita ---
st.header("4️⃣ TIR Implícita (Expectativa de Retorno)")

irr = calculate_implied_irr(data_raw['price'], dpa_implied, growth_g)

col_irr1, col_irr2 = st.columns(2)

with col_irr1:
    st.metric(label="TIR Real Estimada (Nominal)", value=f"{irr:.2%}")
    st.caption("Fórmula: Dividend Yield + Crescimento (g)")

with col_irr2:
    spread = irr - rf_rate
    st.metric(label="Prêmio sobre Renda Fixa", value=f"{spread:.2%}")
    if spread < 0.03:
        st.warning("⚠️ Prêmio de risco baixo (<3%). Considere Renda Fixa.")
    else:
        st.success("✅ Prêmio de risco atrativo.")

# --- 5. Conclusão ---
st.header("5️⃣ Veredito do Investimento")

recommendation = "NEUTRO"
if upside > 0.20 and irr > ke_calc:
    recommendation = "COMPRA"
    rec_color = "green"
elif upside < -0.10:
    recommendation = "VENDA"
    rec_color = "red"
else:
    recommendation = "MANTER / AGUARDAR"
    rec_color = "orange"

st.markdown(f"""
<div style="text-align: center; padding: 20px; background-color: #f8f9fa; border: 2px solid {rec_color}; border-radius: 10px;">
    <h1 style="color: {rec_color}; margin:0;">{recommendation}</h1>
    <p style="margin-top:10px;">Baseado nas premissas: Ke {ke_calc:.1%}, g {growth_g:.1%}, ROE {roe_input:.1%}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Disclaimer: Esta ferramenta é um modelo de simulação para fins educacionais e suporte à decisão. Não constitui recomendação direta de investimento. Preços podem ter delay de 15 minutos.")
