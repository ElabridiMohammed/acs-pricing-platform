"""
CRU Commodity Price Simulator & Formula Decision Platform
Tabs: Monte Carlo Simulation | Formula Lab & Decision
Author: Mohammed ELARIDI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from formula_engine import (
    load_historical_prices, load_phosphate_prices, load_base_data,
    compute_formula_1, compute_formula_2, compute_formula_3,
    compute_formula_4, compute_formula_5,
    apply_formula_to_scenario
)

st.set_page_config(
    page_title="ACS Pricing Decision Platform",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded"
)

STYLES = """
<style>
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .stApp header { background-color: transparent; }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #2d3a4f 0%, #1e2840 100%);
        border: 1px solid #3d4f6f; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label { color: #8fa3bf !important; font-size: 0.9rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #fff !important; font-size: 1.8rem; font-weight: 600; }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2840 0%, #0f1729 100%);
        border-right: 1px solid #2d3a4f;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label { color: #8fa3bf !important; font-weight: 500; }
    
    h1, h2, h3 { color: #ffffff !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #4a8cff 0%, #3366cc 100%);
        color: white; border: none; border-radius: 8px;
        padding: 10px 25px; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a9cff 0%, #4376dc 100%);
        box-shadow: 0 4px 15px rgba(74, 140, 255, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: rgba(30, 40, 64, 0.6); border-radius: 10px; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; color: #8fa3bf; font-weight: 600; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a8cff 0%, #3366cc 100%) !important;
        color: white !important;
    }
    
    .formula-card {
        background: linear-gradient(135deg, #2d3a4f 0%, #1e2840 100%);
        border: 1px solid #3d4f6f; border-radius: 12px; padding: 20px;
        margin: 10px 0;
    }
    .formula-card h4 { color: #4a8cff; margin: 0 0 10px 0; }
    .formula-card .value { font-size: 1.6rem; font-weight: 700; color: #fff; }
    .formula-card .label { color: #8fa3bf; font-size: 0.85rem; }
    
    .formula-eq-box {
        background: #0d1520; border: 1px solid #3d4f6f; border-radius: 10px;
        padding: 18px 24px; margin: 12px 0; text-align: center;
    }
    .formula-eq {
        font-family: 'Courier New', monospace; font-size: 1.2rem;
        color: #ffd700; font-weight: 600;
    }
    .formula-desc {
        color: #8fa3bf; font-size: 0.85rem; margin-top: 6px;
    }
    
    .gain { color: #00d26a !important; }
    .loss { color: #ff4757 !important; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ============================================================
# FORMULA DEFINITIONS
# ============================================================

FORMULA_LIST = [
    "F1 — Sulfur Indexing Only",
    "F2 — Smooth Sulfur Indexing",
    "F3 — Last Month ACS Indexing",
    "F4 — S & DAP Variation Indexing",
    "F5 — Smooth S & Smooth DAP Indexing",
]

FORMULA_EQUATIONS = {
    0: ("P = a × (S_weighted × conv_ratio + prod_cost) + b",
        "Price from sulfur spot prices with weighted ME/NA mix"),
    1: ("P = a × (Smooth_3m(S) × conv_ratio + prod_cost) + b",
        "Like F1 but uses 3-month rolling average of sulfur"),
    2: ("P = a × ACS_weighted(m-1)",
        "Last month's ACS price with regional weighting"),
    3: ("P = ACS₀ × (a + b × S/S₀ + c × DAP/DAP₀)",
        "Variation-based: tracks changes in sulfur and DAP vs reference"),
    4: ("P = ACS₀ × (a + b × Sm(S)/S₀ + c × Sm(DAP)/DAP₀)",
        "Like F4 but uses smoothed sulfur and DAP"),
}


# ============================================================
# SIMULATION ENGINE
# ============================================================

@st.cache_data
def parse_historical_data(file_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(file_path)
    raw = xl.parse('CRU historical', header=None)
    product_types = raw.iloc[0, 3:].fillna(method='ffill')
    regions = raw.iloc[1, 3:].fillna(method='ffill')
    metrics = raw.iloc[3, 3:]
    data = raw.iloc[4:].copy()
    data.columns = ['Year', 'Quarter', 'Month'] + list(range(len(data.columns) - 3))
    col_mapping = {}
    for i, (prod, reg, met) in enumerate(zip(product_types, regions, metrics)):
        if pd.notna(met):
            col_mapping[i] = f"{prod}_{reg}_{met}".replace(' ', '_').replace('/', '_')
    avg_cols = {k: v for k, v in col_mapping.items() if 'Avg' in str(v)}
    result = pd.DataFrame()
    result['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    result['Month'] = pd.to_numeric(data['Month'], errors='coerce')
    for col_idx, col_name in avg_cols.items():
        result[col_name.replace('_Avg', '')] = pd.to_numeric(data.iloc[:, col_idx + 3], errors='coerce')
    result = result.dropna(subset=['Year'])
    result['Date'] = pd.to_datetime(
        result['Year'].astype(int).astype(str) + '-' +
        result['Month'].astype(int).astype(str).str.zfill(2) + '-01'
    )
    return result.set_index('Date')

@st.cache_data
def parse_outlook_data(file_path: str, product: str = 'ACS') -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=f'{product} - CRU Outlook')
    df = df.rename(columns={df.columns[0]: 'Year'})
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data
def calculate_historical_volatility(hist_df: pd.DataFrame, column: str) -> float:
    if column not in hist_df.columns:
        return 0.15
    prices = hist_df[column].dropna()
    if len(prices) < 12:
        return 0.15
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return min(log_returns.std() * np.sqrt(12), 1.0)

def interpolate_trend(outlook_df, column, start_year, end_year, overrides=None):
    df = outlook_df[(outlook_df['Year'] >= start_year - 1) & (outlook_df['Year'] <= end_year + 1)]
    if column not in df.columns:
        st.warning(f"Column {column} not found")
        return None
    valid_data = df[['Year', column]].dropna()
    if len(valid_data) < 2:
        return None
    years = valid_data['Year'].values.copy()
    prices = valid_data[column].values.copy()
    if overrides:
        for yr, price in overrides.items():
            mask = years == yr
            if mask.any():
                prices[mask] = price
    try:
        f = interpolate.interp1d(years, prices, kind='cubic', fill_value='extrapolate')
    except:
        f = interpolate.interp1d(years, prices, kind='linear', fill_value='extrapolate')
    date_range = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='MS')
    return pd.Series(f(date_range.year + (date_range.month - 1) / 12), index=date_range, name='Trend')

def generate_brownian_motion(n_steps, volatility, n_sims=100, smoothing=0.7, seed=None):
    if seed is not None: np.random.seed(seed)
    monthly_vol = (volatility / np.sqrt(12)) * 0.6
    raw_shocks = np.random.normal(0, monthly_vol, (n_sims, n_steps))
    smoothed = np.zeros((n_sims, n_steps))
    smoothed[:, 0] = raw_shocks[:, 0]
    for t in range(1, n_steps):
        smoothed[:, t] = smoothing * smoothed[:, t-1] + (1 - smoothing) * raw_shocks[:, t]
    cumulative = np.cumsum(smoothed, axis=1)
    drift = -0.5 * (monthly_vol ** 2) * np.arange(1, n_steps + 1)
    return cumulative + drift

def generate_spike_process(n_steps, freq, intensity, persistence, decay_type='exponential',
                          strict_mode=False, n_spikes=None, seed=None):
    if seed is not None: np.random.seed(seed)
    contribution = np.zeros(n_steps)
    if strict_mode and n_spikes and n_spikes > 0:
        min_gap = max(persistence + 1, 3)
        slots = list(range(0, n_steps - persistence, min_gap))
        spike_times = sorted(np.random.choice(slots, size=min(n_spikes, len(slots)), replace=False)) if len(slots) >= n_spikes else slots[:n_spikes]
        for t in spike_times:
            for k in range(persistence + 1):
                if t + k < n_steps:
                    p = k / persistence
                    contribution[t + k] += intensity * (max(0, 1.0 - p) if decay_type == 'linear' else np.exp(-3.0 * p))
    else:
        for t in np.where(np.random.binomial(1, min(freq / 12.0, 0.3), n_steps))[0]:
            for k in range(persistence + 1):
                if t + k < n_steps:
                    p = k / persistence
                    contribution[t + k] += intensity * (max(0, 1.0 - p) if decay_type == 'linear' else np.exp(-3.0 * p))
    return contribution

def simulate_prices(trend, volatility, spike_freq, spike_intensity, spike_persistence,
                   decay_type, smoothing=0.7, strict_spikes=False, n_spikes=0,
                   n_sims=500, seed=None):
    n_steps = len(trend)
    trend_values = trend.values
    brownian = generate_brownian_motion(n_steps, volatility, n_sims, smoothing, seed)
    all_paths = np.zeros((n_sims, n_steps))
    best_idx, best_count = 0, 0
    for i in range(n_sims):
        spikes = generate_spike_process(n_steps, spike_freq, spike_intensity, spike_persistence,
                                        decay_type, strict_spikes, n_spikes,
                                        seed=(seed + i * 17) if seed else None)
        events = np.sum(spikes > spike_intensity * 0.5)
        if events > best_count and events <= 3:
            best_count, best_idx = events, i
        all_paths[i, :] = trend_values * np.exp(brownian[i, :]) * (1 + spikes)
    all_paths = np.maximum(all_paths, 1.0)
    sample_spikes = generate_spike_process(n_steps, spike_freq, spike_intensity, spike_persistence,
                                           decay_type, strict_spikes, n_spikes,
                                           seed=(seed + best_idx * 17) if seed else None)
    sample_path = np.maximum(trend_values * np.exp(brownian[best_idx, :]) * (1 + sample_spikes), 1.0)
    return {
        'dates': trend.index, 'trend': trend_values,
        'mean_path': np.mean(all_paths, axis=0),
        'percentile_5': np.percentile(all_paths, 5, axis=0),
        'percentile_25': np.percentile(all_paths, 25, axis=0),
        'percentile_75': np.percentile(all_paths, 75, axis=0),
        'percentile_95': np.percentile(all_paths, 95, axis=0),
        'sample_path': sample_path, 'all_paths': all_paths
    }


# ============================================================
# CHARTS
# ============================================================

CHART_BG = 'rgba(20,25,35,0.95)'
CHART_PAPER = 'rgba(0,0,0,0)'
GRID_COLOR = 'rgba(255,255,255,0.08)'
FONT_COLOR = '#9BA3AF'

def chart_layout(fig, title, height=500, yaxis_title='Price (USD/t)'):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='white'), x=0.5),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR, size=11)),
        yaxis=dict(title=yaxis_title, gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR, size=11)),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER,
        legend=dict(bgcolor='rgba(30,35,50,0.9)', font=dict(color='white', size=10),
                    orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
        hovermode='x unified', margin=dict(l=60, r=40, t=80, b=50), height=height
    )
    return fig


def create_projection_chart(results, product_name, scenario_idx=0):
    dates = results['dates']
    path = results['all_paths'][min(scenario_idx, results['all_paths'].shape[0] - 1), :]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=results['percentile_95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=results['percentile_5'], mode='lines', line=dict(width=0),
                              fill='tonexty', fillcolor='rgba(180,180,180,0.25)', name='95% Confidence', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=results['trend'], mode='lines',
                              line=dict(color='#DC143C', width=2.5, dash='dot'), name='CRU Outlook',
                              hovertemplate='<b>CRU</b><br>%{x|%b %Y}<br>$%{y:.1f}/t<extra></extra>'))
    fig.add_trace(go.Scatter(x=dates, y=path, mode='lines',
                              line=dict(color='#1E90FF', width=2.5), name=f'Scenario #{scenario_idx + 1}',
                              hovertemplate='<b>Scenario</b><br>%{x|%b %Y}<br>$%{y:.1f}/t<extra></extra>'))
    fig.add_trace(go.Scatter(x=dates, y=results['mean_path'], mode='lines',
                              line=dict(color='#4169E1', width=1.5, dash='dash'), name='MC Mean', visible='legendonly'))
    return chart_layout(fig, f'<b>{product_name} Price Projection</b>')


def create_distribution_chart(results, year):
    mask = results['dates'].year == year
    if not any(mask): return None
    prices = results['all_paths'][:, mask][:, -1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=prices, nbinsx=40, marker=dict(color='#4a8cff', line=dict(color='white', width=0.5)), opacity=0.8))
    fig.add_vline(x=np.mean(prices), line=dict(color='#ffaa33', width=2, dash='dash'), annotation_text=f'Mean: ${np.mean(prices):.0f}')
    return chart_layout(fig, f'<b>Distribution - {year}</b>', height=300, yaxis_title='Frequency')


def create_formula_vs_market_chart(bt, formula_name):
    """Chart: Formula price vs Market price over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt['Period'], y=bt['Market'], mode='lines+markers',
                              line=dict(color='#DC143C', width=3), name='Market (ACS CFR N.Africa)',
                              marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=bt['Period'], y=bt['Formula'], mode='lines+markers',
                              line=dict(color='#1E90FF', width=2.5), name=f'{formula_name}',
                              marker=dict(size=5)))
    return chart_layout(fig, f'<b>{formula_name} vs Market</b>')


def create_pnl_chart(bt, formula_name):
    """P&L waterfall for a single formula."""
    colors = ['#00d26a' if v >= 0 else '#ff4757' for v in bt['PnL']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bt['Period'], y=bt['PnL'],
        marker=dict(color=colors, line=dict(color='white', width=0.5)),
        hovertemplate='<b>%{x}</b><br>P&L: $%{y:.1f}/t<extra></extra>'))
    fig.add_hline(y=0, line=dict(color='white', width=1))
    avg = bt['PnL'].mean()
    fig.add_hline(y=avg, line=dict(color='#ffaa33', width=2, dash='dash'),
                  annotation_text=f'Avg: ${avg:.1f}/t')
    return chart_layout(fig, f'<b>Quarterly P&L — {formula_name}</b><br>'
                        f'<sup style="color:#888">Green = market above formula (buyer advantage)</sup>',
                        yaxis_title='P&L ($/t)')


def create_scenario_formula_chart(scenario_result, product_name):
    dates = scenario_result['dates']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=scenario_result['acs_index'], mode='lines',
                              line=dict(color='#DC143C', width=2, dash='dot'), name='ACS Market Index'))
    fig.add_trace(go.Scatter(x=dates, y=scenario_result['blended'], mode='lines',
                              line=dict(color='#ffaa33', width=2.5), name='Blended Formula'))
    fig.add_trace(go.Scatter(x=dates, y=scenario_result['capped'], mode='lines',
                              line=dict(color='#1E90FF', width=3), name='Final (Floor/Cap)'))
    if scenario_result['floor'] > 0:
        fig.add_hline(y=scenario_result['floor'], line=dict(color='#ff4757', width=1, dash='dash'),
                      annotation_text=f'Floor: ${scenario_result["floor"]:.0f}')
    if scenario_result['cap'] < 1000:
        fig.add_hline(y=scenario_result['cap'], line=dict(color='#ff4757', width=1, dash='dash'),
                      annotation_text=f'Cap: ${scenario_result["cap"]:.0f}')
    return chart_layout(fig, f'<b>{product_name} — Formula Applied to Scenario</b>')


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.markdown("""
    <div style='text-align: center; padding: 15px 0;'>
        <h1 style='color: #4a8cff; font-size: 2.3rem; margin-bottom: 5px;'>
            ACS Pricing Decision Platform
        </h1>
        <p style='color: #8fa3bf; font-size: 1rem;'>
            Monte Carlo Simulation  |  Formula Analysis  |  Revenue Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    DATA_PATH = Path(__file__).parent / "Formules de prix_ACS.xlsx"
    FORMULA_PATH = Path(__file__).parent / "0902_1700_ACS Pricing simulator_MG (1).xlsx"
    
    if not DATA_PATH.exists():
        DATA_PATH = Path("/Users/elabridi/Desktop/domaine project/Formules de prix_ACS.xlsx")
    
    tab1, tab2 = st.tabs(["Monte Carlo Simulation", "Formula Lab & Decision"])
    
    # ============================================================
    # TAB 1: MONTE CARLO
    # ============================================================
    with tab1:
        with st.sidebar:
            st.markdown("<h2 style='color: #4a8cff; text-align: center;'>Parameters</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown("### Product")
            product_type = st.selectbox("Commodity", ['Sulfuric Acid', 'Sulphur'], key='prod')
            regions = {
                'Sulfuric Acid': ['CFR US Gulf', 'CFR Brazil', 'FOB Japan/South Korea (Spot)',
                                 'FOB China', 'CFR Chile – contract', 'FOB NW Europe', 'CFR India'],
                'Sulphur': ['FOB Vancouver (spot)', 'FOB Middle East (spot)', 'CFR China  (spot)',
                           'CFR Brazil (spot)', 'CFR North Africa (contract)', 'FOB Tampa (contract)']
            }
            region = st.selectbox("Market / Region", regions[product_type], key='region')
            
            st.markdown("---")
            st.markdown("### Horizon")
            c1, c2 = st.columns(2)
            start_year = c1.selectbox("Start", [2025, 2026, 2027], index=0, key='sy')
            end_year = c2.selectbox("End", [2028, 2029, 2030], index=2, key='ey')
            
            st.markdown("---")
            st.markdown("### CRU Outlook (Editable)")
            cru_editable = st.checkbox("Override CRU values", value=False, key='cru_edit')
            cru_overrides = {}
            if cru_editable:
                for yr in range(start_year, end_year + 1):
                    cru_overrides[yr] = st.number_input(f"{yr} ($/t)", 50, 500, 140, 5, key=f'cru_{yr}')
            
            st.markdown("---")
            st.markdown("### Volatility")
            use_hist_vol = st.checkbox("Use Historical", value=True, key='hist_vol')
            vol_mult = st.slider("Multiplier", 0.5, 2.0, 1.0, 0.1, key='vm') if use_hist_vol else 1.0
            base_vol = None if use_hist_vol else st.slider("Base Vol (%)", 5, 50, 20, key='bv') / 100.0
            
            st.markdown("---")
            st.markdown("### Spikes")
            strict_spikes = st.checkbox("Strict Mode (fixed # of spikes)", value=False, key='strict')
            if strict_spikes:
                n_spikes = st.slider("Number of Spikes", 0, 10, 4, 1, key='ns')
                spike_freq = n_spikes / max(end_year - start_year + 1, 1)
            else:
                spike_freq = st.slider("Frequency (per year)", 0.0, 3.0, 0.5, 0.1, key='sf')
                n_spikes = 0
            spike_intensity = st.slider("Intensity (%)", 0, 100, 30, 5, key='si') / 100.0
            spike_persistence = st.slider("Persistence (months)", 1, 12, 4, key='sp')
            decay_type = st.radio("Decay", ['Exponential', 'Linear'], horizontal=True, key='dt').lower()
            
            st.markdown("---")
            st.markdown("### Simulation")
            n_sims = st.select_slider("MC Paths", [100, 250, 500, 1000, 2000], value=500, key='nsim')
            smoothing = st.slider("Smoothing", 0.3, 0.9, 0.7, 0.05, key='sm')
            seed = st.number_input("Random Seed", 0, 99999, 42, key='seed')
            
            st.markdown("---")
            run_btn = st.button("Run Simulation", use_container_width=True, type="primary")
        
        try:
            product_code = 'ACS' if product_type == 'Sulfuric Acid' else 'S'
            outlook_df = parse_outlook_data(str(DATA_PATH), product_code)
            try:
                hist_df = parse_historical_data(str(DATA_PATH))
                hist_loaded = True
            except:
                hist_df, hist_loaded = None, False
        except Exception as e:
            st.error(f"Could not load data: {e}")
            return
        
        if 'results' not in st.session_state:
            st.session_state.results = None
        
        if run_btn or st.session_state.results is None:
            with st.spinner('Running simulation...'):
                overrides = cru_overrides if cru_editable else None
                trend = interpolate_trend(outlook_df, region.replace('  ', ' '), start_year, end_year, overrides)
                if trend is None:
                    st.error(f"No data for region: {region}")
                    return
                if use_hist_vol and hist_loaded:
                    matching = [c for c in hist_df.columns if any(kw in c.lower() for kw in region.lower().split()[:2])]
                    calc_vol = calculate_historical_volatility(hist_df, matching[0]) if matching else 0.20
                    volatility = calc_vol * vol_mult
                else:
                    volatility = base_vol if base_vol else 0.20
                results = simulate_prices(trend, volatility, spike_freq, spike_intensity,
                                          spike_persistence, decay_type, smoothing,
                                          strict_spikes, n_spikes, n_sims, seed)
                results.update({'volatility_used': volatility, 'region': region, 'product': product_type,
                               'strict_spikes': strict_spikes, 'n_spikes': n_spikes})
                st.session_state.results = results
        
        results = st.session_state.results
        
        if results:
            st.markdown("### Key Metrics")
            cols = st.columns(5)
            avg_outlook = np.mean(results['trend'])
            avg_sim = np.mean(results['mean_path'])
            delta = ((avg_sim - avg_outlook) / avg_outlook) * 100
            cols[0].metric("CRU Outlook", f"${avg_outlook:.0f}/t")
            cols[1].metric("Simulated Mean", f"${avg_sim:.0f}/t", f"{delta:+.1f}%",
                          delta_color="inverse" if delta > 0 else "normal")
            cols[2].metric("95th Pctl Max", f"${np.max(results['percentile_95']):.0f}/t")
            cols[3].metric("5th Pctl Min", f"${np.min(results['percentile_5']):.0f}/t")
            cols[4].metric("Volatility", f"{results['volatility_used']*100:.1f}%")
            
            st.markdown("---")
            st.markdown("### Price Projection")
            n_scenarios = results['all_paths'].shape[0]
            scenario_idx = st.slider("Scenario", 1, n_scenarios, 1, key='sc_idx')
            
            st.plotly_chart(create_projection_chart(results, f"{results['product']} - {results['region']}",
                                                    scenario_idx - 1), use_container_width=True)
            
            st.markdown("### Price Distributions")
            dist_cols = st.columns(3)
            for i, year in enumerate([end_year - 2, end_year - 1, end_year]):
                chart = create_distribution_chart(results, year)
                if chart: dist_cols[i].plotly_chart(chart, use_container_width=True)
    
    # ============================================================
    # TAB 2: FORMULA LAB & DECISION
    # ============================================================
    with tab2:
        if not FORMULA_PATH.exists():
            st.error("Formula Excel file not found.")
            return
        
        st.markdown("### Formula Lab")
        
        # ---- FORMULA SELECTOR ----
        selected = st.selectbox("Choose a formula", FORMULA_LIST, key='formula_pick')
        formula_idx = FORMULA_LIST.index(selected)
        
        eq_text, eq_desc = FORMULA_EQUATIONS[formula_idx]
        st.markdown(f"""
        <div class='formula-eq-box'>
            <div class='formula-eq'>{eq_text}</div>
            <div class='formula-desc'>{eq_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ---- FORMULA-SPECIFIC PARAMETERS ----
        st.markdown("#### Parameters")
        
        hist = load_historical_prices(str(FORMULA_PATH))
        phos = load_phosphate_prices(str(FORMULA_PATH))
        
        if formula_idx == 0:
            # F1: Sulfur Indexing Only
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                f1_a = st.number_input("a (coefficient)", 0.5, 3.0, 1.3, 0.1, key='f1_a')
                f1_b = st.number_input("b (constant)", 0.0, 100.0, 25.0, 5.0, key='f1_b')
            with pc2:
                f1_s_me = st.slider("Sulfur ME weight", 0.0, 1.0, 0.7, 0.05, key='f1_sme')
                f1_s_na = st.slider("Sulfur NA weight", 0.0, 1.0, 0.3, 0.05, key='f1_sna')
            with pc3:
                f1_conv = st.slider("Conversion ratio", 0.1, 1.0, 0.33, 0.01, key='f1_conv')
                f1_prod = st.number_input("Production cost ($/t)", 5, 50, 20, 5, key='f1_prod')
            
            params = {'a': f1_a, 'b': f1_b, 'sulfur_me': f1_s_me, 'sulfur_na': f1_s_na,
                      'conv_ratio': f1_conv, 'prod_cost': f1_prod}
            bt = compute_formula_1(hist, params)
            formula_name = "Sulfur Indexing"

        elif formula_idx == 1:
            # F2: Smooth Sulfur Indexing
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                f2_a = st.number_input("a (coefficient)", 0.5, 3.0, 1.4, 0.1, key='f2_a')
                f2_b = st.number_input("b (constant)", 0.0, 100.0, 15.0, 5.0, key='f2_b')
            with pc2:
                f2_s_me = st.slider("Sulfur ME weight", 0.0, 1.0, 0.7, 0.05, key='f2_sme')
                f2_s_na = st.slider("Sulfur NA weight", 0.0, 1.0, 0.3, 0.05, key='f2_sna')
            with pc3:
                f2_conv = st.slider("Conversion ratio", 0.1, 1.0, 0.33, 0.01, key='f2_conv')
                f2_prod = st.number_input("Production cost ($/t)", 5, 50, 20, 5, key='f2_prod')
            
            params = {'a': f2_a, 'b': f2_b, 'sulfur_me': f2_s_me, 'sulfur_na': f2_s_na,
                      'conv_ratio': f2_conv, 'prod_cost': f2_prod}
            bt = compute_formula_2(hist, params)
            formula_name = "Smooth Sulfur"

        elif formula_idx == 2:
            # F3: Last Month ACS
            pc1, pc2 = st.columns(2)
            with pc1:
                f3_a = st.number_input("a (coefficient)", 0.5, 2.0, 1.0, 0.05, key='f3_a')
            with pc2:
                f3_na = st.slider("ACS N.Africa weight", 0.0, 1.0, 0.05, 0.05, key='f3_na')
                f3_eu = st.slider("ACS Europe weight", 0.0, 1.0, 0.75, 0.05, key='f3_eu')
                f3_jp = st.slider("ACS Japan weight", 0.0, 1.0, 0.0, 0.05, key='f3_jp')
                f3_ch = st.slider("ACS China weight", 0.0, 1.0, 0.20, 0.05, key='f3_ch')
            
            params = {'a': f3_a, 'acs_na': f3_na, 'acs_eu': f3_eu, 'acs_jp': f3_jp, 'acs_ch': f3_ch}
            bt = compute_formula_3(hist, params)
            formula_name = "Last Month ACS"

        elif formula_idx == 3:
            # F4: S & DAP Variation
            pc1, pc2 = st.columns(2)
            with pc1:
                f4_a = st.slider("a (fixed %)", 0.0, 1.0, 0.65, 0.05, key='f4_a')
                f4_b = st.slider("b (sulfur %)", 0.0, 1.0, 0.30, 0.05, key='f4_b')
                f4_c = st.slider("c (DAP %)", 0.0, 1.0, 0.05, 0.05, key='f4_c')
            with pc2:
                f4_acs0 = st.number_input("ACS0 (reference ACS $/t)", 50, 300, 110, 10, key='f4_acs0')
                f4_s0 = st.number_input("S0 (reference sulfur $/t)", 50, 300, 130, 10, key='f4_s0')
                f4_dap0 = st.number_input("DAP0 (reference DAP $/t)", 200, 1000, 500, 50, key='f4_dap0')
            
            params = {'a': f4_a, 'b': f4_b, 'c': f4_c, 'acs0': f4_acs0, 's0': f4_s0, 'dap0': f4_dap0}
            bt = compute_formula_4(hist, phos, params)
            formula_name = "S & DAP Variation"

        elif formula_idx == 4:
            # F5: Smooth S & Smooth DAP
            pc1, pc2 = st.columns(2)
            with pc1:
                f5_a = st.slider("a (fixed %)", 0.0, 1.0, 0.60, 0.05, key='f5_a')
                f5_b = st.slider("b (sulfur %)", 0.0, 1.0, 0.30, 0.05, key='f5_b')
                f5_c = st.slider("c (DAP %)", 0.0, 1.0, 0.10, 0.05, key='f5_c')
            with pc2:
                f5_acs0 = st.number_input("ACS0 (reference ACS $/t)", 50, 300, 110, 10, key='f5_acs0')
                f5_s0 = st.number_input("S0 (reference sulfur $/t)", 50, 300, 130, 10, key='f5_s0')
                f5_dap0 = st.number_input("DAP0 (reference DAP $/t)", 200, 1000, 500, 50, key='f5_dap0')
            
            params = {'a': f5_a, 'b': f5_b, 'c': f5_c, 'acs0': f5_acs0, 's0': f5_s0, 'dap0': f5_dap0}
            bt = compute_formula_5(hist, phos, params)
            formula_name = "Smooth S & DAP"
        
        # ---- FLOOR / CAP ----
        st.markdown("---")
        floor_cap_cols = st.columns(2)
        floor_val = floor_cap_cols[0].number_input("Floor ($/t)", 0, 200, 0, 10, key='floor')
        cap_val = floor_cap_cols[1].number_input("Cap ($/t)", 100, 500, 300, 10, key='cap')
        
        if floor_val > 0 or cap_val < 300:
            bt['Formula'] = bt['Formula'].clip(lower=floor_val, upper=cap_val)
            bt['PnL'] = bt['Market'] - bt['Formula']
        
        # ---- FILTER YEARS ----
        start_yr = st.select_slider("History from", list(range(2002, 2026)), value=2018, key='hist_start')
        bt_filtered = bt[bt['Year'] >= start_yr].copy()
        
        if bt_filtered.empty:
            st.warning("No data for the selected period.")
        else:
            st.markdown("---")
            
            # ---- CHARTS ----
            st.plotly_chart(create_formula_vs_market_chart(bt_filtered, formula_name), use_container_width=True)
            
            st.plotly_chart(create_pnl_chart(bt_filtered, formula_name), use_container_width=True)
            
            # ---- SUMMARY METRICS ----
            st.markdown("### Performance Summary")
            
            avg_pnl = bt_filtered['PnL'].mean()
            total_pnl = bt_filtered['PnL'].sum()
            pct_positive = (bt_filtered['PnL'] >= 0).mean() * 100
            max_loss = bt_filtered['PnL'].min()
            max_gain = bt_filtered['PnL'].max()
            
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            
            clr = '#00d26a' if avg_pnl >= 0 else '#ff4757'
            mc1.markdown(f"""
            <div class='formula-card'>
                <h4>Avg P&L</h4>
                <div class='value' style='color: {clr}'>${avg_pnl:+.1f}/t</div>
                <div class='label'>Per quarter</div>
            </div>
            """, unsafe_allow_html=True)
            
            mc2.markdown(f"""
            <div class='formula-card'>
                <h4>Win Rate</h4>
                <div class='value' style='color: {"#00d26a" if pct_positive >= 50 else "#ff4757"}'>{pct_positive:.0f}%</div>
                <div class='label'>of quarters formula below market</div>
            </div>
            """, unsafe_allow_html=True)
            
            mc3.markdown(f"""
            <div class='formula-card'>
                <h4>Best Quarter</h4>
                <div class='value' style='color: #00d26a'>${max_gain:+.1f}/t</div>
                <div class='label'>Maximum advantage</div>
            </div>
            """, unsafe_allow_html=True)
            
            mc4.markdown(f"""
            <div class='formula-card'>
                <h4>Worst Quarter</h4>
                <div class='value' style='color: #ff4757'>${max_loss:+.1f}/t</div>
                <div class='label'>Maximum disadvantage</div>
            </div>
            """, unsafe_allow_html=True)
            
            vol_kt = 750
            annual_impact = avg_pnl * vol_kt * 1000 / 1e6
            mc5.markdown(f"""
            <div class='formula-card'>
                <h4>Annual Impact</h4>
                <div class='value' style='color: {clr}'>${annual_impact:+.1f}M</div>
                <div class='label'>at {vol_kt} Kt/yr</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Detailed Quarterly Data"):
                st.dataframe(bt_filtered[['Period', 'Market', 'Formula', 'PnL']].round(1),
                             use_container_width=True, hide_index=True)
        
        # ============================================================
        # DECISION SECTION
        # ============================================================
        st.markdown("---")
        st.markdown("### Decision — Formula x Monte Carlo")
        
        results = st.session_state.get('results')
        
        if results is None:
            st.info("Run a Monte Carlo simulation in the first tab to enable scenario-based formula analysis.")
        else:
            vol_c1, vol_c2 = st.columns([1, 3])
            annual_vol = vol_c1.number_input("Annual Volume (Kt)", 100, 5000, 750, 50, key='vol_kt')
            
            base = load_base_data(str(FORMULA_PATH))
            alpha = base.get('alpha', 0.7)
            beta = base.get('beta', 0.3)
            
            formula_weights = {
                'alpha': alpha, 'beta': beta, 'gamma': 0,
                'acs_from_sulphur': base.get('acs_from_sulphur', 3.02),
                'conversion_cost': base.get('conversion_cost', 20)
            }
            
            n_sc = results['all_paths'].shape[0]
            sc_idx = st.slider("Select Monte Carlo Scenario", 1, n_sc, 1, key='d_sc')
            scenario_prices = results['all_paths'][sc_idx - 1, :]
            
            scenario_result = apply_formula_to_scenario(
                scenario_prices, results['dates'], formula_weights, floor_val, cap_val
            )
            
            monthly_vol = annual_vol * 1000 / 12
            market_revenue = np.sum(scenario_prices * monthly_vol) / 1e6
            formula_revenue = np.sum(scenario_result['capped'] * monthly_vol) / 1e6
            savings = market_revenue - formula_revenue
            
            all_savings = []
            for i in range(n_sc):
                sc_prices = results['all_paths'][i, :]
                sc_result = apply_formula_to_scenario(sc_prices, results['dates'], formula_weights, floor_val, cap_val)
                sc_market_rev = np.sum(sc_prices * monthly_vol) / 1e6
                sc_formula_rev = np.sum(sc_result['capped'] * monthly_vol) / 1e6
                all_savings.append(sc_market_rev - sc_formula_rev)
            all_savings = np.array(all_savings)
            
            dc1, dc2, dc3, dc4 = st.columns(4)
            
            dc1.markdown(f"""
            <div class='formula-card'>
                <h4>Market Cost</h4>
                <div class='value'>${market_revenue:.1f}M</div>
                <div class='label'>Spot prices (Scenario #{sc_idx})</div>
            </div>
            """, unsafe_allow_html=True)
            
            dc2.markdown(f"""
            <div class='formula-card'>
                <h4>Formula Cost</h4>
                <div class='value'>${formula_revenue:.1f}M</div>
                <div class='label'>Under blended formula</div>
            </div>
            """, unsafe_allow_html=True)
            
            clr = '#00d26a' if savings >= 0 else '#ff4757'
            dc3.markdown(f"""
            <div class='formula-card'>
                <h4>This Scenario</h4>
                <div class='value' style='color: {clr}'>${savings:+.1f}M</div>
                <div class='label'>{'Savings' if savings >= 0 else 'Extra cost'} vs market</div>
            </div>
            """, unsafe_allow_html=True)
            
            avg_save = np.mean(all_savings)
            clr_avg = '#00d26a' if avg_save >= 0 else '#ff4757'
            dc4.markdown(f"""
            <div class='formula-card'>
                <h4>Expected (All {n_sc})</h4>
                <div class='value' style='color: {clr_avg}'>${avg_save:+.1f}M</div>
                <div class='label'>Mean savings across MC paths</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(create_scenario_formula_chart(scenario_result, 
                            f"{results['product']} - Scenario #{sc_idx}"), use_container_width=True)
            
            # Risk analysis
            st.markdown("### Risk Analysis")
            risk_cols = st.columns(3)
            p5 = np.percentile(all_savings, 5)
            p95 = np.percentile(all_savings, 95)
            prob_save = np.mean(all_savings >= 0) * 100
            
            risk_cols[0].markdown(f"""
            <div class='formula-card'>
                <h4>Best Case (P95)</h4>
                <div class='value' style='color: #00d26a'>${p95:+.1f}M</div>
                <div class='label'>Savings in favorable scenario</div>
            </div>
            """, unsafe_allow_html=True)
            
            risk_cols[1].markdown(f"""
            <div class='formula-card'>
                <h4>Worst Case (P5)</h4>
                <div class='value' style='color: {"#00d26a" if p5 >= 0 else "#ff4757"}'>${p5:+.1f}M</div>
                <div class='label'>Cost in adverse scenario</div>
            </div>
            """, unsafe_allow_html=True)
            
            risk_cols[2].markdown(f"""
            <div class='formula-card'>
                <h4>Probability of Savings</h4>
                <div class='value' style='color: {"#00d26a" if prob_save >= 50 else "#ff4757"}'>{prob_save:.0f}%</div>
                <div class='label'>of scenarios show net savings</div>
            </div>
            """, unsafe_allow_html=True)
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=all_savings, nbinsx=50,
                                             marker=dict(color='#4a8cff', line=dict(color='white', width=0.5)), opacity=0.8))
            fig_dist.add_vline(x=0, line=dict(color='white', width=2))
            fig_dist.add_vline(x=avg_save, line=dict(color='#ffaa33', width=2, dash='dash'),
                              annotation_text=f'Mean: ${avg_save:+.1f}M')
            chart_layout(fig_dist, '<b>Savings Distribution Across All Scenarios</b>',
                        height=350, yaxis_title='Frequency')
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8fa3bf; font-size: 0.85rem;'>
        Built by Mohammed ELARIDI | Data: CRU Outlook & ACS Pricing Simulator
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
