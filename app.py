"""
CRU Commodity Price Simulator & Formula Decision Platform
Tabs: Monte Carlo Simulation | Formula Lab & Decision
Author: Mohammed ELABRIDI
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
    load_historical_prices, load_historical_prices_extended,
    load_phosphate_prices,
    compute_formula_1, compute_formula_2, compute_formula_3,
    compute_formula_4, compute_formula_5, compute_formula_6,
)
from correlation_engine import (
    load_historical_monthly, load_generated_yearly,
    compute_correlation_matrix, apply_correlated_adjustment,
    expand_yearly_to_monthly, CORR_VARS,
)

st.set_page_config(
    page_title="ACS Pricing Decision Platform",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIGHT-MODE STYLES ---

STYLES = """
<style>
    .stApp { background-color: #f8f9fc; }
    .stApp header { background-color: transparent; }
    
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e2e6ed; border-radius: 12px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    div[data-testid="metric-container"] label { color: #5a6a85 !important; font-size: 0.9rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #1a2332 !important; font-size: 1.8rem; font-weight: 600; }
    
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e6ed;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label { color: #3a4a65 !important; font-weight: 500; }
    
    h1, h2, h3 { color: #1a2332 !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: none; border-radius: 8px;
        padding: 10px 25px; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #eef1f6; border-radius: 10px; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; color: #5a6a85; font-weight: 600; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
    }
    
    .formula-card {
        background: #ffffff;
        border: 1px solid #e2e6ed; border-radius: 12px; padding: 20px;
        margin: 10px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .formula-card h4 { color: #2563eb; margin: 0 0 10px 0; }
    .formula-card .value { font-size: 1.6rem; font-weight: 700; color: #1a2332; }
    .formula-card .label { color: #5a6a85; font-size: 0.85rem; }
    
    .formula-eq-box {
        background: #f0f4ff; border: 1px solid #d0daf0; border-radius: 10px;
        padding: 18px 24px; margin: 12px 0; text-align: center;
    }
    .formula-eq {
        font-family: 'Courier New', monospace; font-size: 1.2rem;
        color: #1d4ed8; font-weight: 600;
    }
    .formula-desc {
        color: #5a6a85; font-size: 0.85rem; margin-top: 6px;
    }
    
    .gain { color: #16a34a !important; }
    .loss { color: #dc2626 !important; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# --- FORMULA DEFINITIONS ---

FORMULA_LIST = [
    "F1 — Sulfur Indexing Only",
    "F2 — Smooth Sulfur Indexing",
    "F3 — Last Month ACS Indexing",
    "F4 — S & DAP Variation Indexing",
    "F5 — Smooth S & Smooth DAP Indexing",
    "F6 — S, DAP, Petcoke & Clinker Indexing",
]

FORMULA_EQUATIONS = {
    0: ("P = a × (S_weighted × conv_ratio + prod_cost) + b",
        "Direct sulfur indexing with weighted ME/NA mix"),
    1: ("P = a × (Smooth(S) × conv_ratio + prod_cost) + b",
        "Like F1 but uses smoothed rolling average of sulfur"),
    2: ("P = a × ACS_weighted(m-1)",
        "Last month's ACS price with regional weighting (NA/EU/JP/CH)"),
    3: ("P = ACS₀ × (a + b × S/S₀ + c × DAP/DAP₀)",
        "Variation-based: tracks changes in sulfur and DAP vs reference"),
    4: ("P = ACS₀ × (a + b × Sm(S)/S₀ + c × Sm(DAP)/DAP₀)",
        "Like F4 but uses smoothed sulfur and DAP inputs"),
    5: ("P = ACS₀ × (a + b·S/S₀ + c·DAP/DAP₀ + d·PC/PC₀ + e·(1−CLK/CLK₀))",
        "Full cost-stack variation: sulfur + DAP + petcoke + clinker"),
}


# --- SIMULATION ENGINE ---

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
            else:
                years = np.append(years, yr)
                prices = np.append(prices, price)
                order = np.argsort(years)
                years, prices = years[order], prices[order]
    try:
        f = interpolate.interp1d(years, prices, kind='cubic', fill_value=(prices[0], prices[-1]), bounds_error=False)
    except:
        f = interpolate.interp1d(years, prices, kind='linear', fill_value=(prices[0], prices[-1]), bounds_error=False)
    date_range = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='MS')
    trend_vals = f(date_range.year + (date_range.month - 1) / 12)
    trend_vals = np.maximum(trend_vals, 10.0)  # floor at $10/t
    return pd.Series(trend_vals, index=date_range, name='Trend')

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


# --- CHARTS (Light-mode color scheme) ---

CHART_BG = '#ffffff'
CHART_PAPER = 'rgba(0,0,0,0)'
GRID_COLOR = 'rgba(0,0,0,0.07)'
FONT_COLOR = '#4a5568'

def chart_layout(fig, title, height=500, yaxis_title='Price (USD/t)'):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1a2332'), x=0.5),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR, size=11)),
        yaxis=dict(title=yaxis_title, gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR, size=11),
                   title_font=dict(color=FONT_COLOR)),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER,
        legend=dict(bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a2332', size=10),
                    bordercolor='#e2e6ed', borderwidth=1,
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
                              fill='tonexty', fillcolor='rgba(37,99,235,0.12)', name='95% Confidence', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=results['trend'], mode='lines',
                              line=dict(color='#dc2626', width=2.5, dash='dot'), name='CRU Outlook',
                              hovertemplate='<b>CRU</b><br>%{x|%b %Y}<br>$%{y:.1f}/t<extra></extra>'))
    fig.add_trace(go.Scatter(x=dates, y=path, mode='lines',
                              line=dict(color='#2563eb', width=2.5), name=f'Scenario #{scenario_idx + 1}',
                              hovertemplate='<b>Scenario</b><br>%{x|%b %Y}<br>$%{y:.1f}/t<extra></extra>'))
    fig.add_trace(go.Scatter(x=dates, y=results['mean_path'], mode='lines',
                              line=dict(color='#6366f1', width=1.5, dash='dash'), name='MC Mean', visible='legendonly'))
    return chart_layout(fig, f'<b>{product_name} Price Projection</b>')


def create_distribution_chart(results, year):
    mask = results['dates'].year == year
    if not any(mask): return None
    prices = results['all_paths'][:, mask][:, -1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=prices, nbinsx=40,
                                marker=dict(color='#2563eb', line=dict(color='white', width=0.5)), opacity=0.8))
    fig.add_vline(x=np.mean(prices), line=dict(color='#f59e0b', width=2, dash='dash'),
                  annotation_text=f'Mean: ${np.mean(prices):.0f}')
    return chart_layout(fig, f'<b>Distribution - {year}</b>', height=300, yaxis_title='Frequency')


def create_formula_vs_market_chart(bt, formula_name, floor=None, cap=None):
    """Chart: Formula price vs Market price over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt['Period'], y=bt['Market'], mode='lines+markers',
                              line=dict(color='#dc2626', width=3), name='Market (ACS CFR N.Africa)',
                              marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=bt['Period'], y=bt['Formula'], mode='lines+markers',
                              line=dict(color='#2563eb', width=2.5), name=f'{formula_name}',
                              marker=dict(size=5)))
    if floor is not None:
        fig.add_hline(y=floor, line=dict(color='#f59e0b', width=1.5, dash='dash'),
                      annotation_text=f'Floor: ${floor}')
    if cap is not None:
        fig.add_hline(y=cap, line=dict(color='#f59e0b', width=1.5, dash='dash'),
                      annotation_text=f'Cap: ${cap}')
    return chart_layout(fig, f'<b>{formula_name} vs Market</b>')


def create_pnl_chart(bt, formula_name, view_mode='quarterly'):
    """P&L chart for a single formula. Supports quarterly and annual aggregation."""
    if view_mode == 'annual' and 'Year' in bt.columns:
        agg = bt.groupby('Year').agg({'PnL': 'mean', 'Period': 'first'}).reset_index()
        agg['Period'] = agg['Year'].astype(str)
        data = agg
        period_label = 'Annual'
    else:
        data = bt
        period_label = 'Quarterly'
    
    colors = ['#16a34a' if v >= 0 else '#dc2626' for v in data['PnL']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Period'], y=data['PnL'],
        marker=dict(color=colors, line=dict(color='white', width=0.5)),
        hovertemplate='<b>%{x}</b><br>P&L: $%{y:.1f}/t<extra></extra>'))
    fig.add_hline(y=0, line=dict(color='#94a3b8', width=1))
    avg = data['PnL'].mean()
    fig.add_hline(y=avg, line=dict(color='#f59e0b', width=2, dash='dash'),
                  annotation_text=f'Avg: ${avg:.1f}/t')
    return chart_layout(fig, f'<b>{period_label} P&L — {formula_name}</b><br>'
                        f'<sup style="color:#6b7280">Green = market above formula (buyer advantage)</sup>',
                        yaxis_title='P&L ($/t)')


def create_scenario_formula_chart(scenario_result, product_name):
    dates = scenario_result['dates']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=scenario_result['acs_index'], mode='lines',
                              line=dict(color='#dc2626', width=2, dash='dot'), name='ACS Market Index'))
    fig.add_trace(go.Scatter(x=dates, y=scenario_result['blended'], mode='lines',
                              line=dict(color='#f59e0b', width=2.5), name='Blended Formula'))
    fig.add_trace(go.Scatter(x=dates, y=scenario_result['capped'], mode='lines',
                              line=dict(color='#2563eb', width=3), name='Final (Floor/Cap)'))
    if scenario_result['floor'] > 0:
        fig.add_hline(y=scenario_result['floor'], line=dict(color='#dc2626', width=1, dash='dash'),
                      annotation_text=f'Floor: ${scenario_result["floor"]:.0f}')
    if scenario_result['cap'] < 1000:
        fig.add_hline(y=scenario_result['cap'], line=dict(color='#dc2626', width=1, dash='dash'),
                      annotation_text=f'Cap: ${scenario_result["cap"]:.0f}')
    return chart_layout(fig, f'<b>{product_name} — Formula Applied to Scenario</b>')


# --- MAIN APP ---

def main():
    st.markdown("""
    <div style='text-align: center; padding: 15px 0;'>
        <h1 style='color: #2563eb; font-size: 2.3rem; margin-bottom: 5px;'>
            ACS Pricing Decision Platform
        </h1>
        <p style='color: #5a6a85; font-size: 1rem;'>
            Monte Carlo Simulation  |  Formula Analysis  |  Revenue Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    DATA_PATH = Path(__file__).parent / "Formules de prix_ACS.xlsx"
    NEW_FORMULA_PATH = Path(__file__).parent / "1202_2001_ACS Pricing simulator.xlsx"
    OLD_FORMULA_PATH = Path(__file__).parent / "1202_1203_ACS Pricing simulator.xlsx"
    LEGACY_FORMULA_PATH = Path(__file__).parent / "0902_1700_ACS Pricing simulator_MG (1).xlsx"
    FORMULA_PATH = NEW_FORMULA_PATH if NEW_FORMULA_PATH.exists() else (OLD_FORMULA_PATH if OLD_FORMULA_PATH.exists() else LEGACY_FORMULA_PATH)
    
    if not DATA_PATH.exists():
        DATA_PATH = Path("/Users/elabridi/Desktop/domaine project/Formules de prix_ACS.xlsx")
    
    # Load correlation data from new Excel
    _corr_data_loaded = False
    _corr_matrix = None
    _generated_yearly = None
    if NEW_FORMULA_PATH.exists():
        try:
            _hist_monthly = load_historical_monthly(str(NEW_FORMULA_PATH))
            _generated_yearly = load_generated_yearly(str(NEW_FORMULA_PATH))
            _corr_matrix = compute_correlation_matrix(_hist_monthly)
            _corr_data_loaded = True
        except Exception as e:
            st.warning(f"Could not load correlation data: {e}")
    
    tab1, tab2 = st.tabs(["Monte Carlo Simulation", "Formula Lab & Decision"])
    
    # ============================================================
    # TAB 1: MONTE CARLO
    # ============================================================
    with tab1:
        with st.sidebar:
            st.markdown("<h2 style='color: #2563eb; text-align: center;'>Parameters</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown("### Product")
            # Variables from the Monthly_prices_forecast_direct sheet
            VARIABLE_OPTIONS = [
                'ACS CFR North Africa',
                'ACS NW EU',
                'ACS Japan',
                'ACS China',
                'S ME to CFR',
                'S North Africa to CFR Morocco',
                'Smooth 3 mois S ME to CFR',
                'Smooth S North Africa to CFR Morocco',
                'DAP Bulk North Africa',
            ]
            # Map each variable to (product_code, CRU_outlook_column)
            VARIABLE_TO_CRU = {
                'ACS CFR North Africa': ('ACS', 'CFR US Gulf'),
                'ACS NW EU': ('ACS', 'FOB NW Europe'),
                'ACS Japan': ('ACS', 'FOB Japan/South Korea (Spot)'),
                'ACS China': ('ACS', 'FOB China'),
                'S ME to CFR': ('S', 'FOB Middle East (spot)'),
                'S North Africa to CFR Morocco': ('S', 'CFR North Africa (contract)'),
                'Smooth 3 mois S ME to CFR': ('S', 'FOB Middle East (spot)'),
                'Smooth S North Africa to CFR Morocco': ('S', 'CFR North Africa (contract)'),
                'DAP Bulk North Africa': ('ACS', 'CFR US Gulf'),
            }
            selected_variable = st.selectbox("Variable", VARIABLE_OPTIONS, key='prod_var')
            product_code_map, region = VARIABLE_TO_CRU[selected_variable]
            product_type = 'Sulfuric Acid' if product_code_map == 'ACS' else 'Sulphur'
            
            st.markdown("---")
            st.markdown("### Horizon")
            c1, c2 = st.columns(2)
            start_year = c1.selectbox("Start", [2025, 2026, 2027], index=0, key='sy')
            end_year = c2.selectbox("End", [2030, 2032, 2035], index=2, key='ey')
            
            st.markdown("---")
            st.markdown("### CRU Outlook (Editable)")
            cru_editable = st.checkbox("Override CRU values", value=False, key='cru_edit')
            cru_overrides = {}
            if cru_editable:
                for yr in range(start_year, end_year + 1):
                    cru_overrides[yr] = st.number_input(f"{yr} ($/t)", 50, 500, 140, 5, key=f'cru_{yr}')
            
            st.markdown("---")
            st.markdown("### Petcoke & Clinker Outlook")
            pc_clk_editable = st.checkbox("Override Petcoke / Clinker", value=False, key='pc_clk_edit')
            petcoke_outlook = {}
            clinker_outlook = {}
            if pc_clk_editable:
                for yr in range(start_year, end_year + 1):
                    pc_c1, pc_c2 = st.columns(2)
                    petcoke_outlook[yr] = pc_c1.number_input(f"PC {yr}", 50, 400, 140, 5, key=f'pc_{yr}')
                    clinker_outlook[yr] = pc_c2.number_input(f"CLK {yr}", 50, 400, 130, 5, key=f'clk_{yr}')

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
        
        # DATA SOURCE
        use_new = NEW_FORMULA_PATH.exists()
        use_old = OLD_FORMULA_PATH.exists()
        
        # FORMULA SELECTOR
        available_formulas = FORMULA_LIST if use_new else FORMULA_LIST[:5]  # F6 only with new file
        selected = st.selectbox("Choose a formula", available_formulas, key='formula_pick')
        formula_idx = FORMULA_LIST.index(selected)
        
        eq_text, eq_desc = FORMULA_EQUATIONS[formula_idx]
        st.markdown(f"""
        <div class='formula-eq-box'>
            <div class='formula-eq'>{eq_text}</div>
            <div class='formula-desc'>{eq_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # VIEW TOGGLE
        view_mode = st.radio("View", ["Quarterly", "Annual"], horizontal=True, key='view_mode')
        view_key = 'annual' if view_mode == 'Annual' else 'quarterly'
        
        # LOAD DATA (backtest only — historical up to today)
        if _corr_data_loaded and _hist_monthly is not None:
            hist = _hist_monthly.copy()
            # Add IPP columns for compatibility with F3
            if 'IPP_Europe' not in hist.columns:
                hist['IPP_Europe'] = hist.get('ACS_NWE', 0)
            if 'IPP_Japan' not in hist.columns:
                hist['IPP_Japan'] = hist.get('ACS_Japan', 0)
            if 'IPP_China' not in hist.columns:
                hist['IPP_China'] = hist.get('ACS_China', 0)
        elif OLD_FORMULA_PATH.exists():
            hist = load_historical_prices_extended(str(OLD_FORMULA_PATH))
        else:
            hist = load_historical_prices(str(LEGACY_FORMULA_PATH))
        phos = load_phosphate_prices(str(FORMULA_PATH))
        
        # Filter to historical only (up to current year)
        if hist is not None and 'Year' in hist.columns:
            hist = hist[hist['Year'] <= 2025].copy()
        
        # FORMULA-SPECIFIC PARAMETERS
        st.markdown("#### Parameters")
        
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
            floor_def, cap_def = 110, 220
            params = {'a': f1_a, 'b': f1_b, 'sulfur_me': f1_s_me, 'sulfur_na': f1_s_na,
                      'conv_ratio': f1_conv, 'prod_cost': f1_prod}
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
                f2_window = st.slider("Smooth window (months)", 3, 12, 6, 1, key='f2_window')
            floor_def, cap_def = 110, 220
            params = {'a': f2_a, 'b': f2_b, 'sulfur_me': f2_s_me, 'sulfur_na': f2_s_na,
                      'conv_ratio': f2_conv, 'prod_cost': f2_prod, 'smooth_window': f2_window}
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
            floor_def, cap_def = 110, 220
            params = {'a': f3_a, 'acs_na': f3_na, 'acs_eu': f3_eu, 'acs_jp': f3_jp, 'acs_ch': f3_ch}
            formula_name = "Last Month ACS"

        elif formula_idx == 3:
            # F4: S & DAP Variation
            pc1, pc2 = st.columns(2)
            with pc1:
                f4_a = st.slider("a (fixed %)", 0.0, 1.0, 0.70, 0.05, key='f4_a')
                f4_b = st.slider("b (sulfur %)", 0.0, 1.0, 0.10, 0.05, key='f4_b')
                f4_c = st.slider("c (DAP %)", 0.0, 1.0, 0.20, 0.05, key='f4_c')
            with pc2:
                f4_acs0 = st.number_input("ACS0 (reference ACS $/t)", 50, 300, 110, 10, key='f4_acs0')
                f4_s0 = st.number_input("S0 (reference sulfur $/t)", 50, 300, 130, 10, key='f4_s0')
                f4_dap0 = st.number_input("DAP0 (reference DAP $/t)", 200, 1000, 500, 50, key='f4_dap0')
            floor_def, cap_def = 110, 230
            params = {'a': f4_a, 'b': f4_b, 'c': f4_c, 'acs0': f4_acs0, 's0': f4_s0,
                      'dap0': f4_dap0, '_phos': phos}
            formula_name = "S & DAP Variation"

        elif formula_idx == 4:
            # F5: Smooth S & Smooth DAP
            pc1, pc2 = st.columns(2)
            with pc1:
                f5_a = st.slider("a (fixed %)", 0.0, 1.0, 0.80, 0.05, key='f5_a')
                f5_b = st.slider("b (sulfur %)", 0.0, 1.0, 0.10, 0.05, key='f5_b')
                f5_c = st.slider("c (DAP %)", 0.0, 1.0, 0.10, 0.05, key='f5_c')
            with pc2:
                f5_acs0 = st.number_input("ACS0 (reference ACS $/t)", 50, 300, 110, 10, key='f5_acs0')
                f5_s0 = st.number_input("S0 (reference sulfur $/t)", 50, 300, 130, 10, key='f5_s0')
                f5_dap0 = st.number_input("DAP0 (reference DAP $/t)", 200, 1000, 500, 50, key='f5_dap0')
                f5_window = st.slider("Smooth window (months)", 3, 12, 6, 1, key='f5_window')
            floor_def, cap_def = 110, 230
            params = {'a': f5_a, 'b': f5_b, 'c': f5_c, 'acs0': f5_acs0, 's0': f5_s0,
                      'dap0': f5_dap0, 'smooth_window': f5_window, '_phos': phos}
            formula_name = "Smooth S & DAP"

        elif formula_idx == 5:
            # F6: S, DAP, Petcoke & Clinker
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                f6_a = st.slider("a (fixed %)", 0.0, 1.0, 0.60, 0.05, key='f6_a')
                f6_b = st.slider("b (sulfur %)", 0.0, 1.0, 0.05, 0.01, key='f6_b')
                f6_c = st.slider("c (DAP %)", 0.0, 1.0, 0.20, 0.05, key='f6_c')
                f6_d = st.slider("d (petcoke %)", 0.0, 1.0, 0.05, 0.01, key='f6_d')
                f6_e = st.slider("e (clinker %)", 0.0, 1.0, 0.10, 0.01, key='f6_e')
            with pc2:
                f6_acs0 = st.number_input("ACS0 ($/t)", 50, 300, 110, 10, key='f6_acs0')
                f6_s0 = st.number_input("S0 ($/t)", 50, 300, 130, 10, key='f6_s0')
                f6_dap0 = st.number_input("DAP0 ($/t)", 200, 1000, 500, 50, key='f6_dap0')
            with pc3:
                f6_pc0 = st.number_input("PC0 (petcoke ref $/t)", 50, 300, 140, 10, key='f6_pc0')
                f6_clk0 = st.number_input("CLK0 (clinker ref $/t)", 50, 300, 130, 10, key='f6_clk0')
            floor_def, cap_def = 110, 230
            params = {'a': f6_a, 'b': f6_b, 'c': f6_c, 'd': f6_d, 'e': f6_e,
                      'acs0': f6_acs0, 's0': f6_s0, 'dap0': f6_dap0, 'pc0': f6_pc0, 'clk0': f6_clk0}
            formula_name = "S, DAP, Petcoke & Clinker"
        
        # FLOOR / CAP
        st.markdown("---")
        floor_cap_cols = st.columns(2)
        floor_val = floor_cap_cols[0].number_input("Floor ($/t)", 0, 300, floor_def, 10, key='floor')
        cap_val = floor_cap_cols[1].number_input("Cap ($/t)", 100, 500, cap_def, 10, key='cap')
        
        params['floor'] = floor_val
        params['cap'] = cap_val
        
        # COMPUTE (backtest only)
        compute_fns = [compute_formula_1, compute_formula_2, compute_formula_3,
                       compute_formula_4, compute_formula_5, compute_formula_6]
        
        bt = compute_fns[formula_idx](hist, params, view=view_key)
        
        # FILTER YEARS
        start_yr = st.select_slider("History from", list(range(2002, 2026)), value=2018, key='hist_start')
        bt_filtered = bt[bt['Year'] >= start_yr].copy()
        
        if bt_filtered.empty:
            st.warning("No data for the selected period.")
        else:
            st.markdown("---")
            
            # CHARTS
            st.plotly_chart(create_formula_vs_market_chart(bt_filtered, formula_name, floor=floor_val, cap=cap_val), use_container_width=True)
            
            st.plotly_chart(create_pnl_chart(bt_filtered, formula_name, view_mode=view_key), use_container_width=True)
            
            # SUMMARY METRICS
            st.markdown("### Performance Summary")
            
            avg_pnl = bt_filtered['PnL'].mean()
            total_pnl = bt_filtered['PnL'].sum()
            pct_positive = (bt_filtered['PnL'] >= 0).mean() * 100
            max_loss = bt_filtered['PnL'].min()
            max_gain = bt_filtered['PnL'].max()
            period_label = "year" if view_key == 'annual' else "quarter"
            
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            
            clr = '#16a34a' if avg_pnl >= 0 else '#dc2626'
            mc1.markdown(f"""
            <div class='formula-card'>
                <h4>Avg P&L</h4>
                <div class='value' style='color: {clr}'>${avg_pnl:+.1f}/t</div>
                <div class='label'>Per {period_label}</div>
            </div>
            """, unsafe_allow_html=True)
            
            mc2.markdown(f"""
            <div class='formula-card'>
                <h4>Win Rate</h4>
                <div class='value' style='color: {"#16a34a" if pct_positive >= 50 else "#dc2626"}'>{pct_positive:.0f}%</div>
                <div class='label'>of {period_label}s formula below market</div>
            </div>
            """, unsafe_allow_html=True)
            
            mc3.markdown(f"""
            <div class='formula-card'>
                <h4>Best {period_label.title()}</h4>
                <div class='value' style='color: #16a34a'>${max_gain:+.1f}/t</div>
                <div class='label'>Maximum advantage</div>
            </div>
            """, unsafe_allow_html=True)
            
            mc4.markdown(f"""
            <div class='formula-card'>
                <h4>Worst {period_label.title()}</h4>
                <div class='value' style='color: #dc2626'>${max_loss:+.1f}/t</div>
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
            
            with st.expander("Detailed Data"):
                st.dataframe(bt_filtered[['Period', 'Market', 'Formula', 'PnL']].round(1),
                             use_container_width=True, hide_index=True)
        
        # ============================================================
        # DECISION SECTION — Apply Selected Formula to Monte Carlo
        # ============================================================
        st.markdown("---")
        st.markdown("### Decision — Test Formula on Future Scenarios")
        st.markdown(f"*Using **{formula_name}** ({selected}) with your parameters above*")
        
        results = st.session_state.get('results')
        
        if results is None:
            st.info("Run a Monte Carlo simulation in the **first tab** to generate future market scenarios, then come back here to test your formula.")
        else:
            vol_c1, vol_c2 = st.columns([1, 3])
            annual_vol = vol_c1.number_input("Annual Volume (Kt)", 100, 5000, 750, 50, key='vol_kt')
            
            n_sc = results['all_paths'].shape[0]
            sc_idx = st.slider("Select Monte Carlo Scenario", 1, n_sc, 1, key='d_sc')
            scenario_prices = results['all_paths'][sc_idx - 1, :]
            dates = results['dates']
            
            # Build scenario from generated data + MC price variation
            if _corr_data_loaded and _generated_yearly is not None:
                gen_monthly = expand_yearly_to_monthly(_generated_yearly)
                n_months = len(scenario_prices)
                # Build a lookup from generated monthly data indexed by (year, month)
                gen_monthly['_ym'] = gen_monthly['Year'].astype(int) * 100 + gen_monthly['Month'].astype(int)
                gen_lookup = gen_monthly.set_index('_ym')
                # Create scenario_df aligned to MC dates
                mc_years = dates.year
                mc_months = dates.month
                mc_ym = mc_years * 100 + mc_months
                # Merge by year-month; missing months get NaN then ffill/bfill
                scenario_df = gen_lookup.reindex(mc_ym).reset_index(drop=True)
                # Fill months outside generated range (2025 or beyond 2035)
                scenario_df = scenario_df.ffill().bfill()
                # Overlay MC ACS price variation
                scenario_df['ACS_CFR_NAfrica'] = scenario_prices
                # Apply correlated adjustments based on ACS deviation
                base_acs_lookup = gen_lookup['ACS_CFR_NAfrica'].reindex(mc_ym).values
                base_acs_lookup = pd.Series(base_acs_lookup).ffill().bfill().values
                for idx_row in range(n_months):
                    if base_acs_lookup[idx_row] > 0:
                        acs_change = (scenario_prices[idx_row] - base_acs_lookup[idx_row]) / base_acs_lookup[idx_row]
                        for var in ['S_CFR_ME', 'S_CFR_NA', 'DAP']:
                            if var in scenario_df.columns and var in _corr_matrix.columns:
                                corr_val = _corr_matrix.loc['ACS_CFR_NAfrica', var] if 'ACS_CFR_NAfrica' in _corr_matrix.index else 0.7
                                noise = np.random.uniform(-0.10, 0.10)
                                scenario_df.loc[scenario_df.index[idx_row], var] *= (1 + corr_val * acs_change + noise * abs(acs_change))
                scenario_df['Year'] = mc_years
                scenario_df['Quarter'] = (mc_months - 1) // 3 + 1
                scenario_df['Month'] = mc_months
                # IPP proxies from ACS
                scenario_df['IPP_Europe'] = scenario_prices * 0.95
                scenario_df['IPP_Japan'] = scenario_prices * 0.90
                scenario_df['IPP_China'] = scenario_prices * 0.85
            else:
                # Fallback to synthetic proxies
                scenario_df = pd.DataFrame({
                    'ACS_CFR_NAfrica': scenario_prices,
                    'Year': dates.year,
                    'Quarter': (dates.month - 1) // 3 + 1,
                    'Month': dates.month,
                    'S_CFR_ME': scenario_prices * 0.55,
                    'S_CFR_NA': scenario_prices * 0.50,
                    'IPP_Europe': scenario_prices * 0.95,
                    'IPP_Japan': scenario_prices * 0.90,
                    'IPP_China': scenario_prices * 0.85,
                })
            if 'DAP' not in scenario_df.columns:
                scenario_df['DAP'] = params.get('dap0', 500)
            if pc_clk_editable and petcoke_outlook:
                scenario_df['Petcoke'] = scenario_df['Year'].map(petcoke_outlook).fillna(params.get('pc0', 140))
            elif 'Petcoke' not in scenario_df.columns:
                scenario_df['Petcoke'] = params.get('pc0', 140)
            if pc_clk_editable and clinker_outlook:
                scenario_df['Clinker'] = scenario_df['Year'].map(clinker_outlook).fillna(params.get('clk0', 130))
            elif 'Clinker' not in scenario_df.columns:
                scenario_df['Clinker'] = params.get('clk0', 130)
            
            # Compute formula for this scenario
            formula_prices = compute_fns[formula_idx](scenario_df, params, view='quarterly')
            
            monthly_vol = annual_vol * 1000 / 12
            n_months = len(scenario_prices)
            
            # --- Chart: MC scenario vs formula price ---
            if not formula_prices.empty:
                fig_sc = go.Figure()
                # Both use the same Period x-axis from the formula computation
                fig_sc.add_trace(go.Scatter(
                    x=formula_prices['Period'], y=formula_prices['Market'], mode='lines+markers',
                    line=dict(color='#dc2626', width=2.5), marker=dict(size=5),
                    name='MC Market (quarterly avg)',
                    hovertemplate='<b>Market</b><br>%{x}<br>$%{y:.1f}/t<extra></extra>'))
                fig_sc.add_trace(go.Scatter(
                    x=formula_prices['Period'], y=formula_prices['Formula'], mode='lines+markers',
                    line=dict(color='#2563eb', width=2.5), marker=dict(size=5),
                    name=f'{formula_name} Price',
                    hovertemplate='<b>Formula</b><br>%{x}<br>$%{y:.1f}/t<extra></extra>'))
                
                fig_sc.add_hline(y=floor_val, line=dict(color='#f59e0b', width=1.5, dash='dash'),
                                annotation_text=f'Floor: ${floor_val}')
                fig_sc.add_hline(y=cap_val, line=dict(color='#f59e0b', width=1.5, dash='dash'),
                                annotation_text=f'Cap: ${cap_val}')
                chart_layout(fig_sc, f'<b>{formula_name} Applied to MC Scenario #{sc_idx}</b>')
                st.plotly_chart(fig_sc, use_container_width=True)
                
                # --- P&L on this scenario ---
                st.plotly_chart(create_pnl_chart(formula_prices, formula_name, view_mode='quarterly'),
                               use_container_width=True)
            else:
                st.warning("Could not compute formula on this scenario.")
            
            # --- Revenue metrics ---
            if not formula_prices.empty:
                avg_market_q = formula_prices['Market'].mean()
                avg_formula_q = formula_prices['Formula'].mean()
                avg_pnl_q = formula_prices['PnL'].mean()
                total_market = avg_market_q * annual_vol * 1000 / 1e6
                total_formula = avg_formula_q * annual_vol * 1000 / 1e6
                savings = total_market - total_formula
                
                dc1, dc2, dc3 = st.columns(3)
                dc1.markdown(f"""
                <div class='formula-card'>
                    <h4>Market Cost</h4>
                    <div class='value'>${total_market:.1f}M/yr</div>
                    <div class='label'>Avg market price: ${avg_market_q:.0f}/t</div>
                </div>
                """, unsafe_allow_html=True)
                dc2.markdown(f"""
                <div class='formula-card'>
                    <h4>Formula Cost</h4>
                    <div class='value'>${total_formula:.1f}M/yr</div>
                    <div class='label'>Avg formula price: ${avg_formula_q:.0f}/t</div>
                </div>
                """, unsafe_allow_html=True)
                clr = '#16a34a' if savings >= 0 else '#dc2626'
                dc3.markdown(f"""
                <div class='formula-card'>
                    <h4>{'Savings' if savings >= 0 else 'Extra Cost'}</h4>
                    <div class='value' style='color: {clr}'>${savings:+.1f}M/yr</div>
                    <div class='label'>at {annual_vol} Kt/yr</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- Multi-scenario risk analysis ---
            st.markdown("### Risk Analysis — All MC Scenarios")
            all_savings = []
            sample_size = min(n_sc, 200)  # Cap for speed
            # Pre-compute generated monthly data outside the loop
            if _corr_data_loaded and _generated_yearly is not None:
                _gen_m_risk = expand_yearly_to_monthly(_generated_yearly)
                _gen_m_risk['_ym'] = _gen_m_risk['Year'].astype(int) * 100 + _gen_m_risk['Month'].astype(int)
                _gen_lookup_risk = _gen_m_risk.set_index('_ym')
                _mc_ym_risk = dates.year * 100 + dates.month
            else:
                _gen_lookup_risk = None
            for i in range(sample_size):
                sc_prices = results['all_paths'][i, :]
                n_months = len(sc_prices)
                if _gen_lookup_risk is not None:
                    sc_df = _gen_lookup_risk.reindex(_mc_ym_risk).reset_index(drop=True)
                    sc_df = sc_df.ffill().bfill()
                    sc_df['ACS_CFR_NAfrica'] = sc_prices
                    sc_df['Year'] = dates.year
                    sc_df['Quarter'] = (dates.month - 1) // 3 + 1
                    sc_df['Month'] = dates.month
                    sc_df['IPP_Europe'] = sc_prices * 0.95
                    sc_df['IPP_Japan'] = sc_prices * 0.90
                    sc_df['IPP_China'] = sc_prices * 0.85
                else:
                    sc_df = pd.DataFrame({
                        'ACS_CFR_NAfrica': sc_prices,
                        'Year': dates.year,
                        'Quarter': (dates.month - 1) // 3 + 1,
                        'Month': dates.month,
                        'S_CFR_ME': sc_prices * 0.55,
                        'S_CFR_NA': sc_prices * 0.50,
                        'IPP_Europe': sc_prices * 0.95,
                        'IPP_Japan': sc_prices * 0.90,
                        'IPP_China': sc_prices * 0.85,
                    })
                if 'DAP' not in sc_df.columns:
                    sc_df['DAP'] = params.get('dap0', 500)
                if pc_clk_editable and petcoke_outlook:
                    sc_df['Petcoke'] = sc_df['Year'].map(petcoke_outlook).fillna(params.get('pc0', 140))
                elif 'Petcoke' not in sc_df.columns:
                    sc_df['Petcoke'] = params.get('pc0', 140)
                if pc_clk_editable and clinker_outlook:
                    sc_df['Clinker'] = sc_df['Year'].map(clinker_outlook).fillna(params.get('clk0', 130))
                elif 'Clinker' not in sc_df.columns:
                    sc_df['Clinker'] = params.get('clk0', 130)
                
                sc_result = compute_fns[formula_idx](sc_df, params, view='quarterly')
                if not sc_result.empty:
                    sc_avg_pnl = sc_result['PnL'].mean()
                    sc_savings = sc_avg_pnl * annual_vol * 1000 / 1e6
                    all_savings.append(sc_savings)
            
            all_savings = np.array(all_savings)
            if len(all_savings) > 0:
                avg_save = np.mean(all_savings)
                p5 = np.percentile(all_savings, 5)
                p95 = np.percentile(all_savings, 95)
                prob_save = np.mean(all_savings >= 0) * 100
                
                risk_cols = st.columns(4)
                clr_avg = '#16a34a' if avg_save >= 0 else '#dc2626'
                risk_cols[0].markdown(f"""
                <div class='formula-card'>
                    <h4>Expected Savings</h4>
                    <div class='value' style='color: {clr_avg}'>${avg_save:+.1f}M/yr</div>
                    <div class='label'>Mean across {sample_size} scenarios</div>
                </div>
                """, unsafe_allow_html=True)
                risk_cols[1].markdown(f"""
                <div class='formula-card'>
                    <h4>Best Case (P95)</h4>
                    <div class='value' style='color: #16a34a'>${p95:+.1f}M/yr</div>
                    <div class='label'>Favorable scenario</div>
                </div>
                """, unsafe_allow_html=True)
                risk_cols[2].markdown(f"""
                <div class='formula-card'>
                    <h4>Worst Case (P5)</h4>
                    <div class='value' style='color: {"#16a34a" if p5 >= 0 else "#dc2626"}'>${p5:+.1f}M/yr</div>
                    <div class='label'>Adverse scenario</div>
                </div>
                """, unsafe_allow_html=True)
                risk_cols[3].markdown(f"""
                <div class='formula-card'>
                    <h4>Prob. of Savings</h4>
                    <div class='value' style='color: {"#16a34a" if prob_save >= 50 else "#dc2626"}'>{prob_save:.0f}%</div>
                    <div class='label'>of scenarios show savings</div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=all_savings, nbinsx=50,
                    marker=dict(color='#2563eb', line=dict(color='white', width=0.5)), opacity=0.8))
                fig_dist.add_vline(x=0, line=dict(color='#94a3b8', width=2))
                fig_dist.add_vline(x=avg_save, line=dict(color='#f59e0b', width=2, dash='dash'),
                                  annotation_text=f'Mean: ${avg_save:+.1f}M')
                chart_layout(fig_dist, f'<b>Savings Distribution — {formula_name} on {sample_size} MC Scenarios</b>',
                            height=350, yaxis_title='Frequency')
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #5a6a85; font-size: 0.85rem;'>
        Built by Mohammed ELABRIDI | Data: CRU Outlook & ACS Pricing Simulator
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
