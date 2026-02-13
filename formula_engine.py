"""
ACS Pricing Formula Engine
Loads historical data & computes 6 pricing formulas from the Excel simulator.
Supports both old (0902_1700) and new (1202_1203) Excel files.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_historical_prices(xlsx_path: str) -> pd.DataFrame:
    """Load monthly historical prices from 'Prices S_ACS' or 'Historical prices S_ACS' sheet."""
    # Try both sheet names (new file vs old file)
    try:
        df = pd.read_excel(xlsx_path, sheet_name='Prices S_ACS', header=None)
    except:
        df = pd.read_excel(xlsx_path, sheet_name='Historical prices S_ACS', header=None)
    
    # Columns are always in same positions:
    # B=Year, C=Quarter, D=Month, E=ACS FOB Japan, F=ACS FOB NWE, G=ACS FOB China,
    # H=ACS CFR NAfrica, I=S FOB ME, J=S CFR NAfrica
    # Freight: M=Japan, N=NWE, O=China, P=ME, Q=CRU
    headers = {
        'Year': 'B', 'Quarter': 'C', 'Month': 'D',
        'ACS_FOB_Japan': 'E', 'ACS_FOB_NWE': 'F', 'ACS_FOB_China': 'G',
        'ACS_CFR_NAfrica': 'H', 'S_FOB_ME': 'I', 'S_CFR_NAfrica': 'J',
        'Fret_Japan': 'M', 'Fret_NWE': 'N', 'Fret_China': 'O',
        'Fret_ME': 'P', 'Fret_CRU': 'Q'
    }
    
    col_map = {ord(letter) - ord('A'): name for name, letter in headers.items()}
    
    result = pd.DataFrame()
    for col_idx, col_name in col_map.items():
        if col_idx < df.shape[1]:
            result[col_name] = pd.to_numeric(df.iloc[5:, col_idx], errors='coerce')
    
    result = result.dropna(subset=['Year', 'Quarter']).reset_index(drop=True)
    result['Year'] = result['Year'].astype(int)
    result['Quarter'] = result['Quarter'].astype(int)
    
    # Derived: CFR prices (FOB + Freight)
    result['IPP_Europe'] = result['ACS_FOB_NWE'] + result['Fret_NWE']
    result['IPP_Japan'] = result['ACS_FOB_Japan'] + result['Fret_Japan']
    result['IPP_China'] = result['ACS_FOB_China'] + result['Fret_China']
    result['S_CFR_ME'] = result['S_FOB_ME'] + result['Fret_ME']
    result['S_CFR_NA'] = result['S_CFR_NAfrica'] + result['Fret_CRU']
    
    return result


def load_historical_prices_extended(xlsx_path: str) -> pd.DataFrame:
    """Load from the new-format simulator sheet that has Petcoke & Clinker columns."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name='ACS Simulator with variation', header=None)
    except:
        return None
    
    # Row 19 (0-indexed: 18) is header in new file
    # Cols: 1=Date, 2=Year, 3=Quarter, 4=Month, 5=ACS CFR, 6=ACS NWE, 7=ACS Japan,
    #       8=ACS China, 9=S ME, 10=S NA, 11=Smooth S ME, 12=Smooth S NA,
    #       13=DAP, 14=Smooth DAP, 15=MAP, 16=TSP, 17=Petcoke, 18=Clinker
    header_row = None
    for i in range(20):
        if df.iloc[i, 0] is not None and 'Date' in str(df.iloc[i, 0]):
            header_row = i
            break
    
    if header_row is None:
        return None
    
    data = df.iloc[header_row + 1:].copy()
    result = pd.DataFrame()
    result['Year'] = pd.to_numeric(data.iloc[:, 1], errors='coerce')
    result['Quarter'] = pd.to_numeric(data.iloc[:, 2], errors='coerce')
    result['Month'] = pd.to_numeric(data.iloc[:, 3], errors='coerce')
    result['ACS_CFR_NAfrica'] = pd.to_numeric(data.iloc[:, 4], errors='coerce')
    result['ACS_NWE'] = pd.to_numeric(data.iloc[:, 5], errors='coerce')
    result['ACS_Japan'] = pd.to_numeric(data.iloc[:, 6], errors='coerce')
    result['ACS_China'] = pd.to_numeric(data.iloc[:, 7], errors='coerce')
    result['S_CFR_ME'] = pd.to_numeric(data.iloc[:, 8], errors='coerce')
    result['S_CFR_NA'] = pd.to_numeric(data.iloc[:, 9], errors='coerce')
    result['DAP'] = pd.to_numeric(data.iloc[:, 12], errors='coerce')
    result['MAP'] = pd.to_numeric(data.iloc[:, 14], errors='coerce')
    result['TSP'] = pd.to_numeric(data.iloc[:, 15], errors='coerce')
    
    # Petcoke and Clinker (only in new file)
    if data.shape[1] > 17:
        result['Petcoke'] = pd.to_numeric(data.iloc[:, 16], errors='coerce')
        result['Clinker'] = pd.to_numeric(data.iloc[:, 17], errors='coerce')
    
    result = result.dropna(subset=['Year', 'Quarter']).reset_index(drop=True)
    result['Year'] = result['Year'].astype(int)
    result['Quarter'] = result['Quarter'].astype(int)
    
    # Add IPP columns for compatibility
    result['IPP_Europe'] = result['ACS_NWE']
    result['IPP_Japan'] = result['ACS_Japan']
    result['IPP_China'] = result['ACS_China']
    
    return result


def load_phosphate_prices(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='Phosphate historical prices', header=None)
    
    headers = {
        'Year': 1,
        'DAP_NOLA': 2, 'DAP_Morocco': 3, 'DAP_Jordan': 4,
        'DAP_Saudi': 5, 'DAP_India': 6, 'DAP_China': 7,
        'MAP_Baltic': 8, 'MAP_Brazil': 9,
        'TSP_Morocco': 10, 'TSP_Brazil': 11
    }
    
    result = pd.DataFrame()
    for col_name, col_idx in headers.items():
        result[col_name] = pd.to_numeric(df.iloc[5:, col_idx], errors='coerce')
    
    result = result.dropna(subset=['Year']).reset_index(drop=True)
    result['Year'] = result['Year'].astype(int)
    
    dap_cols = [c for c in result.columns if c.startswith('DAP_')]
    result['DAP_Avg'] = result[dap_cols].mean(axis=1, skipna=True)
    
    return result


def load_forecast_data(xlsx_path: str) -> pd.DataFrame:
    """Load annual forecast data from the 'Input data' sheet (new file only)."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name='Input data', header=None)
    except:
        return None
    
    # Find the header row with 'Year'
    header_row = None
    for i in range(15):
        if df.iloc[i, 0] is not None and 'Year' in str(df.iloc[i, 0]):
            header_row = i
            break
    
    if header_row is None:
        # Fallback: row 12 (0-indexed) based on our analysis
        header_row = 12
    
    data = df.iloc[header_row + 1:].copy()
    result = pd.DataFrame()
    result['Year'] = pd.to_numeric(data.iloc[:, 0], errors='coerce')
    result['ACS_CFR_NAfrica'] = pd.to_numeric(data.iloc[:, 1], errors='coerce')
    result['ACS_NWE'] = pd.to_numeric(data.iloc[:, 2], errors='coerce')
    result['ACS_Japan'] = pd.to_numeric(data.iloc[:, 3], errors='coerce')
    result['ACS_China'] = pd.to_numeric(data.iloc[:, 4], errors='coerce')
    result['S_CFR_ME'] = pd.to_numeric(data.iloc[:, 5], errors='coerce')
    result['S_CFR_NA'] = pd.to_numeric(data.iloc[:, 6], errors='coerce')
    result['DAP'] = pd.to_numeric(data.iloc[:, 7], errors='coerce')
    
    if data.shape[1] > 8:
        result['Petcoke'] = pd.to_numeric(data.iloc[:, 8], errors='coerce')
    if data.shape[1] > 9:
        result['Clinker'] = pd.to_numeric(data.iloc[:, 9], errors='coerce')
    
    result = result.dropna(subset=['Year']).reset_index(drop=True)
    result['Year'] = result['Year'].astype(int)
    
    return result


def load_base_data(xlsx_path: str) -> dict:
    try:
        df = pd.read_excel(xlsx_path, sheet_name='Base Data', header=None)
    except:
        # New file doesn't have Base Data — use Input data instead
        return {
            'w_japan': 0, 'w_europe': 0.75, 'w_china': 0.20, 'w_nafrica': 0.05,
            'w_s_me': 0.7, 'w_s_na': 0.3,
            'w_dap': 1, 'w_map': 0, 'w_tsp': 0,
            'yield_dap': 0.39, 'yield_map': 0.43, 'yield_tsp': 0.55,
            'alpha': 0.7, 'beta': 0.3, 'gamma': 0,
            'conversion_cost': 20, 'acs_from_sulphur': 3.02,
            'fixed_floor': 110, 'fixed_cap': 220,
            'annual_volume_kt': 750, 'formula_pct': 0.75, 'spot_pct': 0.25,
        }
    
    def cell(row, col):
        val = df.iloc[row - 1, col - 1]
        try:
            return float(val)
        except (ValueError, TypeError):
            return val
    
    return {
        'w_japan': cell(12, 3), 'w_europe': cell(13, 3),
        'w_china': cell(14, 3), 'w_nafrica': cell(15, 3),
        'w_s_me': cell(18, 3), 'w_s_na': cell(19, 3),
        'w_dap': cell(22, 3), 'w_map': cell(23, 3), 'w_tsp': cell(24, 3),
        'yield_dap': cell(32, 3), 'yield_map': cell(33, 3), 'yield_tsp': cell(34, 3),
        'alpha': cell(37, 3), 'beta': cell(38, 3), 'gamma': cell(39, 3),
        'conversion_cost': cell(41, 3),
        'acs_from_sulphur': cell(31, 3),
        'fixed_floor': cell(5, 7) if not np.isnan(cell(5, 7)) else 0,
        'fixed_cap': cell(11, 7) if not np.isnan(cell(11, 7)) else 300,
        'annual_volume_kt': cell(9, 3),
        'formula_pct': cell(27, 3), 'spot_pct': cell(28, 3),
    }


# --- AGGREGATION HELPERS ---

def _quarterly_series(hist: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Group historical monthly data into quarterly averages."""
    groups = hist.groupby(['Year', 'Quarter'])
    result = groups[columns].mean().reset_index()
    result['Period'] = result['Year'].astype(str) + ' Q' + result['Quarter'].astype(str)
    return result


def _annual_series(hist: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Group historical monthly data into annual averages."""
    groups = hist.groupby(['Year'])
    result = groups[columns].mean().reset_index()
    result['Period'] = result['Year'].astype(str)
    return result


def _smooth_series(series: pd.Series, window: int = 6, sigma: float = 2.0) -> pd.Series:
    """Gaussian-weighted rolling smooth for a genuinely rounded curve."""
    return series.rolling(window=window, min_periods=1, center=True, win_type='gaussian').mean(std=sigma)


def _smooth_3m(series: pd.Series) -> pd.Series:
    """Excel-style 3-month rolling average (match Excel exactly)."""
    return series.rolling(window=3, min_periods=1).mean()


def _apply_floor_cap(series: pd.Series, floor: float, cap: float) -> pd.Series:
    """Apply per-formula floor and cap."""
    return series.clip(lower=floor, upper=cap)


# --- INDIVIDUAL FORMULA COMPUTATIONS ---

def compute_formula_1(hist: pd.DataFrame, params: dict, view: str = 'quarterly') -> pd.DataFrame:
    """F1: Sulfur Indexing Only — P = a × (S_weighted × conv_ratio + prod_cost) + b"""
    a = params.get('a', 1.3)
    b = params.get('b', 25)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    conv_ratio = params.get('conv_ratio', 0.33)
    prod_cost = params.get('prod_cost', 20)
    floor = params.get('floor', 110)
    cap = params.get('cap', 220)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    df['Formula_raw'] = a * (df['S_weighted'] * conv_ratio + prod_cost) + b
    df['Formula'] = df['Formula_raw'].clip(lower=floor, upper=cap)
    
    agg_fn = _annual_series if view == 'annual' else _quarterly_series
    q = agg_fn(df, ['ACS_CFR_NAfrica', 'S_weighted', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_2(hist: pd.DataFrame, params: dict, view: str = 'quarterly') -> pd.DataFrame:
    """F2: Smooth Sulfur Indexing — P = a × (Smooth(S) × conv_ratio + prod_cost) + b"""
    a = params.get('a', 1.4)
    b = params.get('b', 15)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    conv_ratio = params.get('conv_ratio', 0.33)
    prod_cost = params.get('prod_cost', 20)
    smooth_window = params.get('smooth_window', 6)
    floor = params.get('floor', 110)
    cap = params.get('cap', 220)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    df['S_smooth'] = _smooth_series(df['S_weighted'], window=smooth_window, sigma=smooth_window / 3)
    df['Formula_raw'] = a * (df['S_smooth'] * conv_ratio + prod_cost) + b
    df['Formula'] = df['Formula_raw'].clip(lower=floor, upper=cap)
    
    agg_fn = _annual_series if view == 'annual' else _quarterly_series
    q = agg_fn(df, ['ACS_CFR_NAfrica', 'S_weighted', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_3(hist: pd.DataFrame, params: dict, view: str = 'quarterly') -> pd.DataFrame:
    """F3: Last Month ACS Indexing — P = a × ACS_weighted(m-1)"""
    a = params.get('a', 1.0)
    w_na = params.get('acs_na', 0.05)
    w_eu = params.get('acs_eu', 0.75)
    w_jp = params.get('acs_jp', 0.0)
    w_ch = params.get('acs_ch', 0.20)
    floor = params.get('floor', 110)
    cap = params.get('cap', 220)
    
    df = hist.copy()
    df['ACS_weighted'] = (
        w_na * df['ACS_CFR_NAfrica'].fillna(0) +
        w_eu * df['IPP_Europe'].fillna(0) +
        w_jp * df['IPP_Japan'].fillna(0) +
        w_ch * df['IPP_China'].fillna(0)
    )
    df['ACS_lagged'] = df['ACS_weighted'].shift(1)
    df['Formula_raw'] = a * df['ACS_lagged']
    df['Formula'] = df['Formula_raw'].clip(lower=floor, upper=cap)
    
    agg_fn = _annual_series if view == 'annual' else _quarterly_series
    q = agg_fn(df, ['ACS_CFR_NAfrica', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_4(hist: pd.DataFrame, params: dict, view: str = 'quarterly') -> pd.DataFrame:
    """F4: S & DAP Variation — P = ACS₀ × (a + b × S/S₀ + c × DAP/DAP₀)
    
    Uses monthly DAP from historical data if available, otherwise falls back to annual phosphate data.
    """
    a = params.get('a', 0.7)
    b = params.get('b', 0.1)
    c = params.get('c', 0.2)
    acs0 = params.get('acs0', 110)
    s0 = params.get('s0', 130)
    dap0 = params.get('dap0', 500)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    floor = params.get('floor', 110)
    cap = params.get('cap', 230)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    
    # Use monthly DAP if available (new file has it), otherwise use dap0
    if 'DAP' in df.columns:
        df['DAP_val'] = df['DAP'].fillna(dap0)
    else:
        phos = params.get('_phos')
        if phos is not None and 'DAP_Avg' in phos.columns:
            dap_map = phos.set_index('Year')['DAP_Avg'].to_dict()
            df['DAP_val'] = df['Year'].map(dap_map).fillna(dap0)
        else:
            df['DAP_val'] = dap0
    
    df['Formula_raw'] = acs0 * (a + b * df['S_weighted'] / s0 + c * df['DAP_val'] / dap0)
    df['Formula'] = df['Formula_raw'].clip(lower=floor, upper=cap)
    
    agg_fn = _annual_series if view == 'annual' else _quarterly_series
    q = agg_fn(df, ['ACS_CFR_NAfrica', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_5(hist: pd.DataFrame, params: dict, view: str = 'quarterly') -> pd.DataFrame:
    """F5: Smooth S & Smooth DAP — P = ACS₀ × (a + b × Sm(S)/S₀ + c × Sm(DAP)/DAP₀)"""
    a = params.get('a', 0.8)
    b = params.get('b', 0.1)
    c = params.get('c', 0.1)
    acs0 = params.get('acs0', 110)
    s0 = params.get('s0', 130)
    dap0 = params.get('dap0', 500)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    smooth_window = params.get('smooth_window', 6)
    floor = params.get('floor', 110)
    cap = params.get('cap', 230)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    df['S_smooth'] = _smooth_series(df['S_weighted'], window=smooth_window, sigma=smooth_window / 3)
    
    if 'DAP' in df.columns:
        df['DAP_smooth'] = _smooth_series(df['DAP'].fillna(dap0), window=smooth_window, sigma=smooth_window / 3)
    else:
        phos = params.get('_phos')
        if phos is not None and 'DAP_Avg' in phos.columns:
            dap_map = phos.set_index('Year')['DAP_Avg'].to_dict()
            df['DAP_val'] = df['Year'].map(dap_map).fillna(dap0)
            df['DAP_smooth'] = _smooth_series(df['DAP_val'], window=smooth_window, sigma=smooth_window / 3)
        else:
            df['DAP_smooth'] = dap0
    
    df['Formula_raw'] = acs0 * (a + b * df['S_smooth'] / s0 + c * df['DAP_smooth'] / dap0)
    df['Formula'] = df['Formula_raw'].clip(lower=floor, upper=cap)
    
    agg_fn = _annual_series if view == 'annual' else _quarterly_series
    q = agg_fn(df, ['ACS_CFR_NAfrica', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_6(hist: pd.DataFrame, params: dict, view: str = 'quarterly') -> pd.DataFrame:
    """F6: S, DAP, Petcoke & Clinker — P = ACS₀ × (a + b·S/S₀ + c·DAP/DAP₀ + d·PC/PC₀ + e·(1−CLK/CLK₀))"""
    a = params.get('a', 0.6)
    b = params.get('b', 0.05)
    c = params.get('c', 0.2)
    d = params.get('d', 0.05)
    e = params.get('e', 0.1)
    acs0 = params.get('acs0', 110)
    s0 = params.get('s0', 130)
    dap0 = params.get('dap0', 500)
    pc0 = params.get('pc0', 140)
    clk0 = params.get('clk0', 130)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    floor = params.get('floor', 110)
    cap = params.get('cap', 230)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    
    # DAP
    if 'DAP' in df.columns:
        df['DAP_val'] = df['DAP'].fillna(dap0)
    else:
        df['DAP_val'] = dap0
    
    # Petcoke
    if 'Petcoke' in df.columns:
        df['PC_val'] = df['Petcoke'].fillna(pc0)
    else:
        df['PC_val'] = pc0
    
    # Clinker
    if 'Clinker' in df.columns:
        df['CLK_val'] = df['Clinker'].fillna(clk0)
    else:
        df['CLK_val'] = clk0
    
    df['Formula_raw'] = acs0 * (
        a +
        b * df['S_weighted'] / s0 +
        c * df['DAP_val'] / dap0 +
        d * df['PC_val'] / pc0 +
        e * (1 - df['CLK_val'] / clk0)
    )
    df['Formula'] = df['Formula_raw'].clip(lower=floor, upper=cap)
    
    agg_fn = _annual_series if view == 'annual' else _quarterly_series
    q = agg_fn(df, ['ACS_CFR_NAfrica', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


# --- FORECAST FORMULA COMPUTATIONS (annual data) ---

def compute_forecast(forecast_df: pd.DataFrame, formula_idx: int, params: dict) -> pd.DataFrame:
    """Compute formula on annual forecast data. Returns DataFrame with Year, Market, Formula, PnL."""
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame()
    
    df = forecast_df.copy()
    
    # Market reference
    market = df['ACS_CFR_NAfrica'].values
    
    # Sulfur weighted
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    s_weighted = w_me * df['S_CFR_ME'].fillna(0) + w_na * df['S_CFR_NA'].fillna(0)
    
    floor = params.get('floor', 110)
    cap = params.get('cap', 220)
    
    if formula_idx == 0:  # F1
        a, b = params.get('a', 1.3), params.get('b', 25)
        conv = params.get('conv_ratio', 0.33)
        prod = params.get('prod_cost', 20)
        formula = a * (s_weighted * conv + prod) + b
    
    elif formula_idx == 1:  # F2 — no smooth on annual data, use same as F1
        a, b = params.get('a', 1.4), params.get('b', 15)
        conv = params.get('conv_ratio', 0.33)
        prod = params.get('prod_cost', 20)
        formula = a * (s_weighted * conv + prod) + b
    
    elif formula_idx == 2:  # F3 — use last year ACS
        a = params.get('a', 1.0)
        w_na_a = params.get('acs_na', 0.05)
        w_eu = params.get('acs_eu', 0.75)
        w_jp = params.get('acs_jp', 0.0)
        w_ch = params.get('acs_ch', 0.20)
        acs_weighted = (
            w_na_a * df['ACS_CFR_NAfrica'].fillna(0) +
            w_eu * df.get('ACS_NWE', df['ACS_CFR_NAfrica']).fillna(0) +
            w_jp * df.get('ACS_Japan', pd.Series(0, index=df.index)).fillna(0) +
            w_ch * df.get('ACS_China', pd.Series(0, index=df.index)).fillna(0)
        )
        formula = a * acs_weighted.shift(1)
    
    elif formula_idx == 3:  # F4
        a = params.get('a', 0.7)
        b = params.get('b', 0.1)
        c = params.get('c', 0.2)
        acs0 = params.get('acs0', 110)
        s0 = params.get('s0', 130)
        dap0 = params.get('dap0', 500)
        dap = df['DAP'].fillna(dap0) if 'DAP' in df.columns else dap0
        formula = acs0 * (a + b * s_weighted / s0 + c * dap / dap0)
    
    elif formula_idx == 4:  # F5 — no smooth on annual
        a = params.get('a', 0.8)
        b = params.get('b', 0.1)
        c = params.get('c', 0.1)
        acs0 = params.get('acs0', 110)
        s0 = params.get('s0', 130)
        dap0 = params.get('dap0', 500)
        dap = df['DAP'].fillna(dap0) if 'DAP' in df.columns else dap0
        formula = acs0 * (a + b * s_weighted / s0 + c * dap / dap0)
    
    elif formula_idx == 5:  # F6
        a = params.get('a', 0.6)
        b = params.get('b', 0.05)
        c = params.get('c', 0.2)
        d_coef = params.get('d', 0.05)
        e_coef = params.get('e', 0.1)
        acs0 = params.get('acs0', 110)
        s0 = params.get('s0', 130)
        dap0 = params.get('dap0', 500)
        pc0 = params.get('pc0', 140)
        clk0 = params.get('clk0', 130)
        dap = df['DAP'].fillna(dap0) if 'DAP' in df.columns else dap0
        pc = df['Petcoke'].fillna(pc0) if 'Petcoke' in df.columns else pc0
        clk = df['Clinker'].fillna(clk0) if 'Clinker' in df.columns else clk0
        formula = acs0 * (a + b * s_weighted / s0 + c * dap / dap0 + d_coef * pc / pc0 + e_coef * (1 - clk / clk0))
    else:
        return pd.DataFrame()
    
    result = pd.DataFrame({
        'Year': df['Year'],
        'Period': df['Year'].astype(str),
        'Market': market,
        'Formula': np.clip(formula, floor, cap),
    })
    result['PnL'] = result['Market'] - result['Formula']
    return result.dropna(subset=['Market', 'Formula'])


def apply_floor_cap(price: float, floor: float, cap: float) -> float:
    if np.isnan(price):
        return np.nan
    return max(floor, min(cap, price))


def apply_formula_to_scenario(scenario_prices, dates, weights, floor=0, cap=300):
    """Apply pricing formula to a Monte Carlo scenario path."""
    alpha = weights.get('alpha', 0.7)
    beta = weights.get('beta', 0.3)
    yield_ratio = weights.get('acs_from_sulphur', 3.02)
    conv_cost = weights.get('conversion_cost', 20)
    
    s_acid_prices = scenario_prices / yield_ratio + conv_cost
    blended = alpha * scenario_prices + beta * s_acid_prices
    capped = np.clip(blended, floor, cap)
    
    return {
        'dates': dates,
        'acs_index': scenario_prices,
        's_acid': s_acid_prices,
        'blended': blended,
        'capped': capped,
        'floor': floor,
        'cap': cap
    }


# Legacy backtest (still used for blended comparison)
def backtest_formulas(xlsx_path: str, weights: dict = None) -> pd.DataFrame:
    hist = load_historical_prices(xlsx_path)
    base = load_base_data(xlsx_path)
    if weights:
        base.update(weights)
    
    floor = base.get('fixed_floor', 0)
    cap = base.get('fixed_cap', 300)
    alpha = base.get('alpha', 0.7)
    beta = base.get('beta', 0.3)
    
    quarters = hist[['Year', 'Quarter']].drop_duplicates().sort_values(['Year', 'Quarter'])
    
    results = []
    for _, row in quarters.iterrows():
        yr, qtr = int(row['Year']), int(row['Quarter'])
        mask = (hist['Year'] == yr) & (hist['Quarter'] == qtr)
        vals = hist[mask]
        if vals.empty:
            continue
        
        market = vals['ACS_CFR_NAfrica'].mean()
        
        ipp_e = vals['IPP_Europe'].mean()
        ipp_j = vals['IPP_Japan'].mean()
        ipp_c = vals['IPP_China'].mean()
        acs_na = vals['ACS_CFR_NAfrica'].mean()
        
        ipp = (base['w_japan'] * ipp_j + base['w_europe'] * ipp_e + 
               base['w_china'] * ipp_c + base['w_nafrica'] * acs_na)
        
        s_me_val = vals['S_CFR_ME'].mean()
        s_na_val = vals['S_CFR_NA'].mean()
        s_weighted = base['w_s_me'] * s_me_val + base['w_s_na'] * s_na_val
        s_acid = s_weighted / base.get('acs_from_sulphur', 3.02) + base.get('conversion_cost', 20)
        
        blended = alpha * ipp + beta * s_acid
        
        ipp_c2 = max(floor, min(cap, ipp))
        s_acid_c = max(floor, min(cap, s_acid))
        blended_c = max(floor, min(cap, blended))
        
        results.append({
            'Year': yr, 'Quarter': qtr, 'Period': f'{yr} Q{qtr}',
            'Market_Price': market,
            'IPP_Acid': ipp, 'IPP_Acid_Capped': ipp_c2,
            'S_Acid': s_acid, 'S_Acid_Capped': s_acid_c,
            'Blended': blended, 'Blended_Capped': blended_c,
            'PnL_IPP': market - ipp_c2, 'PnL_SAcid': market - s_acid_c,
            'PnL_Blended': market - blended_c,
        })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Test with new file
    new_path = str(Path(__file__).parent / "1202_1203_ACS Pricing simulator.xlsx")
    old_path = str(Path(__file__).parent / "0902_1700_ACS Pricing simulator_MG (1).xlsx")
    
    print("=" * 60)
    print("Testing with NEW file (1202_1203)")
    print("=" * 60)
    
    hist_ext = load_historical_prices_extended(new_path)
    if hist_ext is not None:
        print(f"Extended data: {hist_ext.shape[0]} rows, Years: {hist_ext['Year'].min()}-{hist_ext['Year'].max()}")
        has_pc = 'Petcoke' in hist_ext.columns
        has_clk = 'Clinker' in hist_ext.columns
        print(f"Petcoke: {'Yes' if has_pc else 'No'}, Clinker: {'Yes' if has_clk else 'No'}")
        
        for i, fn in enumerate([compute_formula_1, compute_formula_2, compute_formula_3]):
            r = fn(hist_ext, {})
            print(f"F{i+1}: {len(r)} quarters, avg PnL: ${r['PnL'].mean():.1f}/t")
        
        r4 = compute_formula_4(hist_ext, {})
        print(f"F4: {len(r4)} quarters, avg PnL: ${r4['PnL'].mean():.1f}/t")
        
        r5 = compute_formula_5(hist_ext, {})
        print(f"F5: {len(r5)} quarters, avg PnL: ${r5['PnL'].mean():.1f}/t")
        
        r6 = compute_formula_6(hist_ext, {})
        print(f"F6: {len(r6)} quarters, avg PnL: ${r6['PnL'].mean():.1f}/t")
    
    # Test forecast
    forecast = load_forecast_data(new_path)
    if forecast is not None:
        print(f"\nForecast data: {forecast.shape[0]} rows, Years: {forecast['Year'].min()}-{forecast['Year'].max()}")
        for i in range(6):
            r = compute_forecast(forecast, i, {})
            if not r.empty:
                print(f"F{i+1} Forecast: {len(r)} years, avg Formula: ${r['Formula'].mean():.1f}/t")
    
    print("\n" + "=" * 60)
    print("Testing with OLD file (0902_1700)")
    print("=" * 60)
    
    hist = load_historical_prices(old_path)
    phos = load_phosphate_prices(old_path)
    print(f"Historical: {hist.shape[0]} rows, Years: {hist['Year'].min()}-{hist['Year'].max()}")
    
    for i, fn in enumerate([compute_formula_1, compute_formula_2, compute_formula_3]):
        r = fn(hist, {})
        print(f"F{i+1}: {len(r)} quarters, avg PnL: ${r['PnL'].mean():.1f}/t")
    
    r4 = compute_formula_4(hist, {'_phos': phos})
    print(f"F4: {len(r4)} quarters, avg PnL: ${r4['PnL'].mean():.1f}/t")
    
    r5 = compute_formula_5(hist, {'_phos': phos})
    print(f"F5: {len(r5)} quarters, avg PnL: ${r5['PnL'].mean():.1f}/t")
