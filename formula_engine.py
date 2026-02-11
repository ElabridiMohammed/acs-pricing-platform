"""
ACS Pricing Formula Engine
Loads historical data & computes 5 pricing formulas from the Excel simulator.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_historical_prices(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='Historical prices S_ACS', header=None)
    
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


def load_base_data(xlsx_path: str) -> dict:
    df = pd.read_excel(xlsx_path, sheet_name='Base Data', header=None)
    
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
        'fixed_floor': cell(5, 7), 'fixed_cap': cell(11, 7),
        'annual_volume_kt': cell(9, 3),
        'formula_pct': cell(27, 3), 'spot_pct': cell(28, 3),
    }


# ============================================================
# INDIVIDUAL FORMULA COMPUTATIONS
# ============================================================

def _quarterly_series(hist: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Group historical monthly data into quarterly averages."""
    groups = hist.groupby(['Year', 'Quarter'])
    result = groups[columns].mean().reset_index()
    result['Period'] = result['Year'].astype(str) + ' Q' + result['Quarter'].astype(str)
    return result


def _smooth_3m(series: pd.Series) -> pd.Series:
    """3-month rolling average (smooth)."""
    return series.rolling(window=3, min_periods=1).mean()


def compute_formula_1(hist: pd.DataFrame, params: dict) -> pd.DataFrame:
    """F1: Sulfur Indexing Only — P = a × (S_weighted × conv_ratio + prod_cost) + b"""
    a = params.get('a', 1.3)
    b = params.get('b', 25)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    conv_ratio = params.get('conv_ratio', 0.33)
    prod_cost = params.get('prod_cost', 20)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    df['Formula'] = a * (df['S_weighted'] * conv_ratio + prod_cost) + b
    
    q = _quarterly_series(df, ['ACS_CFR_NAfrica', 'S_weighted', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_2(hist: pd.DataFrame, params: dict) -> pd.DataFrame:
    """F2: Smooth Sulfur Indexing — P = a × (Smooth_3m(S) × conv_ratio + prod_cost) + b"""
    a = params.get('a', 1.4)
    b = params.get('b', 15)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    conv_ratio = params.get('conv_ratio', 0.33)
    prod_cost = params.get('prod_cost', 20)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    df['S_smooth'] = _smooth_3m(df['S_weighted'])
    df['Formula'] = a * (df['S_smooth'] * conv_ratio + prod_cost) + b
    
    q = _quarterly_series(df, ['ACS_CFR_NAfrica', 'S_weighted', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_3(hist: pd.DataFrame, params: dict) -> pd.DataFrame:
    """F3: Last Month ACS Indexing — P = a × ACS_weighted(m-1)"""
    a = params.get('a', 1.0)
    w_na = params.get('acs_na', 0.05)
    w_eu = params.get('acs_eu', 0.75)
    w_jp = params.get('acs_jp', 0.0)
    w_ch = params.get('acs_ch', 0.20)
    
    df = hist.copy()
    df['ACS_weighted'] = (
        w_na * df['ACS_CFR_NAfrica'].fillna(0) +
        w_eu * df['IPP_Europe'].fillna(0) +
        w_jp * df['IPP_Japan'].fillna(0) +
        w_ch * df['IPP_China'].fillna(0)
    )
    df['ACS_lagged'] = df['ACS_weighted'].shift(1)
    df['Formula'] = a * df['ACS_lagged']
    
    q = _quarterly_series(df, ['ACS_CFR_NAfrica', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_4(hist: pd.DataFrame, phos: pd.DataFrame, params: dict) -> pd.DataFrame:
    """F4: S & DAP Variation — P = ACS0 × (a + b × S/S0 + c × DAP/DAP0)"""
    a = params.get('a', 0.65)
    b = params.get('b', 0.30)
    c = params.get('c', 0.05)
    acs0 = params.get('acs0', 110)
    s0 = params.get('s0', 130)
    dap0 = params.get('dap0', 500)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    
    # Merge DAP data by year (annual phosphate data)
    if phos is not None and 'DAP_Avg' in phos.columns:
        dap_map = phos.set_index('Year')['DAP_Avg'].to_dict()
        df['DAP'] = df['Year'].map(dap_map)
    else:
        df['DAP'] = dap0
    
    df['Formula'] = acs0 * (a + b * df['S_weighted'] / s0 + c * df['DAP'].fillna(dap0) / dap0)
    
    q = _quarterly_series(df, ['ACS_CFR_NAfrica', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


def compute_formula_5(hist: pd.DataFrame, phos: pd.DataFrame, params: dict) -> pd.DataFrame:
    """F5: Smooth S & Smooth DAP — P = ACS0 × (a + b × Sm(S)/S0 + c × Sm(DAP)/DAP0)"""
    a = params.get('a', 0.60)
    b = params.get('b', 0.30)
    c = params.get('c', 0.10)
    acs0 = params.get('acs0', 110)
    s0 = params.get('s0', 130)
    dap0 = params.get('dap0', 500)
    w_me = params.get('sulfur_me', 0.7)
    w_na = params.get('sulfur_na', 0.3)
    
    df = hist.copy()
    df['S_weighted'] = w_me * df['S_CFR_ME'] + w_na * df['S_CFR_NA']
    df['S_smooth'] = _smooth_3m(df['S_weighted'])
    
    if phos is not None and 'DAP_Avg' in phos.columns:
        dap_map = phos.set_index('Year')['DAP_Avg'].to_dict()
        df['DAP'] = df['Year'].map(dap_map)
        df['DAP_smooth'] = _smooth_3m(df['DAP'].fillna(dap0))
    else:
        df['DAP_smooth'] = dap0
    
    df['Formula'] = acs0 * (a + b * df['S_smooth'] / s0 + c * df['DAP_smooth'] / dap0)
    
    q = _quarterly_series(df, ['ACS_CFR_NAfrica', 'Formula'])
    q.rename(columns={'ACS_CFR_NAfrica': 'Market'}, inplace=True)
    q['PnL'] = q['Market'] - q['Formula']
    return q.dropna(subset=['Market', 'Formula'])


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
    path = str(Path(__file__).parent / "0902_1700_ACS Pricing simulator_MG (1).xlsx")
    print("Testing formula engine...")
    
    hist = load_historical_prices(path)
    phos = load_phosphate_prices(path)
    print(f"Historical: {hist.shape[0]} rows, Years: {hist['Year'].min()}-{hist['Year'].max()}")
    
    for i, fn in enumerate([compute_formula_1, compute_formula_2, compute_formula_3], 1):
        if i < 3:
            r = fn(hist, {})
        else:
            r = fn(hist, {})
        print(f"F{i}: {len(r)} quarters, avg PnL: ${r['PnL'].mean():.1f}/t")
    
    for i, fn in enumerate([compute_formula_4, compute_formula_5], 4):
        r = fn(hist, phos, {})
        print(f"F{i}: {len(r)} quarters, avg PnL: ${r['PnL'].mean():.1f}/t")
