"""
Volume-Based Revenue Analysis Engine
Calculates weighted average prices and revenues based on volume splits
for the ACS project with OCP contract scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_price_time_series(xlsx_path: str) -> pd.DataFrame:
    """
    Load ACS CFR North Africa (market reference) and FOB India/Indonesia proxy
    from Historical + Generated data sheets.

    Returns DataFrame with columns:
        Year, Month, ACS_CFR_NAfrica, FOB_IO (FOB India/Indonesia proxy),
        S_FOB_ME (Sulphur FOB Middle East)
    """
    xl = pd.ExcelFile(xlsx_path)

    # --- Historical monthly from Monthly_prices_forecast_direct ---
    dfm = xl.parse('Monthly_prices_forecast_direct', header=10)
    dfm['Year'] = pd.to_numeric(dfm.get('Year'), errors='coerce')
    dfm['Month'] = pd.to_numeric(dfm.get('Month'), errors='coerce')
    dfm = dfm.dropna(subset=['Year', 'Month']).copy()

    hist = pd.DataFrame()
    hist['Year'] = dfm['Year'].astype(int)
    hist['Month'] = dfm['Month'].astype(int)

    # ACS CFR North Africa = market reference
    for col in dfm.columns:
        if 'ACS CFR North Africa' in str(col) or 'ACS_CFR_NAfrica' in str(col):
            hist['ACS_CFR_NAfrica'] = pd.to_numeric(dfm[col], errors='coerce')
            break

    # FOB India/Indonesia proxy: use ACS FOB Japan as closest available index
    # (the Excel has Japan/South Korea FOB, which is the Indian Ocean reference)
    for col in dfm.columns:
        cstr = str(col).lower()
        if 'japan' in cstr or 'acs japan' in cstr:
            hist['FOB_IO'] = pd.to_numeric(dfm[col], errors='coerce')
            break

    # Sulphur FOB ME as another reference
    for col in dfm.columns:
        cstr = str(col).lower()
        if 's me' in cstr or 'sulfur me' in cstr or 'sulphur' in cstr and 'me' in cstr:
            hist['S_FOB_ME'] = pd.to_numeric(dfm[col], errors='coerce')
            break

    # ACS FOB NW Europe
    for col in dfm.columns:
        cstr = str(col).lower()
        if 'nw eu' in cstr or 'europe' in cstr:
            hist['ACS_FOB_NWE'] = pd.to_numeric(dfm[col], errors='coerce')
            break

    hist = hist[hist['Year'] <= 2025].copy()

    # --- Generated yearly projections (2026-2035) ---
    try:
        dfg = xl.parse('Generated data', header=None)
        header_idx = None
        for i in range(dfg.shape[0]):
            if 'Year' in [str(x) for x in dfg.iloc[i].tolist()]:
                header_idx = i
                break
        if header_idx is not None:
            dfg.columns = dfg.iloc[header_idx].values
            dfg = dfg.iloc[header_idx + 1:].copy()
            dfg['Year'] = pd.to_numeric(dfg['Year'], errors='coerce')
            dfg = dfg.dropna(subset=['Year']).copy()
            dfg['Year'] = dfg['Year'].astype(int)

            proj = pd.DataFrame()
            rows = []
            for _, row in dfg.iterrows():
                yr = int(row['Year'])
                acs = None
                fob_io = None
                s_me = None
                acs_nwe = None
                for c in dfg.columns:
                    cs = str(c)
                    if 'ACS CFR North Africa' in cs:
                        acs = pd.to_numeric(row[c], errors='coerce')
                    if 'ACS Japan' in cs or 'Japan' in cs:
                        fob_io = pd.to_numeric(row[c], errors='coerce')
                    if 'S ME' in cs:
                        s_me = pd.to_numeric(row[c], errors='coerce')
                    if 'ACS NW EU' in cs or 'NW EU' in cs:
                        acs_nwe = pd.to_numeric(row[c], errors='coerce')
                # Expand to monthly
                for m in range(1, 13):
                    rows.append({
                        'Year': yr,
                        'Month': m,
                        'ACS_CFR_NAfrica': acs,
                        'FOB_IO': fob_io,
                        'S_FOB_ME': s_me,
                        'ACS_FOB_NWE': acs_nwe,
                    })
            if rows:
                proj = pd.DataFrame(rows)
                hist = pd.concat([hist, proj], ignore_index=True)
    except Exception:
        pass

    # Ensure FOB_IO exists; fallback to ACS_CFR_NAfrica * 0.85
    if 'FOB_IO' not in hist.columns or hist['FOB_IO'].isna().all():
        hist['FOB_IO'] = hist.get('ACS_CFR_NAfrica', 120) * 0.85

    hist = hist.sort_values(['Year', 'Month']).reset_index(drop=True)
    return hist


def compute_yearly_averages(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Group monthly time series to yearly averages.
    Returns DataFrame with Year and average price columns.
    """
    num_cols = [c for c in ts.columns if c not in ('Year', 'Month')]
    yearly = ts.groupby('Year')[num_cols].mean().reset_index()
    return yearly


def compute_negotiated_price(fob_io: float, coeff_a: float = 1.0,
                             premium_b: float = 0.0) -> float:
    """Negotiated price = A * FOB_IO + B"""
    return coeff_a * fob_io + premium_b


def compute_weighted_avg_price(fixed_price: float, negotiated_price: float,
                               vf_pct: float) -> float:
    """
    Weighted average price = Vf * Fixed + Vn * Negotiated
    vf_pct: fraction (0-1) of volume at fixed price
    """
    vn_pct = 1.0 - vf_pct
    return vf_pct * fixed_price + vn_pct * negotiated_price


def build_project_revenue_table(yearly: pd.DataFrame,
                                total_vol_kt: float = 750.0,
                                fixed_pct: float = 0.70,
                                fixed_price: float = 110.0,
                                coeff_a: float = 1.0,
                                premium_b: float = 0.0) -> pd.DataFrame:
    """
    Build project revenue table (assumes 100% of production sold).

    Returns DataFrame with:
        Year, ACS_CFR_NAfrica, FOB_IO, Negotiated_Price, Weighted_Avg_Price,
        Revenue_M (millions $), Vol_Fixed_KT, Vol_Var_KT
    """
    var_pct = 1.0 - fixed_pct
    vol_fixed = total_vol_kt * fixed_pct
    vol_var = total_vol_kt * var_pct

    result = yearly[['Year']].copy()
    result['ACS_CFR_NAfrica'] = yearly.get('ACS_CFR_NAfrica', 120.0)
    result['FOB_IO'] = yearly.get('FOB_IO', 100.0)
    result['Fixed_Price'] = fixed_price
    result['Negotiated_Price'] = result['FOB_IO'].apply(
        lambda x: compute_negotiated_price(x, coeff_a, premium_b)
    )
    result['Weighted_Avg_Price'] = result.apply(
        lambda r: compute_weighted_avg_price(
            fixed_price, r['Negotiated_Price'], fixed_pct
        ), axis=1
    )
    result['Vol_Fixed_KT'] = vol_fixed
    result['Vol_Var_KT'] = vol_var
    result['Revenue_Fixed_M'] = vol_fixed * fixed_price / 1000.0
    result['Revenue_Var_M'] = vol_var * result['Negotiated_Price'] / 1000.0
    result['Revenue_Total_M'] = result['Revenue_Fixed_M'] + result['Revenue_Var_M']

    return result


def build_ocp_scenarios(yearly: pd.DataFrame,
                        total_vol_kt: float = 750.0,
                        fixed_price: float = 110.0,
                        coeff_a: float = 1.0,
                        premium_b: float = 0.0) -> dict:
    """
    Build three OCP purchasing scenarios:
      1. OCP buys 70% at fixed only (525 KT)
      2. OCP buys 100% at fixed price (750 KT)
      3. OCP buys 70% fixed + 30% market/negotiated

    Returns dict of scenario name -> DataFrame with:
        Year, OCP_Volume_KT, OCP_Cost_M, Market_Cost_M, Value_Gain_M,
        Gain_Fixed_M, Gain_Var_M
    """
    market_price = yearly.get('ACS_CFR_NAfrica', pd.Series([120.0]))
    fob_io = yearly.get('FOB_IO', pd.Series([100.0]))
    neg_price = fob_io.apply(lambda x: compute_negotiated_price(x, coeff_a, premium_b))

    scenarios = {}

    # Scenario 1: OCP buys 70% at fixed only
    vol = total_vol_kt * 0.70
    df1 = yearly[['Year']].copy()
    df1['Scenario'] = '70% Fixed Only'
    df1['OCP_Volume_KT'] = vol
    df1['OCP_Cost_M'] = vol * fixed_price / 1000.0
    df1['Market_Cost_M'] = vol * market_price / 1000.0
    df1['Value_Gain_M'] = df1['Market_Cost_M'] - df1['OCP_Cost_M']
    df1['Gain_Fixed_M'] = df1['Value_Gain_M']
    df1['Gain_Var_M'] = 0.0
    df1['Avg_Price_Paid'] = fixed_price
    scenarios['70% Fixed Only'] = df1

    # Scenario 2: OCP buys 100% at fixed
    vol = total_vol_kt
    df2 = yearly[['Year']].copy()
    df2['Scenario'] = '100% Fixed'
    df2['OCP_Volume_KT'] = vol
    df2['OCP_Cost_M'] = vol * fixed_price / 1000.0
    df2['Market_Cost_M'] = vol * market_price / 1000.0
    df2['Value_Gain_M'] = df2['Market_Cost_M'] - df2['OCP_Cost_M']
    df2['Gain_Fixed_M'] = df2['Value_Gain_M']
    df2['Gain_Var_M'] = 0.0
    df2['Avg_Price_Paid'] = fixed_price
    scenarios['100% Fixed'] = df2

    # Scenario 3: 70% fixed + 30% negotiated
    vol_fixed = total_vol_kt * 0.70
    vol_var = total_vol_kt * 0.30
    df3 = yearly[['Year']].copy()
    df3['Scenario'] = '70% Fixed + 30% Negotiated'
    df3['OCP_Volume_KT'] = total_vol_kt
    df3['OCP_Cost_M'] = (vol_fixed * fixed_price + vol_var * neg_price) / 1000.0
    df3['Market_Cost_M'] = total_vol_kt * market_price / 1000.0
    df3['Value_Gain_M'] = df3['Market_Cost_M'] - df3['OCP_Cost_M']
    df3['Gain_Fixed_M'] = vol_fixed * (market_price - fixed_price) / 1000.0
    df3['Gain_Var_M'] = vol_var * (market_price - neg_price) / 1000.0
    df3['Avg_Price_Paid'] = (vol_fixed * fixed_price + vol_var * neg_price) / total_vol_kt
    scenarios['70% Fixed + 30% Negotiated'] = df3

    return scenarios


def build_custom_ocp_scenario(yearly: pd.DataFrame,
                              total_vol_kt: float,
                              ocp_pct: float,
                              fixed_pct: float,
                              fixed_price: float,
                              coeff_a: float,
                              premium_b: float) -> pd.DataFrame:
    """
    Custom OCP scenario with user-defined purchase % and split.

    ocp_pct: fraction of total production OCP purchases (0-1)
    fixed_pct: fraction of OCP's purchase at fixed price (0-1)
    """
    market_price = yearly.get('ACS_CFR_NAfrica', pd.Series([120.0]))
    fob_io = yearly.get('FOB_IO', pd.Series([100.0]))
    neg_price = fob_io.apply(lambda x: compute_negotiated_price(x, coeff_a, premium_b))

    ocp_vol = total_vol_kt * ocp_pct
    vol_fixed = ocp_vol * fixed_pct
    vol_var = ocp_vol * (1.0 - fixed_pct)

    df = yearly[['Year']].copy()
    df['OCP_Volume_KT'] = ocp_vol
    df['Vol_Fixed_KT'] = vol_fixed
    df['Vol_Var_KT'] = vol_var
    df['OCP_Cost_M'] = (vol_fixed * fixed_price + vol_var * neg_price) / 1000.0
    df['Market_Cost_M'] = ocp_vol * market_price / 1000.0
    df['Value_Gain_M'] = df['Market_Cost_M'] - df['OCP_Cost_M']
    df['Gain_Fixed_M'] = vol_fixed * (market_price - fixed_price) / 1000.0
    df['Gain_Var_M'] = vol_var * (market_price - neg_price) / 1000.0
    df['Avg_Price_Paid'] = np.where(
        ocp_vol > 0,
        (vol_fixed * fixed_price + vol_var * neg_price) / ocp_vol,
        0
    )
    return df


def compute_breakeven_price(fixed_price: float, fixed_pct: float,
                            coeff_a: float, premium_b: float,
                            fob_io: float) -> float:
    """
    Compute the market price at which OCP's blended price equals market.
    i.e., the breakeven ACS CFR North Africa price.
    """
    neg = compute_negotiated_price(fob_io, coeff_a, premium_b)
    return fixed_pct * fixed_price + (1.0 - fixed_pct) * neg


def build_sensitivity_table(yearly: pd.DataFrame,
                            total_vol_kt: float,
                            fixed_price: float,
                            coeff_a: float,
                            premium_b: float,
                            splits: list = None) -> pd.DataFrame:
    """
    Generate sensitivity analysis for different volume splits.
    Returns table with avg revenue for each split.
    """
    if splits is None:
        splits = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rows = []
    for fp in splits:
        proj = build_project_revenue_table(
            yearly, total_vol_kt, fp, fixed_price, coeff_a, premium_b
        )
        avg_rev = proj['Revenue_Total_M'].mean()
        avg_price = proj['Weighted_Avg_Price'].mean()
        rows.append({
            'Fixed %': f"{fp*100:.0f}%",
            'Variable %': f"{(1-fp)*100:.0f}%",
            'Vol Fixed (KT)': total_vol_kt * fp,
            'Vol Var (KT)': total_vol_kt * (1 - fp),
            'Avg Weighted Price ($/t)': round(avg_price, 1),
            'Avg Annual Revenue ($M)': round(avg_rev, 1),
        })
    return pd.DataFrame(rows)
