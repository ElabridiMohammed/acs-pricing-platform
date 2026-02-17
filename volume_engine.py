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
    Load ACS CFR North Africa (market reference) and ACS FOB NW Europe
    from Historical + Generated data sheets.

    Returns DataFrame with columns:
        Year, Month, ACS_CFR_NAfrica, ACS_FOB_NWE (FOB NW Europe),
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

    # ACS FOB NW Europe — index used in the variable price formula
    for col in dfm.columns:
        cstr = str(col).lower()
        if 'nw eu' in cstr or 'europe' in cstr:
            hist['ACS_FOB_NWE'] = pd.to_numeric(dfm[col], errors='coerce')
            break

    # Sulphur FOB ME as another reference
    for col in dfm.columns:
        cstr = str(col).lower()
        if 's me' in cstr or 'sulfur me' in cstr or 'sulphur' in cstr and 'me' in cstr:
            hist['S_FOB_ME'] = pd.to_numeric(dfm[col], errors='coerce')
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
                s_me = None
                acs_nwe = None
                for c in dfg.columns:
                    cs = str(c)
                    if 'ACS CFR North Africa' in cs:
                        acs = pd.to_numeric(row[c], errors='coerce')
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
                        'ACS_FOB_NWE': acs_nwe,
                        'S_FOB_ME': s_me,
                    })
            if rows:
                proj = pd.DataFrame(rows)
                hist = pd.concat([hist, proj], ignore_index=True)
    except Exception:
        pass

    # Ensure ACS_FOB_NWE exists; fallback to ACS_CFR_NAfrica * 0.85
    if 'ACS_FOB_NWE' not in hist.columns or hist['ACS_FOB_NWE'].isna().all():
        hist['ACS_FOB_NWE'] = hist.get('ACS_CFR_NAfrica', 120) * 0.85

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


def compute_negotiated_price(fob_nwe: float, coeff_a: float = 1.0,
                             premium_b: float = 0.0) -> float:
    """Negotiated price = A * FOB_NWE + B"""
    return coeff_a * fob_nwe + premium_b


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
                                premium_b: float = 0.0,
                                inflation_rate: float = 0.0,
                                inflation_start_year: int = 2030) -> pd.DataFrame:
    """
    Build project revenue table (assumes 100% of production sold).

    Returns DataFrame with:
        Year, ACS_CFR_NAfrica, ACS_FOB_NWE, Negotiated_Price, Weighted_Avg_Price,
        Revenue_M (millions $), Vol_Fixed_KT, Vol_Var_KT
    """
    var_pct = 1.0 - fixed_pct
    vol_fixed = total_vol_kt * fixed_pct
    vol_var = total_vol_kt * var_pct

    result = yearly[['Year']].copy()
    result['ACS_CFR_NAfrica'] = yearly.get('ACS_CFR_NAfrica', 120.0)
    result['ACS_FOB_NWE'] = yearly.get('ACS_FOB_NWE', 100.0)

    # Apply inflation to fixed price (only from inflation_start_year)
    result['Fixed_Price'] = result['Year'].apply(
        lambda y: fixed_price * (1 + inflation_rate) ** max(y - inflation_start_year, 0)
    )
    result['Negotiated_Price'] = result['ACS_FOB_NWE'].apply(
        lambda x: compute_negotiated_price(x, coeff_a, premium_b)
    )
    result['Weighted_Avg_Price'] = result.apply(
        lambda r: compute_weighted_avg_price(
            r['Fixed_Price'], r['Negotiated_Price'], fixed_pct
        ), axis=1
    )
    result['Vol_Fixed_KT'] = vol_fixed
    result['Vol_Var_KT'] = vol_var
    result['Revenue_Fixed_M'] = vol_fixed * result['Fixed_Price'] / 1000.0
    result['Revenue_Var_M'] = vol_var * result['Negotiated_Price'] / 1000.0
    result['Revenue_Total_M'] = result['Revenue_Fixed_M'] + result['Revenue_Var_M']

    return result


def build_ocp_scenarios(yearly: pd.DataFrame,
                        total_vol_kt: float = 750.0,
                        fixed_pct: float = 0.70,
                        fixed_price: float = 110.0,
                        coeff_a: float = 1.0,
                        premium_b: float = 0.0,
                        inflation_rate: float = 0.0,
                        inflation_start_year: int = 2030) -> dict:
    """
    Build three OCP purchasing scenarios based on variable purchase %:
      1. Fixed only (OCP buys 0% of remaining variable volume)
      2. Fixed + 100% of remaining (OCP buys all)
      3. Fixed + 50% of remaining (OCP buys half)

    Returns dict of scenario name -> DataFrame with:
        Year, OCP_Volume_KT, OCP_Cost_M, Market_Cost_M, Value_Gain_M,
        Gain_Fixed_M, Gain_Var_M
    """
    market_price = yearly.get('ACS_CFR_NAfrica', pd.Series([120.0]))
    fob_nwe = yearly.get('ACS_FOB_NWE', pd.Series([100.0]))
    neg_price = fob_nwe.apply(lambda x: compute_negotiated_price(x, coeff_a, premium_b))

    years = yearly['Year'].astype(int)
    inflation_factors = (1 + inflation_rate) ** np.maximum(years - inflation_start_year, 0)
    fp_inflated = fixed_price * inflation_factors

    vol_fixed = total_vol_kt * fixed_pct
    remaining_vol = total_vol_kt * (1 - fixed_pct)
    fp_int = int(round(fixed_pct * 100))
    vp_int = int(round((1 - fixed_pct) * 100))

    scenarios = {}

    # Scenario 1: OCP buys fixed only (0% of remaining)
    df1 = yearly[['Year']].copy()
    s1_label = f'{fp_int}% Fixed Only'
    df1['Scenario'] = s1_label
    df1['OCP_Volume_KT'] = vol_fixed
    df1['OCP_Cost_M'] = vol_fixed * fp_inflated / 1000.0
    df1['Market_Cost_M'] = vol_fixed * market_price / 1000.0
    df1['Value_Gain_M'] = df1['Market_Cost_M'] - df1['OCP_Cost_M']
    df1['Gain_Fixed_M'] = df1['Value_Gain_M']
    df1['Gain_Var_M'] = 0.0
    df1['Avg_Price_Paid'] = fp_inflated.values
    scenarios[f'{fp_int}% Fixed Only'] = df1

    # Scenario 2: OCP buys fixed + 100% of remaining
    vol_var_2 = remaining_vol
    ocp_vol_2 = vol_fixed + vol_var_2
    df2 = yearly[['Year']].copy()
    s2_label = f'100% ({fp_int}F + {vp_int}V)'
    df2['Scenario'] = f'100% ({fp_int}% Fixed + {vp_int}% Variable)'
    df2['OCP_Volume_KT'] = ocp_vol_2
    df2['OCP_Cost_M'] = (vol_fixed * fp_inflated + vol_var_2 * neg_price) / 1000.0
    df2['Market_Cost_M'] = ocp_vol_2 * market_price / 1000.0
    df2['Value_Gain_M'] = df2['Market_Cost_M'] - df2['OCP_Cost_M']
    df2['Gain_Fixed_M'] = vol_fixed * (market_price - fp_inflated) / 1000.0
    df2['Gain_Var_M'] = vol_var_2 * (market_price - neg_price) / 1000.0
    df2['Avg_Price_Paid'] = np.where(
        ocp_vol_2 > 0,
        (vol_fixed * fp_inflated + vol_var_2 * neg_price) / ocp_vol_2,
        0
    )
    scenarios[s2_label] = df2

    # Scenario 3: OCP buys fixed + 50% of remaining
    vol_var_3 = remaining_vol * 0.50
    ocp_vol_3 = vol_fixed + vol_var_3
    mid_pct = int(round((fixed_pct + (1 - fixed_pct) * 0.50) * 100))
    half_vp = int(round(vp_int * 0.50))
    df3 = yearly[['Year']].copy()
    s3_label = f'{mid_pct}% ({fp_int}F + {half_vp}V)'
    df3['Scenario'] = f'{mid_pct}% ({fp_int}% Fixed + {half_vp}% Variable)'
    df3['OCP_Volume_KT'] = ocp_vol_3
    df3['OCP_Cost_M'] = (vol_fixed * fp_inflated + vol_var_3 * neg_price) / 1000.0
    df3['Market_Cost_M'] = ocp_vol_3 * market_price / 1000.0
    df3['Value_Gain_M'] = df3['Market_Cost_M'] - df3['OCP_Cost_M']
    df3['Gain_Fixed_M'] = vol_fixed * (market_price - fp_inflated) / 1000.0
    df3['Gain_Var_M'] = vol_var_3 * (market_price - neg_price) / 1000.0
    df3['Avg_Price_Paid'] = np.where(
        ocp_vol_3 > 0,
        (vol_fixed * fp_inflated + vol_var_3 * neg_price) / ocp_vol_3,
        0
    )
    scenarios[s3_label] = df3

    return scenarios


def build_custom_ocp_scenario(yearly: pd.DataFrame,
                              total_vol_kt: float,
                              fixed_pct: float,
                              var_buy_pct: float,
                              fixed_price: float,
                              coeff_a: float,
                              premium_b: float,
                              inflation_rate: float = 0.0,
                              inflation_start_year: int = 2030) -> pd.DataFrame:
    """
    Custom OCP scenario with two independent parameters:
      fixed_pct: fraction of total production sold at fixed price (0-1)
      var_buy_pct: fraction of remaining variable volume OCP buys (0-1)

    Fixed volume = fixed_pct * total_vol_kt (all bought by OCP)
    Variable volume = (1 - fixed_pct) * var_buy_pct * total_vol_kt
    """
    market_price = yearly.get('ACS_CFR_NAfrica', pd.Series([120.0]))
    fob_nwe = yearly.get('ACS_FOB_NWE', pd.Series([100.0]))
    neg_price = fob_nwe.apply(lambda x: compute_negotiated_price(x, coeff_a, premium_b))

    years = yearly['Year'].astype(int)
    inflation_factors = (1 + inflation_rate) ** np.maximum(years - inflation_start_year, 0)
    fp_inflated = fixed_price * inflation_factors

    vol_fixed = total_vol_kt * fixed_pct
    vol_var = total_vol_kt * (1 - fixed_pct) * var_buy_pct
    ocp_vol = vol_fixed + vol_var

    df = yearly[['Year']].copy()
    df['OCP_Volume_KT'] = ocp_vol
    df['Vol_Fixed_KT'] = vol_fixed
    df['Vol_Var_KT'] = vol_var
    df['OCP_Cost_M'] = (vol_fixed * fp_inflated + vol_var * neg_price) / 1000.0
    df['Market_Cost_M'] = ocp_vol * market_price / 1000.0
    df['Value_Gain_M'] = df['Market_Cost_M'] - df['OCP_Cost_M']
    df['Gain_Fixed_M'] = vol_fixed * (market_price - fp_inflated) / 1000.0
    df['Gain_Var_M'] = vol_var * (market_price - neg_price) / 1000.0
    df['Avg_Price_Paid'] = np.where(
        ocp_vol > 0,
        (vol_fixed * fp_inflated + vol_var * neg_price) / ocp_vol,
        0
    )
    return df


def compute_breakeven_price(fixed_price: float, fixed_pct: float,
                            coeff_a: float, premium_b: float,
                            fob_nwe: float) -> float:
    """
    Compute the market price at which OCP's blended price equals market.
    i.e., the breakeven ACS CFR North Africa price.
    """
    neg = compute_negotiated_price(fob_nwe, coeff_a, premium_b)
    return fixed_pct * fixed_price + (1.0 - fixed_pct) * neg


def build_sensitivity_table(yearly: pd.DataFrame,
                            total_vol_kt: float,
                            fixed_price: float,
                            coeff_a: float,
                            premium_b: float,
                            inflation_rate: float = 0.0,
                            inflation_start_year: int = 2030,
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
            yearly, total_vol_kt, fp, fixed_price, coeff_a, premium_b,
            inflation_rate=inflation_rate, inflation_start_year=inflation_start_year
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
