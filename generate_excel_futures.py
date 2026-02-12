"""
Generate Excel file with simulated future price series for S, ACS, DAP, and MAP.
Uses the same stochastic model as the Streamlit app (GBM + spike process).
Author: Mohammed ELARIDI
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ─── Configuration ───────────────────────────────────────────────────────────

START_YEAR = 2025
END_YEAR = 2030
N_SIMS = 500
SMOOTHING = 0.7
SEED = 42
SPIKE_FREQ = 0.5        # spikes per year
SPIKE_INTENSITY = 0.30   # 30%
SPIKE_PERSISTENCE = 4    # months
DECAY_TYPE = 'exponential'
VOL_OVERRIDE = 0.20      # default annual volatility


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_acs_s_outlook(file_path: str, product: str) -> pd.DataFrame:
    """Load ACS or S outlook from the main Excel file."""
    df = pd.read_excel(file_path, sheet_name=f'{product} - CRU Outlook')
    df = df.rename(columns={df.columns[0]: 'Year'})
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_fert_outlook(file_path: str) -> pd.DataFrame:
    """Load DAP/MAP outlook from the fertilizer Excel file."""
    df = pd.read_excel(file_path, header=None)
    # Row 5 has the column labels, row 6+ has data
    col_names = df.iloc[5].tolist()
    col_names[0] = 'Year'
    data = df.iloc[6:].copy()
    data.columns = col_names
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data = data.dropna(subset=['Year'])
    data['Year'] = data['Year'].astype(int)
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


# ─── Stochastic Model (reused from app.py) ───────────────────────────────────

def interpolate_trend(outlook_df: pd.DataFrame, column: str,
                      start_year: int, end_year: int) -> pd.Series:
    df = outlook_df[(outlook_df['Year'] >= start_year - 1) &
                    (outlook_df['Year'] <= end_year + 1)]
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    valid = df[['Year', column]].dropna()
    if len(valid) < 2:
        raise ValueError(f"Not enough data points for '{column}'")

    years = valid['Year'].values
    prices = valid[column].values
    try:
        f = interpolate.interp1d(years, prices, kind='cubic', fill_value='extrapolate')
    except Exception:
        f = interpolate.interp1d(years, prices, kind='linear', fill_value='extrapolate')

    date_range = pd.date_range(start=f'{start_year}-01-01',
                               end=f'{end_year}-12-31', freq='MS')
    fractional_years = date_range.year + (date_range.month - 1) / 12
    return pd.Series(f(fractional_years), index=date_range, name='Trend')


def generate_brownian_motion(n_steps, volatility, n_sims=100,
                             smoothing=0.7, seed=None):
    if seed is not None:
        np.random.seed(seed)
    monthly_vol = (volatility / np.sqrt(12)) * 0.6
    raw = np.random.normal(0, monthly_vol, (n_sims, n_steps))
    smoothed = np.zeros((n_sims, n_steps))
    smoothed[:, 0] = raw[:, 0]
    for t in range(1, n_steps):
        smoothed[:, t] = smoothing * smoothed[:, t-1] + (1 - smoothing) * raw[:, t]
    cumulative = np.cumsum(smoothed, axis=1)
    drift = -0.5 * (monthly_vol ** 2) * np.arange(1, n_steps + 1)
    return cumulative + drift


def generate_spike_process(n_steps, freq, intensity, persistence,
                           decay_type='exponential', seed=None):
    if seed is not None:
        np.random.seed(seed)
    contribution = np.zeros(n_steps)
    p_spike = min(freq / 12.0, 0.3)
    spikes = np.random.binomial(1, p_spike, n_steps)
    for t in range(n_steps):
        if spikes[t] == 1:
            for k in range(persistence + 1):
                if t + k < n_steps:
                    progress = k / persistence
                    decay = (max(0, 1.0 - progress) if decay_type == 'linear'
                             else np.exp(-3.0 * progress))
                    contribution[t + k] += intensity * decay
    return contribution


def simulate_prices(trend, volatility, spike_freq, spike_intensity,
                    spike_persistence, decay_type, smoothing=0.7,
                    n_sims=500, seed=None):
    n_steps = len(trend)
    trend_values = trend.values

    brownian = generate_brownian_motion(n_steps, volatility, n_sims, smoothing, seed)
    all_paths = np.zeros((n_sims, n_steps))

    for i in range(n_sims):
        spikes = generate_spike_process(
            n_steps, spike_freq, spike_intensity, spike_persistence, decay_type,
            seed=(seed + i * 17) if seed else None
        )
        all_paths[i, :] = trend_values * np.exp(brownian[i, :]) * (1 + spikes)

    all_paths = np.maximum(all_paths, 1.0)

    return {
        'dates': trend.index,
        'trend': trend_values,
        'mean_path': np.mean(all_paths, axis=0),
        'percentile_5': np.percentile(all_paths, 5, axis=0),
        'percentile_25': np.percentile(all_paths, 25, axis=0),
        'percentile_75': np.percentile(all_paths, 75, axis=0),
        'percentile_95': np.percentile(all_paths, 95, axis=0),
        'all_paths': all_paths,
    }


# ─── Product Definitions ─────────────────────────────────────────────────────

PRODUCTS = {
    'S': {
        'name': 'Sulphur',
        'file': 'Formules de prix_ACS.xlsx',
        'loader': 'acs_s',
        'product_code': 'S',
        'regions': [
            'FOB Vancouver (spot)',
            'FOB Middle East (spot)',
            'CFR China  (spot)',
            'CFR Brazil (spot)',
            'CFR North Africa (contract)',
            'FOB Tampa (contract)',
        ],
    },
    'ACS': {
        'name': 'Sulfuric Acid',
        'file': 'Formules de prix_ACS.xlsx',
        'loader': 'acs_s',
        'product_code': 'ACS',
        'regions': [
            'CFR US Gulf',
            'CFR Brazil',
            'FOB Japan/South Korea (Spot)',
            'FOB China',
            'CFR Chile – contract',
            'FOB NW Europe',
            'CFR India',
        ],
    },
    'DAP': {
        'name': 'DAP FOB Morocco',
        'file': 'cru outlook fert.xlsx',
        'loader': 'fert',
        'column': 'DAP FOB Morocco',
    },
    'MAP': {
        'name': 'MAP FOB Baltic/Black Sea',
        'file': 'cru outlook fert.xlsx',
        'loader': 'fert',
        'column': 'MAP FOB Baltic/Black Sea',
    },
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).parent

    # Load data sources
    acs_s_path = str(base / 'Formules de prix_ACS.xlsx')
    fert_path = str(base / 'cru outlook fert.xlsx')

    acs_outlook = load_acs_s_outlook(acs_s_path, 'ACS')
    s_outlook = load_acs_s_outlook(acs_s_path, 'S')
    fert_outlook = load_fert_outlook(fert_path)

    output_file = base / f"simulation_futures_{datetime.now():%Y%m%d}.xlsx"

    with pd.ExcelWriter(str(output_file), engine='openpyxl') as writer:

        summary_rows = []

        # ── Sulphur (S) ──────────────────────────────────────────────────
        print("▶ Simulating Sulphur (S)...")
        for region in PRODUCTS['S']['regions']:
            col = region  # keep original spacing (some columns have double spaces)
            try:
                trend = interpolate_trend(s_outlook, col, START_YEAR, END_YEAR)
            except Exception as e:
                print(f"  ⚠ Skipping S / {region}: {e}")
                continue

            results = simulate_prices(trend, VOL_OVERRIDE, SPIKE_FREQ,
                                      SPIKE_INTENSITY, SPIKE_PERSISTENCE,
                                      DECAY_TYPE, SMOOTHING, N_SIMS, SEED)

            sheet_name = _safe_sheet(f"S - {region}")
            df = _build_monthly_df(results)
            df.to_excel(writer, sheet_name=sheet_name)
            summary_rows.append(_build_summary(results, 'S', region))
            print(f"  ✓ S / {region}")

        # ── Sulfuric Acid (ACS) ──────────────────────────────────────────
        print("▶ Simulating Sulfuric Acid (ACS)...")
        for region in PRODUCTS['ACS']['regions']:
            col = region  # keep original spacing
            try:
                trend = interpolate_trend(acs_outlook, col, START_YEAR, END_YEAR)
            except Exception as e:
                print(f"  ⚠ Skipping ACS / {region}: {e}")
                continue

            results = simulate_prices(trend, VOL_OVERRIDE, SPIKE_FREQ,
                                      SPIKE_INTENSITY, SPIKE_PERSISTENCE,
                                      DECAY_TYPE, SMOOTHING, N_SIMS, SEED)

            sheet_name = _safe_sheet(f"ACS - {region}")
            df = _build_monthly_df(results)
            df.to_excel(writer, sheet_name=sheet_name)
            summary_rows.append(_build_summary(results, 'ACS', region))
            print(f"  ✓ ACS / {region}")

        # ── DAP ──────────────────────────────────────────────────────────
        print("▶ Simulating DAP...")
        try:
            trend = interpolate_trend(fert_outlook, PRODUCTS['DAP']['column'],
                                      START_YEAR, END_YEAR)
            results = simulate_prices(trend, VOL_OVERRIDE, SPIKE_FREQ,
                                      SPIKE_INTENSITY, SPIKE_PERSISTENCE,
                                      DECAY_TYPE, SMOOTHING, N_SIMS, SEED)
            df = _build_monthly_df(results)
            df.to_excel(writer, sheet_name='DAP FOB Morocco')
            summary_rows.append(_build_summary(results, 'DAP', 'FOB Morocco'))
            print("  ✓ DAP / FOB Morocco")
        except Exception as e:
            print(f"  ⚠ DAP failed: {e}")

        # ── MAP ──────────────────────────────────────────────────────────
        print("▶ Simulating MAP...")
        try:
            trend = interpolate_trend(fert_outlook, PRODUCTS['MAP']['column'],
                                      START_YEAR, END_YEAR)
            results = simulate_prices(trend, VOL_OVERRIDE, SPIKE_FREQ,
                                      SPIKE_INTENSITY, SPIKE_PERSISTENCE,
                                      DECAY_TYPE, SMOOTHING, N_SIMS, SEED)
            df = _build_monthly_df(results)
            df.to_excel(writer, sheet_name='MAP FOB BalticBlackSea')
            summary_rows.append(_build_summary(results, 'MAP', 'FOB Baltic/Black Sea'))
            print("  ✓ MAP / FOB Baltic/Black Sea")
        except Exception as e:
            print(f"  ⚠ MAP failed: {e}")

        # ── Summary sheet ────────────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Auto-adjust column widths for all sheets
        for ws in writer.book.worksheets:
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    val = str(cell.value) if cell.value is not None else ''
                    max_len = max(max_len, len(val))
                ws.column_dimensions[col_letter].width = min(max_len + 3, 25)

    print(f"\n✅ Excel file saved: {output_file}")
    print(f"   Sheets: {len(summary_rows) + 1} (incl. Summary)")


def _build_monthly_df(results: dict) -> pd.DataFrame:
    """Build a monthly DataFrame with trend, mean, percentiles."""
    return pd.DataFrame({
        'Date': [d.strftime('%Y-%m') for d in results['dates']],
        'CRU Outlook ($/t)': np.round(results['trend'], 2),
        'Sim Mean ($/t)': np.round(results['mean_path'], 2),
        'P5 ($/t)': np.round(results['percentile_5'], 2),
        'P25 ($/t)': np.round(results['percentile_25'], 2),
        'P75 ($/t)': np.round(results['percentile_75'], 2),
        'P95 ($/t)': np.round(results['percentile_95'], 2),
    })


def _build_summary(results: dict, product: str, region: str) -> dict:
    """Build a summary row for the summary sheet."""
    yearly_summaries = {}
    dates = results['dates']
    for year in range(START_YEAR, END_YEAR + 1):
        mask = dates.year == year
        if any(mask):
            yearly_summaries[f'CRU {year}'] = round(np.mean(results['trend'][mask]), 1)
            yearly_summaries[f'Mean {year}'] = round(np.mean(results['mean_path'][mask]), 1)
            yearly_summaries[f'P5 {year}'] = round(np.mean(results['percentile_5'][mask]), 1)
            yearly_summaries[f'P95 {year}'] = round(np.mean(results['percentile_95'][mask]), 1)

    return {
        'Product': product,
        'Region': region,
        'Avg CRU ($/t)': round(np.mean(results['trend']), 1),
        'Avg Sim Mean ($/t)': round(np.mean(results['mean_path']), 1),
        'Overall P5 ($/t)': round(np.mean(results['percentile_5']), 1),
        'Overall P95 ($/t)': round(np.mean(results['percentile_95']), 1),
        **yearly_summaries,
    }


def _safe_sheet(name: str) -> str:
    """Sanitize sheet name for Excel (31 chars, no invalid chars)."""
    for ch in ['/', '\\', '?', '*', '[', ']', ':']:
        name = name.replace(ch, '-')
    return name[:31]


if __name__ == '__main__':
    main()
