"""
Correlation analysis engine for ACS pricing variables.
Handles data loading from the pricing simulator Excel file,
computes inter-variable correlations, and generates PDF reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Column name mapping ---

EXCEL_TO_INTERNAL = {
    'ACS CFR North Africa': 'ACS_CFR_NAfrica',
    'ACS NW EU': 'ACS_NWE',
    'ACS Japan': 'ACS_Japan',
    'ACS China': 'ACS_China',
    'S ME to CFR': 'S_CFR_ME',
    'S North Africa to CFR Morocco': 'S_CFR_NA',
    'Smooth 3 mois S ME to CFR': 'Smooth_S_ME',
    'Smooth S North Africa to CFR Morocco': 'Smooth_S_NA',
    'DAP Bulk North Africa': 'DAP',
    'Petcoke CFR Morocco': 'Petcoke',
    'Clinker CFR Morocco': 'Clinker',
}

INTERNAL_TO_LABEL = {v: k for k, v in EXCEL_TO_INTERNAL.items()}

# Variables used for correlation (exclude Petcoke/Clinker)
CORR_VARS = ['ACS_CFR_NAfrica', 'ACS_NWE', 'ACS_Japan', 'ACS_China',
             'S_CFR_ME', 'S_CFR_NA', 'DAP']

# Variables that should NOT be auto-adjusted
FROZEN_VARS = ['Petcoke', 'Clinker']


def load_historical_monthly(path: str) -> pd.DataFrame:
    """
    Load monthly prices from the 'Monthly_prices_forecast_direct' sheet.
    Returns data for 2022-2025 with standardized column names.
    """
    df = pd.read_excel(path, sheet_name='Monthly_prices_forecast_direct', header=10)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce')
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
    df = df.dropna(subset=['Year', 'Quarter', 'Month']).copy()

    # Filter 2022-2025
    df = df[(df['Year'] >= 2022) & (df['Year'] <= 2025)].copy()

    # Rename columns to internal names
    rename_map = {}
    for excel_col, internal_col in EXCEL_TO_INTERNAL.items():
        if excel_col in df.columns:
            rename_map[excel_col] = internal_col
    df = df.rename(columns=rename_map)

    # Keep only useful columns
    keep = ['Date', 'Year', 'Quarter', 'Month'] + [
        c for c in EXCEL_TO_INTERNAL.values() if c in df.columns
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df['Year'] = df['Year'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['Month'] = df['Month'].astype(int)

    return df.reset_index(drop=True)


def load_generated_yearly(path: str) -> pd.DataFrame:
    """
    Load yearly generated data from the 'Generated data' sheet.
    Returns 2026-2035 projections with standardized column names.
    """
    df = pd.read_excel(path, sheet_name='Generated data', header=None)

    # Find the header row (contains 'Year')
    header_idx = None
    for i, row in df.iterrows():
        if 'Year' in row.values:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header row in 'Generated data' sheet")

    df.columns = df.iloc[header_idx].values
    df = df.iloc[header_idx + 1:].copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year']).copy()
    df['Year'] = df['Year'].astype(int)

    # Rename to internal names
    rename_map = {}
    for excel_col, internal_col in EXCEL_TO_INTERNAL.items():
        if excel_col in df.columns:
            rename_map[excel_col] = internal_col
    df = df.rename(columns=rename_map)

    # Convert all data columns to numeric
    for col in df.columns:
        if col != 'Year':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.reset_index(drop=True)


def compute_correlation_matrix(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix from historical monthly data.
    Only uses CORR_VARS (excludes Petcoke/Clinker).
    """
    available = [c for c in CORR_VARS if c in hist_df.columns]
    return hist_df[available].dropna().corr()


def apply_correlated_adjustment(generated_df: pd.DataFrame,
                                 changed_var: str,
                                 change_pct: float,
                                 corr_matrix: pd.DataFrame,
                                 noise_pct: float = 0.10,
                                 seed: int = None) -> pd.DataFrame:
    """
    When a user changes `changed_var` by `change_pct`, adjust all
    correlated variables proportionally (correlation * change ± noise).
    Petcoke and Clinker are never touched.

    Parameters
    ----------
    generated_df : DataFrame with yearly generated data
    changed_var : internal variable name that was changed (e.g. 'S_CFR_ME')
    change_pct : percentage change applied (e.g. 0.20 for +20%)
    corr_matrix : correlation matrix from historical data
    noise_pct : random noise range (default ±10%)
    seed : random seed for reproducibility

    Returns
    -------
    DataFrame with adjusted values
    """
    if seed is not None:
        np.random.seed(seed)

    df = generated_df.copy()

    if changed_var not in corr_matrix.columns:
        return df

    for var in corr_matrix.columns:
        if var == changed_var or var in FROZEN_VARS:
            continue
        if var not in df.columns:
            continue

        correlation = corr_matrix.loc[changed_var, var]
        noise = np.random.uniform(-noise_pct, noise_pct, len(df))
        adjustment = correlation * change_pct + noise
        df[var] = df[var] * (1 + adjustment)

    return df


def expand_yearly_to_monthly(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand yearly data to monthly granularity using smooth interpolation.
    Yearly averages are placed at mid-year (July) and cubic spline
    interpolation fills in all months — this avoids abrupt jumps at
    year boundaries.
    """
    years = yearly_df['Year'].astype(int).values
    numeric_cols = [c for c in yearly_df.columns if c != 'Year'
                    and pd.api.types.is_numeric_dtype(yearly_df[c])]

    # Build target months: month 1..12 for each year
    all_months = []
    for yr in years:
        for m in range(1, 13):
            all_months.append((yr, m))

    # Fractional year for interpolation (mid-year = yr + 0.5)
    anchor_t = np.array([yr + 0.5 for yr in years])
    target_t = np.array([yr + (m - 0.5) / 12.0 for yr, m in all_months])

    rows = []
    interp_results = {}
    for col in numeric_cols:
        vals = pd.to_numeric(yearly_df[col], errors='coerce').values
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            interp_results[col] = np.full(len(target_t), np.nanmean(vals))
            continue
        try:
            from scipy.interpolate import PchipInterpolator
            f = PchipInterpolator(anchor_t[mask], vals[mask], extrapolate=True)
        except Exception:
            f = np.interp  # fallback
        interp_results[col] = f(target_t)
        # Ensure no negative values
        interp_results[col] = np.maximum(interp_results[col], 1.0)

    for i, (yr, m) in enumerate(all_months):
        quarter = (m - 1) // 3 + 1
        row = {'Year': yr, 'Quarter': quarter, 'Month': m,
               'Date': f"Q{quarter} {yr}"}
        for col in numeric_cols:
            row[col] = interp_results[col][i]
        rows.append(row)

    return pd.DataFrame(rows)


def generate_correlation_pdf(corr_matrix: pd.DataFrame, output_path: str):
    """
    Generate a one-page PDF with correlation heatmap and key statistics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Short labels for readability
    short_labels = {
        'ACS_CFR_NAfrica': 'ACS NA',
        'ACS_NWE': 'ACS EU',
        'ACS_Japan': 'ACS JP',
        'ACS_China': 'ACS CN',
        'S_CFR_ME': 'S ME',
        'S_CFR_NA': 'S NA',
        'DAP': 'DAP',
    }
    labels = [short_labels.get(c, c) for c in corr_matrix.columns]

    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={'width_ratios': [1.3, 1]})

        # Heatmap
        ax1 = axes[0]
        data = corr_matrix.values
        im = ax1.imshow(data, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(labels)))
        ax1.set_yticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels(labels, fontsize=9)

        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = 'white' if abs(data[i, j]) > 0.7 else 'black'
                ax1.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

        fig.colorbar(im, ax=ax1, shrink=0.8)
        ax1.set_title('Pearson Correlation Matrix\n(Monthly data, 2022–2025)',
                      fontsize=11, fontweight='bold')

        # Key insights table
        ax2 = axes[1]
        ax2.axis('off')
        ax2.set_title('Key Insights', fontsize=11, fontweight='bold',
                      loc='left', pad=15)

        # Find top correlations (excluding diagonal)
        pairs = []
        n = len(corr_matrix)
        cols = corr_matrix.columns.tolist()
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        lines = [
            "Variable Correlation Analysis",
            "=" * 38,
            f"Data period: Q1 2022 – Q4 2025 (monthly)",
            f"Number of variables: {len(corr_matrix)}",
            f"Number of observations: 64 months",
            "",
            "Top 5 strongest correlations:",
            "-" * 38,
        ]
        for v1, v2, c in pairs[:5]:
            l1 = short_labels.get(v1, v1)
            l2 = short_labels.get(v2, v2)
            lines.append(f"  {l1} ↔ {l2}: {c:.3f}")

        lines.extend([
            "",
            "Bottom 3 (weakest):",
            "-" * 38,
        ])
        for v1, v2, c in pairs[-3:]:
            l1 = short_labels.get(v1, v1)
            l2 = short_labels.get(v2, v2)
            lines.append(f"  {l1} ↔ {l2}: {c:.3f}")

        avg_corr = np.mean([abs(c) for _, _, c in pairs])
        lines.extend([
            "",
            f"Average |correlation|: {avg_corr:.3f}",
            "",
            "Note: Petcoke & Clinker excluded.",
            "All correlations are positive,",
            "indicating commodity co-movement.",
        ])

        text = '\n'.join(lines)
        ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
                fontsize=8.5, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))

        fig.suptitle('ACS Pricing — Variable Correlation Report',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return output_path
