################################################################################################################
# Duplicate EDF Energy Statistical Dashboard — New Tariff Comparison & Energy Disaggregation
#
# Processes exportable daily CSV data from the EDF Energy Hub ("At a glance" -> "Download your data as CSV"),
# compares historical costs against the new flat-rate all-electric tariff (effective 19/11/2026), and adds
# an Energy Breakdown & Weekly Peer Comparison section calibrated with EDF's energy disaggregation and the
# household profile (Liberty Court, IG11 0XY, 56 m² / 606 sq ft, 3 occupants, all-electric).
#
# The dashboard produces:
#   - Top KPI Summary Cards (Tariff Change, 86-Day Savings, Annualized Projection, Property Intensity)
#   1. Time Series: Daily Cost Comparison (Historical vs. New Tariff) & Daily kWh
#   2. Scatter & Regression: Historical OLS Tariff Fit vs. New Flat Tariff Line
#   3. Daily Cost Distribution: Marginal Boxplots + Overlaid Histograms & KDE Curves
#   4. Energy Breakdown by Category (EDF Disaggregation w/c 9 Nov 2026: Historical vs. New Tariff)
#   5. Weekly Peer Comparison ("How You Compare": This Week w/c 9 Nov & Week Before w/c 2 Nov)
#   6. Monthly Cost Comparison & Cumulative Savings Trajectory
#
# Execution & Usage (Linux / Python 3.12 64-bit):
#   1. Activate the environment: source .venv/bin/activate
#   2. Execute the script:       python analyze_electricity_new_tariff.py
#   Optional silent mode (save PNG only without opening interactive window):
#                                python analyze_electricity_new_tariff.py --no-show
################################################################################################################

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# --------------------------------------------------------------------------------------------------------------
# 1. Configuration, Tariff, Household Profile & EDF Disaggregation Constants
# --------------------------------------------------------------------------------------------------------------
CSV_FILE = "Usage Analysis using the new Smart Meter Data Log.csv"
OUTPUT_IMAGE = "statistical_analysis_visuals_new_tariff.png"

NEW_TARIFF = {
    "effective_date": pd.Timestamp("2026-11-19"),
    "day_rate_gbp": 0.2660,        # 26.60p per kWh (includes VAT)
    "night_rate_gbp": 0.2660,      # 26.60p per kWh (includes VAT — flat rate)
    "standing_charge_gbp": 0.3607, # 36.07p per day (includes VAT)
}

HOUSEHOLD_PROFILE = {
    "property_name": "Liberty Court (Ground Floor, 2-Bed Flat)",
    "floor_area_sqm": 56.0,
    "floor_area_sqft": 606,
    "rooms_sqm": {
        "Reception/Kitchen (6.2x3.4m)": 21.08,
        "Bedroom 1 (4.7x3.0m)": 14.10,
        "Bedroom 2 (3.2x3.0m)": 9.60,
        "Bathroom (2.0x1.9m)": 3.80,
        "Hall & Boiler Cupboard": 7.42,
    },
    "postcode": "IG11 0XY",
    "location_label": "Barking, East London",
    "latitude": 51.52786732124059,
    "longitude": 0.11762578261683365,
    "climate": "UK Temperate Maritime (Cfb)",
    "occupants": 3,
    "heating_type": "All-Electric (space heating, hot water, cooking)",
}

# EDF Energy Hub disaggregation & peer benchmarks (screenshot: Week beginning 9th Nov 2026)
EDF_DISAGGREGATION = {
    "week_start": "2026-11-09",
    "week_end": "2026-11-15",
    "prev_week_start": "2026-11-02",
    "prev_week_end": "2026-11-08",
    "categories_gbp": {
        "Home entertainment": 8.39,
        "Fridges and freezers": 7.03,
        "Cooking": 6.74,
        "Hot water": 5.79,
        "Heating": 5.09,
        "Laundry and dishwashing": 4.10,
        "Lighting": 3.66,
        "Always on": 3.25,
        "Other": 0.26,
    },
    "peer_comparison": {
        "this_week": {
            "label": "This Week (w/c 9 Nov 2026)",
            "average": 29.42,
            "most_efficient": 10.33,
            "your_usage_edf": 44.31,
        },
        "week_before": {
            "label": "Week Before (w/c 2 Nov 2026)",
            "average": 27.48,
            "most_efficient": 11.20,
            "your_usage_edf": 44.89,
        },
    },
}

# Consistent color palette
COLORS = {
    "historical": "#c0392b",      # Crimson
    "csv_week": "#e67e22",        # Warm amber-orange
    "new_tariff": "#0d9488",      # Emerald-teal
    "consumption": "#3b82f6",     # Blue
    "most_efficient": "#2563eb",  # Royal blue
    "average": "#64748b",         # Slate gray
    "savings": "#15803d",         # Forest green
    "savings_fill": "#10b981",    # Mint green
    "scatter": "#5b21b6",         # Deep indigo
    "text_dark": "#1e293b",       # Slate 800
    "text_muted": "#475569",      # Slate 600
    "card_bg": "#f8fafc",         # Slate 50
    "card_border": "#cbd5e1",     # Slate 300
}


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load EDF smart meter CSV, clean empty rows, parse dates, and compute new tariff metrics."""
    try:
        raw = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        sys.exit(1)

    df = raw.dropna(subset=["Timestamp"]).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y")
    df = df.sort_values("Timestamp").set_index("Timestamp")
    df = df.rename(
        columns={
            "Electricity consumption (kWh)": "Consumption_kWh",
            "Electricity cost (£)": "Cost_GBP",
        }
    )

    df["New_Cost_GBP"] = (
        df["Consumption_kWh"] * NEW_TARIFF["day_rate_gbp"]
        + NEW_TARIFF["standing_charge_gbp"]
    )
    df["Savings_GBP"] = df["Cost_GBP"] - df["New_Cost_GBP"]
    df["Savings_Pct"] = np.where(
        df["Cost_GBP"] > 0, df["Savings_GBP"] / df["Cost_GBP"] * 100.0, 0.0
    )
    return df


def build_disaggregation_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute per-category shares, kWh estimates, and historical vs. new tariff weekly/annual costs."""
    w_start = pd.Timestamp(EDF_DISAGGREGATION["week_start"])
    w_end = pd.Timestamp(EDF_DISAGGREGATION["week_end"])
    p_start = pd.Timestamp(EDF_DISAGGREGATION["prev_week_start"])
    p_end = pd.Timestamp(EDF_DISAGGREGATION["prev_week_end"])

    this_week_df = df.loc[(df.index >= w_start) & (df.index <= w_end)]
    prev_week_df = df.loc[(df.index >= p_start) & (df.index <= p_end)]

    this_week_kwh = float(this_week_df["Consumption_kWh"].sum())
    this_week_csv_gbp = float(this_week_df["Cost_GBP"].sum())
    this_week_new_gbp = float(this_week_df["New_Cost_GBP"].sum())
    this_week_days = len(this_week_df)

    prev_week_kwh = float(prev_week_df["Consumption_kWh"].sum())
    prev_week_csv_gbp = float(prev_week_df["Cost_GBP"].sum())
    prev_week_new_gbp = float(prev_week_df["New_Cost_GBP"].sum())
    prev_week_days = len(prev_week_df)

    cats = EDF_DISAGGREGATION["categories_gbp"]
    edf_week_total = float(sum(cats.values()))
    new_week_energy_gbp = this_week_kwh * NEW_TARIFF["day_rate_gbp"]
    new_week_sc_gbp = this_week_days * NEW_TARIFF["standing_charge_gbp"]
    period_days = len(df)
    period_kwh = float(df["Consumption_kWh"].sum())
    annual_kwh = float(df["Consumption_kWh"].mean() * 365.0)

    rows = []
    for cat, edf_gbp in cats.items():
        share = edf_gbp / edf_week_total
        est_week_kwh = share * this_week_kwh
        new_energy_gbp = est_week_kwh * NEW_TARIFF["day_rate_gbp"]
        new_total_gbp = share * this_week_new_gbp
        diff_vs_edf = edf_gbp - new_total_gbp
        est_period_new_gbp = share * float(df["New_Cost_GBP"].sum())
        est_annual_new_gbp = share * float(df["New_Cost_GBP"].mean() * 365.0)
        est_annual_kwh = share * annual_kwh
        rows.append(
            {
                "Category": cat,
                "Share_Pct": share * 100.0,
                "Est_Week_kWh": est_week_kwh,
                "EDF_Week_GBP": edf_gbp,
                "New_Energy_GBP": new_energy_gbp,
                "New_Total_GBP": new_total_gbp,
                "Diff_vs_EDF_GBP": diff_vs_edf,
                "Est_Period_New_GBP": est_period_new_gbp,
                "Est_Annual_kWh": est_annual_kwh,
                "Est_Annual_New_GBP": est_annual_new_gbp,
            }
        )

    disagg_df = pd.DataFrame(rows).sort_values("EDF_Week_GBP", ascending=False).reset_index(drop=True)
    weekly_meta = {
        "edf_week_total": edf_week_total,
        "this_week_days": this_week_days,
        "this_week_kwh": this_week_kwh,
        "this_week_csv_gbp": this_week_csv_gbp,
        "this_week_new_gbp": this_week_new_gbp,
        "new_week_energy_gbp": new_week_energy_gbp,
        "new_week_sc_gbp": new_week_sc_gbp,
        "prev_week_days": prev_week_days,
        "prev_week_kwh": prev_week_kwh,
        "prev_week_csv_gbp": prev_week_csv_gbp,
        "prev_week_new_gbp": prev_week_new_gbp,
        "period_days": period_days,
        "period_kwh": period_kwh,
    }
    return disagg_df, weekly_meta


def print_console_report(
    df: pd.DataFrame,
    hist_reg: tuple[float, float, float, float, float],
    disagg_df: pd.DataFrame,
    weekly_meta: dict,
) -> None:
    """Print detailed household profile, descriptive stats, tariff comparison, and breakdown tables."""
    slope, intercept, r_value, p_value, _ = hist_reg
    n_days = len(df)
    mean_kwh = float(df["Consumption_kWh"].mean())
    hist_total = float(df["Cost_GBP"].sum())
    new_total = float(df["New_Cost_GBP"].sum())
    total_savings = hist_total - new_total
    savings_pct = total_savings / hist_total * 100.0
    annual_kwh = mean_kwh * 365.0
    annual_hist = float(df["Cost_GBP"].mean() * 365.0)
    annual_new = float(df["New_Cost_GBP"].mean() * 365.0)
    annual_sav = annual_hist - annual_new

    print("=" * 78)
    print(" HOUSEHOLD PROFILE & ENERGY INTENSITY")
    print("=" * 78)
    print(f"Property       : {HOUSEHOLD_PROFILE['property_name']}")
    print(
        f"Floor Area     : {HOUSEHOLD_PROFILE['floor_area_sqm']:.0f} m² "
        f"({HOUSEHOLD_PROFILE['floor_area_sqft']} sq ft)"
    )
    print(
        f"Location       : {HOUSEHOLD_PROFILE['postcode']} ({HOUSEHOLD_PROFILE['location_label']}) | "
        f"Coords ({HOUSEHOLD_PROFILE['latitude']:.5f}, {HOUSEHOLD_PROFILE['longitude']:.5f})"
    )
    print(f"Climate        : {HOUSEHOLD_PROFILE['climate']}")
    print(
        f"Occupants/Fuel : {HOUSEHOLD_PROFILE['occupants']} occupants | "
        f"{HOUSEHOLD_PROFILE['heating_type']}"
    )
    print(
        f"Daily Avg Use  : {mean_kwh:.3f} kWh/day  |  "
        f"{mean_kwh / HOUSEHOLD_PROFILE['occupants']:.2f} kWh/person/day  |  "
        f"{mean_kwh / HOUSEHOLD_PROFILE['floor_area_sqm']:.3f} kWh/m²/day "
        f"({annual_kwh / HOUSEHOLD_PROFILE['floor_area_sqm']:.1f} kWh/m²/yr est.)\n"
    )

    print("=" * 78)
    print(f" DESCRIPTIVE STATISTICS ({n_days} days: {df.index.min().strftime('%d/%m/%Y')} – {df.index.max().strftime('%d/%m/%Y')})")
    print("=" * 78)
    stats_df = df[["Consumption_kWh", "Cost_GBP", "New_Cost_GBP", "Savings_GBP"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.95]
    ).T
    stats_df["range"] = stats_df["max"] - stats_df["min"]
    print(stats_df[["mean", "std", "min", "max", "range", "25%", "50%", "75%", "95%"]].round(3))
    print()

    print("=" * 78)
    print(" HISTORICAL TARIFF (OLS FIT) vs. NEW TARIFF (FROM 19/11/2026)")
    print("=" * 78)
    print(f"Historical OLS : Cost(£) = {slope:.4f} * kWh + {intercept:.4f}  (R² = {r_value**2:.4f}, r = {r_value:.4f}, p = {p_value:.3e})")
    print(
        f"New Flat Tariff: Cost(£) = {NEW_TARIFF['day_rate_gbp']:.4f} * kWh + "
        f"{NEW_TARIFF['standing_charge_gbp']:.4f}  "
        f"(Day/Night {NEW_TARIFF['day_rate_gbp']*100:.2f}p/kWh, SC {NEW_TARIFF['standing_charge_gbp']*100:.2f}p/day incl. VAT)"
    )
    print(
        f"{n_days}-Day Totals  : Historical £{hist_total:.2f} (£{df['Cost_GBP'].mean():.2f}/d) -> "
        f"New £{new_total:.2f} (£{df['New_Cost_GBP'].mean():.2f}/d) | "
        f"Saved £{total_savings:.2f} (-{savings_pct:.2f}%, £{df['Savings_GBP'].mean():.2f}/d)"
    )
    print(
        f"Annualized Est.: {annual_kwh:,.0f} kWh/yr | Historical £{annual_hist:,.2f}/yr -> "
        f"New £{annual_new:,.2f}/yr | Est. Savings £{annual_sav:,.2f}/yr (pre-winter pace)\n"
    )

    print("=" * 78)
    print(" MONTHLY COST & SAVINGS SUMMARY")
    print("=" * 78)
    monthly = df.assign(Month=df.index.strftime("%Y-%m")).groupby("Month").agg(
        Days=("Consumption_kWh", "count"),
        kWh=("Consumption_kWh", "sum"),
        Hist_GBP=("Cost_GBP", "sum"),
        New_GBP=("New_Cost_GBP", "sum"),
        Savings_GBP=("Savings_GBP", "sum"),
    )
    monthly["Savings_Pct"] = monthly["Savings_GBP"] / monthly["Hist_GBP"] * 100.0
    print(monthly.round(2))
    print()

    print("=" * 78)
    print(" ENERGY DISAGGREGATION BREAKDOWN (EDF w/c 9 Nov 2026 — Estimates)")
    print("=" * 78)
    print(
        f"Week 9–15 Nov  : {weekly_meta['this_week_kwh']:.3f} kWh | "
        f"EDF Reported £{weekly_meta['edf_week_total']:.2f} | "
        f"CSV Daily Sum £{weekly_meta['this_week_csv_gbp']:.2f} | "
        f"New Tariff £{weekly_meta['this_week_new_gbp']:.2f} "
        f"(£{weekly_meta['new_week_energy_gbp']:.2f} energy + £{weekly_meta['new_week_sc_gbp']:.2f} SC)\n"
    )
    disp = disagg_df[
        [
            "Category",
            "Share_Pct",
            "Est_Week_kWh",
            "EDF_Week_GBP",
            "New_Energy_GBP",
            "New_Total_GBP",
            "Diff_vs_EDF_GBP",
            "Est_Annual_New_GBP",
        ]
    ].copy()
    print(disp.round(2).to_string(index=False))
    print()

    print("=" * 78)
    print(" WEEKLY PEER BENCHMARKS ('HOW YOU COMPARE')")
    print("=" * 78)
    peers = EDF_DISAGGREGATION["peer_comparison"]
    for key, wk_csv, wk_new, wk_kwh in [
        ("this_week", weekly_meta["this_week_csv_gbp"], weekly_meta["this_week_new_gbp"], weekly_meta["this_week_kwh"]),
        ("week_before", weekly_meta["prev_week_csv_gbp"], weekly_meta["prev_week_new_gbp"], weekly_meta["prev_week_kwh"]),
    ]:
        p = peers[key]
        print(
            f"{p['label']}: {wk_kwh:.2f} kWh | "
            f"EDF £{p['your_usage_edf']:.2f} (+£{p['your_usage_edf'] - p['average']:.2f} vs avg) | "
            f"CSV £{wk_csv:.2f} | "
            f"New Tariff £{wk_new:.2f} (+£{wk_new - p['average']:.2f} vs avg) | "
            f"Avg £{p['average']:.2f} | Efficient £{p['most_efficient']:.2f}"
        )
    print()


def draw_kpi_card(
    ax: plt.Axes,
    title: str,
    headline: str,
    detail: str,
    accent_color: str,
    bg_color: str = "#f8fafc",
) -> None:
    """Render a clean rounded KPI summary card inside a dedicated axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.015, 0.04),
        0.97,
        0.92,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor=bg_color,
        edgecolor=COLORS["card_border"],
        linewidth=1.2,
    )
    ax.add_patch(card)

    # Top accent bar
    accent = FancyBboxPatch(
        (0.015, 0.90),
        0.97,
        0.06,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        facecolor=accent_color,
        edgecolor="none",
    )
    ax.add_patch(accent)

    ax.text(
        0.5,
        0.76,
        title,
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=COLORS["text_muted"],
    )
    ax.text(
        0.5,
        0.51,
        headline,
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["text_dark"],
    )
    ax.text(
        0.5,
        0.22,
        detail,
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["text_muted"],
        linespacing=1.45,
    )


def build_dashboard(
    df: pd.DataFrame,
    hist_reg: tuple[float, float, float, float, float],
    disagg_df: pd.DataFrame,
    weekly_meta: dict,
) -> plt.Figure:
    """Build the 6-panel + KPI banner duplicate dashboard comparing historical vs. new tariff and disaggregation."""
    slope, intercept, r_value, _, _ = hist_reg

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
        }
    )

    fig = plt.figure(figsize=(22, 14), facecolor="white")
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("EDF Energy — Historical vs. New Tariff & Disaggregation")
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=[0.48, 1.0, 1.10],
        hspace=0.38,
        wspace=0.28,
    )
    fig.subplots_adjust(left=0.088, right=0.962, top=0.975, bottom=0.060)

    # ----------------------------------------------------------------------------------------------------------
    # Row 0: 4 KPI Summary Cards
    # ----------------------------------------------------------------------------------------------------------
    kpi_gs = gs[0, :].subgridspec(1, 4, wspace=0.08)

    n_days = len(df)
    hist_total = float(df["Cost_GBP"].sum())
    new_total = float(df["New_Cost_GBP"].sum())
    sav_total = hist_total - new_total
    sav_pct = sav_total / hist_total * 100.0
    mean_kwh = float(df["Consumption_kWh"].mean())
    annual_kwh = mean_kwh * 365.0
    annual_hist = float(df["Cost_GBP"].mean() * 365.0)
    annual_new = float(df["New_Cost_GBP"].mean() * 365.0)
    annual_sav = annual_hist - annual_new
    unit_drop_pct = (1.0 - NEW_TARIFF["day_rate_gbp"] / slope) * 100.0
    sc_drop_pct = (1.0 - NEW_TARIFF["standing_charge_gbp"] / intercept) * 100.0

    draw_kpi_card(
        fig.add_subplot(kpi_gs[0, 0]),
        "TARIFF CHANGE (FROM 19 NOV 2026)",
        "26.60p/kWh + 36.07p/day",
        f"vs. Hist. OLS ~{slope*100:.2f}p/kWh + {intercept*100:.2f}p/d\n"
        f"Flat Day/Night (incl. VAT) · Unit −{unit_drop_pct:.1f}% · SC −{sc_drop_pct:.1f}%",
        accent_color=COLORS["new_tariff"],
    )
    draw_kpi_card(
        fig.add_subplot(kpi_gs[0, 1]),
        f"{n_days}-DAY ACTUAL vs. NEW TARIFF",
        f"Saved £{sav_total:.2f}  (−{sav_pct:.1f}%)",
        f"Hist: £{hist_total:.2f} (£{df['Cost_GBP'].mean():.2f}/d) → New: £{new_total:.2f} (£{df['New_Cost_GBP'].mean():.2f}/d)\n"
        f"Avg daily saving: £{df['Savings_GBP'].mean():.2f}/day across {df['Consumption_kWh'].sum():,.1f} kWh",
        accent_color=COLORS["savings"],
        bg_color="#f0fdf4",
    )
    draw_kpi_card(
        fig.add_subplot(kpi_gs[0, 2]),
        "ANNUALIZED PROJECTION (PRE-WINTER PACE)",
        f"£{annual_new:,.0f}/yr  (−£{annual_sav:,.0f}/yr)",
        f"Hist. pace: £{annual_hist:,.0f}/yr across ~{annual_kwh:,.0f} kWh/yr\n"
        f"Note: Winter all-electric heating will lift actual winter kWh",
        accent_color=COLORS["consumption"],
    )
    draw_kpi_card(
        fig.add_subplot(kpi_gs[0, 3]),
        "PROPERTY ENERGY INTENSITY",
        f"{mean_kwh:.2f} kWh/day  ({mean_kwh / HOUSEHOLD_PROFILE['occupants']:.2f}/person)",
        f"{mean_kwh / HOUSEHOLD_PROFILE['floor_area_sqm']:.3f} kWh/m²/d · ~{annual_kwh / HOUSEHOLD_PROFILE['floor_area_sqm']:.0f} kWh/m²/yr · {HOUSEHOLD_PROFILE['occupants']} Occupants\n"
        f"Liberty Court ({HOUSEHOLD_PROFILE['floor_area_sqm']:.0f} m² / {HOUSEHOLD_PROFILE['floor_area_sqft']} sq ft) · {HOUSEHOLD_PROFILE['postcode']} · All-Electric",
        accent_color=COLORS["scatter"],
    )

    # ----------------------------------------------------------------------------------------------------------
    # Panel 1 (Row 1, Col 0): Time Series — Daily Cost Comparison & Consumption
    # ----------------------------------------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1, 0])
    ax1_kwh = ax1.twinx()

    ax1_kwh.bar(
        df.index,
        df["Consumption_kWh"],
        width=0.75,
        color=COLORS["consumption"],
        alpha=0.16,
        label="Daily kWh (right axis)",
        zorder=1,
    )
    ax1_kwh.set_ylabel("Consumption (kWh)", fontsize=9.5, color=COLORS["consumption"])
    ax1_kwh.tick_params(axis="y", colors=COLORS["consumption"], labelsize=8.5)
    ax1_kwh.set_ylim(0, df["Consumption_kWh"].max() * 1.32)
    ax1_kwh.grid(False)

    # Ensure primary cost lines render cleanly above background bars
    ax1.set_zorder(ax1_kwh.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax1.plot(
        df.index,
        df["Cost_GBP"],
        color=COLORS["historical"],
        linewidth=1.5,
        marker="o",
        markersize=2.8,
        label=f"Historical Cost (mean £{df['Cost_GBP'].mean():.2f}/d)",
        zorder=4,
    )
    ax1.plot(
        df.index,
        df["New_Cost_GBP"],
        color=COLORS["new_tariff"],
        linewidth=1.7,
        label=f"New Tariff Cost (mean £{df['New_Cost_GBP'].mean():.2f}/d)",
        zorder=5,
    )
    ax1.fill_between(
        df.index,
        df["New_Cost_GBP"],
        df["Cost_GBP"],
        where=df["Cost_GBP"] >= df["New_Cost_GBP"],
        color=COLORS["savings_fill"],
        alpha=0.22,
        label="Daily Savings",
        zorder=3,
    )
    ax1.axhline(
        df["Cost_GBP"].mean(),
        color=COLORS["historical"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )
    ax1.axhline(
        df["New_Cost_GBP"].mean(),
        color=COLORS["new_tariff"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )

    ax1.set_title("1. Time Series: Daily Cost (Historical vs. New Tariff)", fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("Daily Cost (£)", fontsize=9.5)
    ax1.set_ylim(0, df["Cost_GBP"].max() * 1.30)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.tick_params(axis="x", rotation=28, labelsize=8.5)
    ax1.tick_params(axis="y", labelsize=8.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_kwh.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        ncol=2,
        fontsize=7.6,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )

    # ----------------------------------------------------------------------------------------------------------
    # Panel 2 (Row 1, Col 1): Regression & Tariff Comparison
    # ----------------------------------------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    x_vals = df["Consumption_kWh"].to_numpy()
    y_vals = df["Cost_GBP"].to_numpy()

    ax2.scatter(
        x_vals,
        y_vals,
        color=COLORS["scatter"],
        alpha=0.58,
        s=34,
        edgecolors="white",
        linewidth=0.5,
        label="Historical Daily Observations",
        zorder=3,
    )

    x_grid = np.linspace(max(0.0, x_vals.min() - 1.5), x_vals.max() + 1.5, 200)
    y_hist_fit = slope * x_grid + intercept
    y_new_line = NEW_TARIFF["day_rate_gbp"] * x_grid + NEW_TARIFF["standing_charge_gbp"]

    ax2.plot(
        x_grid,
        y_hist_fit,
        color=COLORS["historical"],
        linewidth=2.2,
        label=f"Historical OLS: y = {slope:.3f}x + {intercept:.3f} (r={r_value:.3f})",
        zorder=4,
    )
    ax2.plot(
        x_grid,
        y_new_line,
        color=COLORS["new_tariff"],
        linewidth=2.3,
        linestyle="--",
        label=f"New Flat Tariff: y = {NEW_TARIFF['day_rate_gbp']:.3f}x + {NEW_TARIFF['standing_charge_gbp']:.3f}",
        zorder=5,
    )
    ax2.fill_between(
        x_grid,
        y_new_line,
        y_hist_fit,
        where=y_hist_fit >= y_new_line,
        color=COLORS["savings_fill"],
        alpha=0.16,
        label="Savings Gap (widens with usage)",
        zorder=2,
    )

    # Annotate savings gap at mean and peak consumption
    peak_kwh = float(df["Consumption_kWh"].max())
    for x_ref, tag, dx, dy in [
        (mean_kwh, "Mean", 1.2, -2.05),
        (peak_kwh, "Peak", -8.2, -2.55),
    ]:
        y_h = slope * x_ref + intercept
        y_n = NEW_TARIFF["day_rate_gbp"] * x_ref + NEW_TARIFF["standing_charge_gbp"]
        gap = y_h - y_n
        ax2.plot([x_ref, x_ref], [y_n, y_h], color=COLORS["savings"], linewidth=1.6, zorder=6)
        ax2.annotate(
            f"{tag} ({x_ref:.1f} kWh)\n−£{gap:.2f}/day",
            xy=(x_ref, (y_h + y_n) / 2.0),
            xytext=(x_ref + dx, max(0.8, y_n + dy)),
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["savings"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f0fdf4", "edgecolor": COLORS["savings"], "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": COLORS["savings"], "lw": 1.1},
            zorder=7,
        )

    ax2.set_title("2. Regression & Tariff Slope Comparison", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Consumption (kWh)", fontsize=9.5)
    ax2.set_ylabel("Daily Cost (£)", fontsize=9.5)
    ax2.set_ylim(0, max(y_vals.max(), y_hist_fit.max()) * 1.12)
    ax2.tick_params(labelsize=8.5)
    ax2.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.92)

    # ----------------------------------------------------------------------------------------------------------
    # Panel 3 (Row 1, Col 2): Daily Cost Distribution (Marginal Boxplots + Histogram & KDE)
    # ----------------------------------------------------------------------------------------------------------
    dist_gs = gs[1, 2].subgridspec(2, 1, height_ratios=[0.30, 1.0], hspace=0.06)
    ax3_box = fig.add_subplot(dist_gs[0, 0])
    ax3_hist = fig.add_subplot(dist_gs[1, 0], sharex=ax3_box)

    hist_costs = df["Cost_GBP"].to_numpy()
    new_costs = df["New_Cost_GBP"].to_numpy()

    bp = ax3_box.boxplot(
        [new_costs, hist_costs],
        orientation="horizontal",
        tick_labels=["New", "Hist."],
        widths=0.56,
        patch_artist=True,
        medianprops={"color": "#0f172a", "linewidth": 1.8},
        whiskerprops={"color": "#475569", "linewidth": 1.1},
        capprops={"color": "#475569", "linewidth": 1.1},
        flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.6},
    )
    for patch, col in zip(bp["boxes"], [COLORS["new_tariff"], COLORS["historical"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor(col)

    ax3_box.set_title("3. Daily Cost Distribution (Boxplots, Histograms & KDE)", fontsize=12, fontweight="bold", pad=8)
    ax3_box.tick_params(axis="x", labelbottom=False)
    ax3_box.tick_params(axis="y", labelsize=8.5)

    bins = np.linspace(min(new_costs.min(), hist_costs.min()) - 0.3, max(new_costs.max(), hist_costs.max()) + 0.3, 15)
    ax3_hist.hist(
        hist_costs,
        bins=bins,
        density=True,
        color=COLORS["historical"],
        alpha=0.28,
        edgecolor=COLORS["historical"],
        linewidth=0.7,
        label=f"Historical (med £{np.median(hist_costs):.2f})",
    )
    ax3_hist.hist(
        new_costs,
        bins=bins,
        density=True,
        color=COLORS["new_tariff"],
        alpha=0.30,
        edgecolor=COLORS["new_tariff"],
        linewidth=0.7,
        label=f"New Tariff (med £{np.median(new_costs):.2f})",
    )

    x_kde = np.linspace(bins[0], bins[-1], 300)
    kde_hist = stats.gaussian_kde(hist_costs)(x_kde)
    kde_new = stats.gaussian_kde(new_costs)(x_kde)
    ax3_hist.plot(x_kde, kde_hist, color=COLORS["historical"], linewidth=2.0)
    ax3_hist.plot(x_kde, kde_new, color=COLORS["new_tariff"], linewidth=2.0)

    ax3_hist.axvline(np.median(hist_costs), color=COLORS["historical"], linestyle="--", linewidth=1.2, alpha=0.85)
    ax3_hist.axvline(np.median(new_costs), color=COLORS["new_tariff"], linestyle="--", linewidth=1.2, alpha=0.85)
    ax3_hist.set_xlabel("Daily Cost (£)", fontsize=9.5)
    ax3_hist.set_ylabel("Density", fontsize=9.5)
    ax3_hist.tick_params(labelsize=8.5)
    ax3_hist.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.92)

    # ----------------------------------------------------------------------------------------------------------
    # Panel 4 (Row 2, Col 0): Energy Breakdown by Category (EDF Disaggregation)
    # ----------------------------------------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    y_pos = np.arange(len(disagg_df))
    bar_h = 0.37

    ax4.barh(
        y_pos - bar_h / 2,
        disagg_df["EDF_Week_GBP"],
        height=bar_h,
        color=COLORS["historical"],
        alpha=0.86,
        label=f"EDF Historical (£{weekly_meta['edf_week_total']:.2f})",
    )
    ax4.barh(
        y_pos + bar_h / 2,
        disagg_df["New_Total_GBP"],
        height=bar_h,
        color=COLORS["new_tariff"],
        alpha=0.90,
        label=f"New Tariff Est. (£{weekly_meta['this_week_new_gbp']:.2f})",
    )

    for idx, row in disagg_df.iterrows():
        x_max = max(row["EDF_Week_GBP"], row["New_Total_GBP"])
        ax4.text(
            x_max + 0.16,
            idx,
            f"£{row['EDF_Week_GBP']:.2f} → £{row['New_Total_GBP']:.2f}  ({row['Share_Pct']:.1f}% | {row['Est_Week_kWh']:.1f} kWh)",
            va="center",
            ha="left",
            fontsize=7.8,
            color=COLORS["text_dark"],
        )

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(disagg_df["Category"], fontsize=8.5)
    ax4.invert_yaxis()
    ax4.set_xlim(0, disagg_df["EDF_Week_GBP"].max() * 1.92)
    ax4.set_xlabel("Weekly Cost (£) — w/c 9 Nov 2026 (124.77 kWh)", fontsize=9.5)
    ax4.set_title(
        "4. Energy Breakdown by Category (EDF Disaggregation Est.)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax4.text(
        0.5,
        1.01,
        f"New tariff = £{weekly_meta['new_week_energy_gbp']:.2f} energy + £{weekly_meta['new_week_sc_gbp']:.2f} standing charge",
        transform=ax4.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["text_muted"],
    )
    ax4.tick_params(axis="x", labelsize=8.5)
    ax4.legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.92)

    # ----------------------------------------------------------------------------------------------------------
    # Panel 5 (Row 2, Col 1): How You Compare — Weekly Peer Benchmarks
    # ----------------------------------------------------------------------------------------------------------
    peer_gs = gs[2, 1].subgridspec(2, 1, hspace=0.52)
    peers = EDF_DISAGGREGATION["peer_comparison"]
    peer_configs = [
        (
            peer_gs[0, 0],
            peers["this_week"],
            weekly_meta["this_week_csv_gbp"],
            weekly_meta["this_week_new_gbp"],
            weekly_meta["this_week_kwh"],
            "5a. How You Compare — This Week (w/c 9 Nov)",
        ),
        (
            peer_gs[1, 0],
            peers["week_before"],
            weekly_meta["prev_week_csv_gbp"],
            weekly_meta["prev_week_new_gbp"],
            weekly_meta["prev_week_kwh"],
            "5b. How You Compare — Week Before (w/c 2 Nov)",
        ),
    ]

    for sub_spec, p_data, csv_wk, new_wk, kwh_wk, sub_title in peer_configs:
        ax_p = fig.add_subplot(sub_spec)
        labels = [
            "EDF Reported",
            "CSV Historical",
            "New Tariff Est.",
            "Average Household",
            "Most Efficient",
        ]
        vals = [
            p_data["your_usage_edf"],
            csv_wk,
            new_wk,
            p_data["average"],
            p_data["most_efficient"],
        ]
        cols = [
            COLORS["historical"],
            COLORS["csv_week"],
            COLORS["new_tariff"],
            COLORS["average"],
            COLORS["most_efficient"],
        ]
        yp = np.arange(len(labels))
        ax_p.barh(yp, vals, height=0.60, color=cols, alpha=0.90)
        ax_p.axvline(p_data["average"], color=COLORS["average"], linestyle="--", linewidth=1.2, alpha=0.85)

        for i, (lbl, v) in enumerate(zip(labels, vals)):
            if lbl in ("EDF Reported", "CSV Historical", "New Tariff Est."):
                diff = v - p_data["average"]
                suffix = f" (+£{diff:.2f} vs avg)" if diff >= 0 else f" (−£{abs(diff):.2f} vs avg)"
            else:
                suffix = ""
            ax_p.text(
                v + 0.8,
                i,
                f"£{v:.2f}{suffix}",
                va="center",
                ha="left",
                fontsize=7.9,
                fontweight="bold" if lbl in ("EDF Reported", "New Tariff Est.") else "normal",
                color=COLORS["text_dark"],
            )

        ax_p.set_yticks(yp)
        ax_p.set_yticklabels(labels, fontsize=8.2)
        ax_p.invert_yaxis()
        ax_p.set_xlim(0, 70)
        ax_p.set_title(f"{sub_title} · {kwh_wk:.1f} kWh", fontsize=10.5, fontweight="bold", pad=5)
        ax_p.tick_params(axis="x", labelsize=8.0)

    ax_p.set_xlabel("Weekly Cost (£)", fontsize=9.5)

    # ----------------------------------------------------------------------------------------------------------
    # Panel 6 (Row 2, Col 2): Monthly Cost Comparison & Cumulative Savings
    # ----------------------------------------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 2])
    monthly = (
        df.assign(MonthKey=df.index.to_period("M"))
        .groupby("MonthKey")
        .agg(
            Days=("Consumption_kWh", "count"),
            kWh=("Consumption_kWh", "sum"),
            Hist_GBP=("Cost_GBP", "sum"),
            New_GBP=("New_Cost_GBP", "sum"),
            Savings_GBP=("Savings_GBP", "sum"),
        )
        .sort_index()
    )
    monthly["Cum_Savings_GBP"] = monthly["Savings_GBP"].cumsum()
    monthly["Label"] = [
        f"{p.strftime('%b %y')}{'*\n' if row['Days'] < 28 else '\n'}({int(row['Days'])}d)"
        for p, row in monthly.iterrows()
    ]

    x_m = np.arange(len(monthly))
    w_m = 0.35
    bars_h = ax6.bar(
        x_m - w_m / 2,
        monthly["Hist_GBP"],
        width=w_m,
        color=COLORS["historical"],
        alpha=0.88,
        label="Historical Cost (£)",
        zorder=3,
    )
    bars_n = ax6.bar(
        x_m + w_m / 2,
        monthly["New_GBP"],
        width=w_m,
        color=COLORS["new_tariff"],
        alpha=0.90,
        label="New Tariff Cost (£)",
        zorder=3,
    )

    for b_h, b_n, (_, m_row) in zip(bars_h, bars_n, monthly.iterrows()):
        ax6.text(
            b_h.get_x() + b_h.get_width() / 2,
            b_h.get_height() + 3.0,
            f"£{m_row['Hist_GBP']:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=COLORS["historical"],
            fontweight="bold",
        )
        ax6.text(
            b_n.get_x() + b_n.get_width() / 2,
            b_n.get_height() + 3.0,
            f"£{m_row['New_GBP']:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=COLORS["new_tariff"],
            fontweight="bold",
        )
        sav_pct_m = m_row["Savings_GBP"] / m_row["Hist_GBP"] * 100.0
        top_y = max(b_h.get_height(), b_n.get_height()) + 16.0
        ax6.text(
            (b_h.get_x() + b_n.get_x() + b_n.get_width()) / 2,
            top_y,
            f"−£{m_row['Savings_GBP']:.2f} (−{sav_pct_m:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=COLORS["savings"],
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#f0fdf4", "edgecolor": "#bbf7d0", "lw": 0.8},
        )

    ax6.set_xticks(x_m)
    ax6.set_xticklabels(monthly["Label"], fontsize=8.5)
    ax6.set_ylabel("Monthly Cost (£)", fontsize=9.5)
    ax6.set_ylim(0, monthly["Hist_GBP"].max() * 1.33)
    ax6.tick_params(axis="y", labelsize=8.5)
    ax6.set_title("6. Monthly Cost Comparison & Cumulative Savings", fontsize=12, fontweight="bold", pad=14)
    ax6.text(
        0.5,
        1.01,
        "*Partial months (dataset covers 24 Aug – 17 Nov 2026)",
        transform=ax6.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["text_muted"],
    )

    # Cumulative savings line on secondary y-axis
    ax6_cum = ax6.twinx()
    ax6_cum.plot(
        x_m,
        monthly["Cum_Savings_GBP"],
        color=COLORS["savings"],
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=5.5,
        label="Cumulative Savings (£, right)",
        zorder=5,
    )
    ax6_cum.set_ylabel("Cumulative Savings (£)", fontsize=9.5, color=COLORS["savings"])
    ax6_cum.tick_params(axis="y", colors=COLORS["savings"], labelsize=8.5)
    ax6_cum.set_ylim(0, monthly["Cum_Savings_GBP"].max() * 1.48)
    ax6_cum.grid(False)

    # Label cumulative savings at each month point with a crisp white pill badge
    for idx_m, cum_val in enumerate(monthly["Cum_Savings_GBP"]):
        y_off = 4.2 if idx_m != 1 else -5.5
        ax6_cum.text(
            idx_m,
            cum_val + y_off,
            f"£{cum_val:.2f}",
            ha="center",
            va="bottom" if y_off > 0 else "top",
            fontsize=7.6,
            fontweight="bold",
            color=COLORS["savings"],
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": COLORS["savings"], "alpha": 0.94, "lw": 0.8},
            zorder=6,
        )

    h6_1, l6_1 = ax6.get_legend_handles_labels()
    h6_2, l6_2 = ax6_cum.get_legend_handles_labels()
    ax6.legend(
        h6_1 + h6_2,
        l6_1 + l6_2,
        loc="upper left",
        fontsize=7.6,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )

    # Footer disclaimer
    fig.text(
        0.5,
        0.013,
        (
            #"Note: Energy disaggregation and peer benchmarks are EDF Energy Hub estimates (w/c 9 Nov 2026) designed to be directionally accurate. "
            #"CSV daily totals (£41.91 / £42.87) differ slightly (~5%) from EDF weekly breakdown totals (£44.31 / £44.89) due to VAT/estimation timing."
        ),
        ha="center",
        va="center",
        fontsize=8.3,
        color=COLORS["text_muted"],
        style="italic",
    )

    return fig


def main() -> None:
    df = load_and_prepare_data(CSV_FILE)

    # Fit OLS regression exclusively on historical tariff rows before effective_date
    hist_df = df[df.index < NEW_TARIFF["effective_date"]]
    if hist_df.empty:
        hist_df = df

    hist_reg = stats.linregress(hist_df["Consumption_kWh"], hist_df["Cost_GBP"])
    disagg_df, weekly_meta = build_disaggregation_table(df)

    print_console_report(df, hist_reg, disagg_df, weekly_meta)

    fig = build_dashboard(df, hist_reg, disagg_df, weekly_meta)
    fig.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Visualizations successfully saved to '{OUTPUT_IMAGE}'.")

    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()