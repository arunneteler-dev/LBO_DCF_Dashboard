"""
app.py — SME Acquisition Screener (Simplified DCF + LBO)
=========================================================
Run with:  streamlit run app.py

Libraries: streamlit, pandas, numpy, plotly (nothing else).

STRUCTURE
---------
1. CONFIG & GLOBAL CONSTANTS   -> holding period, sensitivity step sizes
2. ASSUMPTIONS DATACLASS       -> single source of truth for every input
3. SIDEBAR                     -> collects user inputs, builds Assumptions
4. FINANCIAL ENGINE            -> run_model(): ALL formulas live here
5. SENSITIVITY ENGINE          -> re-runs run_model() over a grid
6. CHART / TABLE BUILDERS      -> presentation only, no finance logic
7. MAIN                        -> page layout

TO CHANGE A FORMULA: go to run_model() and look for the "# >>> EDIT HERE" markers
(tax, capex, working capital, D&A, margin ramp, debt repayment, exit, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. CONFIG & GLOBAL CONSTANTS
# =============================================================================
st.set_page_config(
    page_title="SME Acquisition Screener",
    page_icon="📊",
    layout="wide",
)

HOLD_YEARS = 5                 # Holding period / forecast horizon (years)
SENS_EXIT_STEP = 1.0           # Sensitivity: step between exit multiples (x EBITDA)
SENS_GROWTH_STEP = 0.02        # Sensitivity: step between growth rates (2 pp)
SENS_POINTS_EACH_SIDE = 2      # Grid = base +/- N steps  -> (2N+1) x (2N+1) table

# Colours used in the capital structure chart
COLOR_BANK = "#1f3b5c"
COLOR_SELLER = "#e0a030"
COLOR_EQUITY = "#2e9e6b"


# =============================================================================
# 2. ASSUMPTIONS DATACLASS
# =============================================================================
@dataclass(frozen=True)
class Assumptions:
    """Every input of the model. Percentages are stored as decimals (0.15 = 15%)."""

    # --- Financials (€) ---
    revenue: float
    ebitda_margin: float

    # --- Valuation ---
    entry_multiple: float
    exit_multiple: float

    # --- Financing structure (% of TOTAL USES = EV + fees) ---
    equity_pct: float
    seller_pct: float
    bank_pct: float            # auto-calculated = 1 - equity - seller

    # --- Cost of debt ---
    bank_rate: float
    seller_rate: float
    amort_years: int           # Bank loan straight-line amortisation period

    # --- Operations (forecast) ---
    growth: float
    target_margin: float       # EBITDA margin reached in the final forecast year

    # --- Advanced / easily adjustable financial assumptions ---
    tax_rate: float            # Corporate tax on (EBIT - interest)
    da_pct: float              # D&A as % of revenue
    capex_pct: float           # Capex as % of revenue
    nwc_pct: float             # Net working capital as % of INCREMENTAL revenue
    fee_pct: float             # Transaction fees as % of EV
    sweep_pct: float           # % of positive post-debt-service FCF used to prepay bank loan
    hurdle_irr: float          # Target IRR used only as a benchmark in the KPI delta


# =============================================================================
# 3. SIDEBAR — ALL USER INPUTS
# =============================================================================
def render_sidebar() -> Assumptions:
    """Draws the sidebar and returns an Assumptions object. No calculations here
    except the bank-loan plug (100% - equity - seller note)."""
    sb = st.sidebar
    sb.title("⚙️ Deal Assumptions")

    # ---- Financials ----
    sb.subheader("Financials")
    revenue_m = sb.number_input("Current Revenue (€M)", min_value=0.5, max_value=100.0,
                                value=5.0, step=0.1, format="%.1f")
    ebitda_margin = sb.slider("Current EBITDA Margin (%)", 1.0, 40.0, 15.0, 0.5, format="%.1f%%")

    # ---- Valuation ----
    sb.subheader("Valuation")
    entry_multiple = sb.slider("Purchase Price Multiple (x EBITDA)", 1.0, 12.0, 4.5, 0.1, format="%.1fx")
    exit_multiple = sb.slider("Exit Multiple (x EBITDA)", 1.0, 12.0, 5.0, 0.1, format="%.1fx")

    # ---- Financing structure ----
    sb.subheader("Financing Structure (% of total uses)")
    equity_pct = sb.slider("Equity Contribution (%)", 10.0, 100.0, 30.0, 1.0, format="%.0f%%")
    seller_pct = sb.slider("Seller Note (%)", 0.0, 50.0, 10.0, 1.0, format="%.0f%%")
    bank_pct = 100.0 - equity_pct - seller_pct          # AUTO-PLUG to reach 100%
    if bank_pct < 0:
        sb.error("Equity + Seller Note exceed 100%. Reduce one of them.")
        st.error("Invalid financing structure: Equity % + Seller Note % must be ≤ 100%.")
        st.stop()
    sb.metric("Bank Loan (auto-calculated)", f"{bank_pct:.0f}%")

    # ---- Cost of debt ----
    sb.subheader("Cost of Debt")
    bank_rate = sb.slider("Bank Loan Interest (%)", 0.0, 15.0, 7.0, 0.25, format="%.2f%%")
    seller_rate = sb.slider("Seller Note Interest (%)", 0.0, 15.0, 5.0, 0.25, format="%.2f%%")
    amort_years = sb.slider("Bank Debt Amortization Period (years)", 1, 15, 7, 1)

    # ---- Operations ----
    sb.subheader("Operations (5-Year Forecast)")
    growth = sb.slider("Annual Revenue Growth (%)", -10.0, 30.0, 5.0, 0.5, format="%.1f%%")
    target_margin = sb.slider("Target EBITDA Margin in Year 5 (%)", 1.0, 50.0, 18.0, 0.5, format="%.1f%%")

    # ---- Advanced assumptions (kept out of the way, but still no hardcoding) ----
    with sb.expander("Advanced assumptions"):
        tax_rate = st.slider("Corporate Tax Rate (%)", 0.0, 45.0, 30.0, 0.5, format="%.1f%%")
        da_pct = st.slider("D&A (% of revenue)", 0.0, 10.0, 2.0, 0.1, format="%.1f%%")
        capex_pct = st.slider("Capex (% of revenue)", 0.0, 15.0, 2.5, 0.1, format="%.1f%%")
        nwc_pct = st.slider("Net Working Capital (% of incremental revenue)", 0.0, 40.0, 10.0, 0.5, format="%.1f%%")
        fee_pct = st.slider("Transaction Fees (% of EV)", 0.0, 10.0, 3.0, 0.25, format="%.2f%%")
        sweep_pct = st.slider("Cash Sweep (% of free cash flow to prepay bank loan)", 0.0, 100.0, 0.0, 5.0, format="%.0f%%")
        hurdle_irr = st.slider("Target IRR Hurdle (%)", 5.0, 50.0, 25.0, 1.0, format="%.0f%%")

    return Assumptions(
        revenue=revenue_m * 1e6,
        ebitda_margin=ebitda_margin / 100,
        entry_multiple=entry_multiple,
        exit_multiple=exit_multiple,
        equity_pct=equity_pct / 100,
        seller_pct=seller_pct / 100,
        bank_pct=bank_pct / 100,
        bank_rate=bank_rate / 100,
        seller_rate=seller_rate / 100,
        amort_years=int(amort_years),
        growth=growth / 100,
        target_margin=target_margin / 100,
        tax_rate=tax_rate / 100,
        da_pct=da_pct / 100,
        capex_pct=capex_pct / 100,
        nwc_pct=nwc_pct / 100,
        fee_pct=fee_pct / 100,
        sweep_pct=sweep_pct / 100,
        hurdle_irr=hurdle_irr / 100,
    )


# =============================================================================
# 4. FINANCIAL ENGINE
# =============================================================================
@dataclass
class ModelResult:
    """Container for everything the UI needs."""
    # Transaction (Year 0)
    ebitda_0: float
    enterprise_value: float
    fees: float
    total_uses: float
    equity_check: float
    bank_0: float
    seller_0: float
    # Yearly schedule (index = Year 1..N)
    schedule: pd.DataFrame
    # Exit & returns
    exit_ev: float
    exit_equity: float
    moic: float
    irr: float


def run_model(a: Assumptions) -> ModelResult:
    """Full LBO + simplified DCF-style cash flow engine.

    All financial logic is in this function. It is pure (no Streamlit calls),
    so it is also reused by the sensitivity grid.
    """
    n = HOLD_YEARS

    # ---------------- Transaction at close (Year 0) ----------------
    ebitda_0 = a.revenue * a.ebitda_margin
    ev = ebitda_0 * a.entry_multiple                 # Enterprise Value
    fees = ev * a.fee_pct                            # >>> EDIT HERE: transaction fees
    total_uses = ev + fees                           # Uses = EV + fees (no cash-to-B/S assumed)

    equity_check = total_uses * a.equity_pct         # Sources: Equity
    seller_0 = total_uses * a.seller_pct             # Sources: Seller note
    bank_0 = total_uses - equity_check - seller_0    # Sources: Bank (plug)

    # Bank loan: straight-line amortisation over `amort_years`
    # >>> EDIT HERE: switch to annuity / balloon structure if desired
    mandatory_amort = bank_0 / a.amort_years

    # ---------------- Yearly projections ----------------
    rows = []
    prev_rev = a.revenue
    bank_bal, seller_bal, cash_bal = bank_0, seller_0, 0.0

    for t in range(1, n + 1):
        # --- Operating forecast ---
        revenue = a.revenue * (1 + a.growth) ** t
        # >>> EDIT HERE: margin path. Currently linear ramp from current to target margin.
        margin = a.ebitda_margin + (a.target_margin - a.ebitda_margin) * t / n
        ebitda = revenue * margin
        da = revenue * a.da_pct                       # >>> EDIT HERE: D&A assumption

        # --- Interest on OPENING balances (avoids circularity) ---
        int_bank = bank_bal * a.bank_rate
        int_seller = seller_bal * a.seller_rate       # Cash-pay interest
        interest = int_bank + int_seller

        # --- Taxes (no loss carry-forward; floor at zero) ---
        # >>> EDIT HERE: tax logic (e.g. NOL carry-forward, interest deductibility cap)
        ebt = ebitda - da - interest
        taxes = max(0.0, ebt) * a.tax_rate

        # --- Capex and working capital ---
        capex = revenue * a.capex_pct                 # >>> EDIT HERE: capex assumption
        d_nwc = (revenue - prev_rev) * a.nwc_pct      # >>> EDIT HERE: working capital assumption

        # --- Cash available for debt service ---
        cfads = ebitda - taxes - capex - d_nwc        # Cash Flow Available for Debt Service

        # --- Debt repayment ---
        # Bank: mandatory straight-line amortisation during the amortisation period.
        # Seller note: bullet, repaid at exit out of proceeds (cash-pay interest meanwhile).
        # >>> EDIT HERE: seller note amortisation, if any
        mandatory = min(bank_bal, mandatory_amort) if t <= a.amort_years else 0.0

        # Free Cash Flow to equity after interest and mandatory principal
        fcf = cfads - interest - mandatory

        # Optional cash sweep: prepay bank loan with a % of positive FCF
        sweep = min(bank_bal - mandatory, max(0.0, fcf) * a.sweep_pct)

        # --- Roll balances forward ---
        bank_bal = bank_bal - mandatory - sweep
        cash_bal = cash_bal + fcf - sweep             # Retained cash accrues to equity (no dividends)

        # Debt service coverage ratio
        debt_service = interest + mandatory
        dscr = cfads / debt_service if debt_service > 0 else np.nan

        # Implied EV at the exit multiple (used for the capital structure chart)
        implied_ev = ebitda * a.exit_multiple
        implied_equity = max(implied_ev - bank_bal - seller_bal + cash_bal, 0.0)

        rows.append({
            "Year": t,
            "Revenue": revenue,
            "EBITDA Margin": margin,
            "EBITDA": ebitda,
            "D&A": da,
            "Interest": interest,
            "Taxes": taxes,
            "Capex": capex,
            "Change in NWC": d_nwc,
            "CFADS": cfads,
            "Mandatory Principal": mandatory,
            "Cash Sweep": sweep,
            "Principal Repayment": mandatory + sweep,
            "Free Cash Flow": fcf,
            "DSCR": dscr,
            "Bank Loan Balance": bank_bal,
            "Seller Note Balance": seller_bal,
            "Cash Balance": cash_bal,
            "Implied EV": implied_ev,
            "Equity Value": implied_equity,
        })
        prev_rev = revenue

    schedule = pd.DataFrame(rows).set_index("Year")

    # ---------------- Exit & returns ----------------
    # >>> EDIT HERE: exit costs, management incentive plan, etc.
    ebitda_exit = schedule.loc[n, "EBITDA"]
    exit_ev = ebitda_exit * a.exit_multiple
    exit_equity = exit_ev - bank_bal - seller_bal + cash_bal   # net of all debt, plus cash

    moic = exit_equity / equity_check if equity_check > 0 else np.nan
    # No interim dividends -> IRR has a closed form (CAGR of equity value)
    irr = moic ** (1 / n) - 1 if (moic is not None and moic > 0) else -1.0

    return ModelResult(
        ebitda_0=ebitda_0, enterprise_value=ev, fees=fees, total_uses=total_uses,
        equity_check=equity_check, bank_0=bank_0, seller_0=seller_0,
        schedule=schedule, exit_ev=exit_ev, exit_equity=exit_equity,
        moic=moic, irr=irr,
    )


# =============================================================================
# 5. SENSITIVITY ENGINE
# =============================================================================
def build_irr_sensitivity(a: Assumptions) -> pd.DataFrame:
    """IRR grid: rows = revenue growth, columns = exit multiple.
    Re-runs the full model for every cell (the model is fast)."""
    k = SENS_POINTS_EACH_SIDE
    exit_multiples = [a.exit_multiple + SENS_EXIT_STEP * i for i in range(-k, k + 1)]
    growth_rates = [a.growth + SENS_GROWTH_STEP * i for i in range(-k, k + 1)]

    grid = np.array([
        [run_model(replace(a, exit_multiple=max(m, 0.1), growth=g)).irr for m in exit_multiples]
        for g in growth_rates
    ])
    return pd.DataFrame(
        grid,
        index=[f"{g:.1%}" for g in growth_rates],
        columns=[f"{m:.1f}x" for m in exit_multiples],
    )


# =============================================================================
# 6. CHART / TABLE BUILDERS (presentation only)
# =============================================================================
def eur_m(v: float) -> str:
    """Format € value in millions."""
    return f"€{v / 1e6:,.2f}M"


def accounting(v: float) -> str:
    """Accounting-style number format: negatives in parentheses."""
    if pd.isna(v):
        return "–"
    return f"({abs(v):,.0f})" if v < 0 else f"{v:,.0f}"


def build_capital_structure_chart(res: ModelResult) -> go.Figure:
    """Stacked bar: Senior Debt / Seller Note / Equity Value at close and each year-end.
    Year 0 shows the capital invested (cost basis). Years 1-N show equity value implied
    by the exit multiple: EBITDA x exit multiple - debt + cash."""
    s = res.schedule
    labels = ["Close"] + [f"Year {y}" for y in s.index]
    bank = [res.bank_0] + s["Bank Loan Balance"].tolist()
    seller = [res.seller_0] + s["Seller Note Balance"].tolist()
    equity = [res.equity_check] + s["Equity Value"].tolist()

    fig = go.Figure()
    for name, values, color in [
        ("Senior Debt (Bank)", bank, COLOR_BANK),
        ("Seller Note", seller, COLOR_SELLER),
        ("Equity Value", equity, COLOR_EQUITY),
    ]:
        fig.add_trace(go.Bar(
            x=labels, y=[v / 1e6 for v in values], name=name, marker_color=color,
            hovertemplate=f"{name}: €%{{y:,.2f}}M<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        yaxis_title="€ millions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        hovermode="x unified",
    )
    return fig


def build_cash_flow_table(res: ModelResult) -> pd.DataFrame:
    """Main cash flow table (€ thousands). Metrics as rows, years as columns."""
    cols = ["Revenue", "EBITDA", "Interest", "Principal Repayment", "Free Cash Flow"]
    df = res.schedule[cols].copy() / 1e3
    df.index = [f"Year {y}" for y in df.index]
    df.columns = [f"{c} (€k)" for c in df.columns]
    return df.T


def _red_yellow_green(t: float) -> str:
    """Map t in [0,1] to a red -> yellow -> green CSS background (no matplotlib needed)."""
    red, yellow, green = np.array([248, 105, 107]), np.array([255, 235, 132]), np.array([99, 190, 123])
    rgb = red + (yellow - red) * (t / 0.5) if t < 0.5 else yellow + (green - yellow) * ((t - 0.5) / 0.5)
    r, g, b = [int(round(x)) for x in rgb]
    return f"background-color: rgb({r},{g},{b}); color: #1a1a1a;"


def style_sensitivity(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Pandas Styler with a colour gradient (red->green) and the base case highlighted.
    The gradient is implemented manually so matplotlib is NOT required."""
    vmin, vmax = float(np.nanmin(df.values)), float(np.nanmax(df.values))
    span = (vmax - vmin) or 1.0

    def gradient(data: pd.DataFrame) -> pd.DataFrame:
        return data.applymap(lambda v: _red_yellow_green((v - vmin) / span)) \
            if hasattr(data, "applymap") else data.map(lambda v: _red_yellow_green((v - vmin) / span))

    mid = SENS_POINTS_EACH_SIDE   # base case sits in the centre of the grid

    def highlight_base(data: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=data.index, columns=data.columns)
        css.iloc[mid, mid] = "font-weight: 800; border: 3px solid #111;"
        return css

    return (
        df.style
        .apply(gradient, axis=None)
        .apply(highlight_base, axis=None)
        .format("{:.1%}")
    )


# =============================================================================
# 7. MAIN PAGE
# =============================================================================
def main() -> None:
    a = render_sidebar()
    res = run_model(a)
    s = res.schedule

    st.title("📊 SME Acquisition Screener")
    st.caption("Simplified DCF + LBO model for search fund / ETA deal screening. "
               "All inputs are in the sidebar; results update instantly.")

    # ---------- 1. TOP-LEVEL KPIs ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Enterprise Value", eur_m(res.enterprise_value),
              help=f"Entry multiple {a.entry_multiple:.1f}x × current EBITDA {eur_m(res.ebitda_0)}")
    k2.metric("Required Equity Check", eur_m(res.equity_check),
              delta=f"{a.equity_pct:.0%} of total uses", delta_color="off",
              help="Equity contribution % × (EV + transaction fees)")
    k3.metric(f"{HOLD_YEARS}-Year IRR", f"{res.irr:.1%}",
              delta=f"{(res.irr - a.hurdle_irr) * 100:+.1f} pp vs {a.hurdle_irr:.0%} hurdle")
    k4.metric("Cash-on-Cash Return (MOIC)", f"{res.moic:.2f}x",
              help="Exit equity proceeds ÷ initial equity check")

    # Secondary KPIs for deal viability
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Total Uses (EV + fees)", eur_m(res.total_uses))
    k6.metric("Bank Loan / Seller Note", f"{eur_m(res.bank_0)} / {eur_m(res.seller_0)}")
    k7.metric("Total Debt / EBITDA at close", f"{(res.bank_0 + res.seller_0) / res.ebitda_0:.1f}x")
    k8.metric("Minimum DSCR", f"{np.nanmin(s['DSCR'].values):.2f}x",
              help="CFADS ÷ (interest + mandatory principal). Banks typically want > 1.3x.")

    # Warnings that matter for viability
    if (s["Cash Balance"] < 0).any():
        st.warning("⚠️ Cumulative cash turns negative: the business cannot fully service its debt. "
                   "A revolver or a lower leverage / longer amortisation would be required.")
    if res.exit_equity <= 0:
        st.error("Exit equity value is zero or negative: equity is wiped out under these assumptions.")

    st.divider()

    # ---------- 2. CAPITAL STRUCTURE EVOLUTION ----------
    st.subheader("Capital Structure Evolution & Debt Paydown")
    st.plotly_chart(build_capital_structure_chart(res))
    st.caption("Close = capital invested (cost basis). Years 1–5: Equity Value = EBITDA × exit multiple "
               "− remaining debt + accumulated cash.")

    # ---------- 3. CASH FLOW TABLE ----------
    st.subheader("Cash Flow Summary (Years 1–5)")
    cf_table = build_cash_flow_table(res)
    st.dataframe(cf_table.style.format(accounting))

    with st.expander("Detailed cash flow bridge & debt schedule"):
        detail_cols = ["Revenue", "EBITDA Margin", "EBITDA", "D&A", "Interest", "Taxes", "Capex",
                       "Change in NWC", "CFADS", "Mandatory Principal", "Cash Sweep",
                       "Free Cash Flow", "DSCR", "Bank Loan Balance", "Seller Note Balance",
                       "Cash Balance"]
        detail = s[detail_cols].copy()
        detail.index = [f"Year {y}" for y in detail.index]
        detail = detail.T
        fmt = {}
        for row in detail.index:
            if row == "EBITDA Margin":
                fmt[row] = "{:.1%}"
            elif row == "DSCR":
                fmt[row] = "{:.2f}x"
            else:
                fmt[row] = accounting
        # Format each row separately (rows have different units)
        styled = detail.style
        for row, f in fmt.items():
            styled = styled.format(f, subset=pd.IndexSlice[row, :])
        st.dataframe(styled)
        st.caption("Amounts in €. CFADS = EBITDA − taxes − capex − ΔNWC. "
                   "Free Cash Flow = CFADS − interest − mandatory principal.")

    with st.expander("Sources & Uses and exit waterfall"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Uses**")
            st.write(pd.DataFrame({"€": [res.enterprise_value, res.fees, res.total_uses]},
                                  index=["Enterprise Value", "Transaction Fees", "Total Uses"])
                     .style.format("{:,.0f}"))
        with c2:
            st.markdown("**Sources**")
            st.write(pd.DataFrame(
                {"€": [res.equity_check, res.seller_0, res.bank_0, res.total_uses],
                 "% of total": [a.equity_pct, a.seller_pct, a.bank_pct, 1.0]},
                index=["Equity", "Seller Note", "Bank Loan", "Total Sources"])
                .style.format({"€": "{:,.0f}", "% of total": "{:.0%}"}))
        with c3:
            st.markdown("**Exit (Year 5)**")
            st.write(pd.DataFrame(
                {"€": [res.exit_ev, -s.iloc[-1]["Bank Loan Balance"], -s.iloc[-1]["Seller Note Balance"],
                       s.iloc[-1]["Cash Balance"], res.exit_equity]},
                index=["Exit Enterprise Value", "– Bank Loan", "– Seller Note", "+ Cash", "Exit Equity"])
                .style.format("{:,.0f}"))

    st.divider()

    # ---------- 4. SENSITIVITY ANALYSIS ----------
    st.subheader(f"{HOLD_YEARS}-Year IRR Sensitivity")
    st.caption("Columns: Exit Multiple (x EBITDA) · Rows: Annual Revenue Growth. "
               "The outlined cell is the current base case.")
    sens = build_irr_sensitivity(a)
    sens.index.name = "Revenue Growth ↓ / Exit Multiple →"
    st.dataframe(style_sensitivity(sens))

    with st.expander("Model methodology & where to edit assumptions"):
        st.markdown(
            """
- **Entry:** EV = current EBITDA × purchase multiple. Total uses = EV + transaction fees.
- **Financing:** Equity %, Seller Note % and Bank Loan % are percentages of total uses; the Bank Loan is the plug to 100%.
- **Forecast:** Revenue grows at a constant rate; EBITDA margin ramps linearly to the target margin by Year 5.
- **Cash flow:** EBITDA − taxes − capex − ΔNWC − interest − mandatory principal. Interest is computed on opening balances.
- **Debt:** Bank loan amortises straight-line over the selected period (optional cash sweep). The seller note is a bullet repaid at exit with cash-pay interest.
- **Exit:** Exit EV = Year 5 EBITDA × exit multiple; equity proceeds = exit EV − remaining debt + accumulated cash.
- **Returns:** MOIC = exit equity ÷ equity check; IRR = MOIC^(1/5) − 1 (no interim dividends).
- **To modify formulas:** edit `run_model()` — search for the `# >>> EDIT HERE` comments.
            """
        )


if __name__ == "__main__":
    main()
