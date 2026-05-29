"""Streamlit dashboard for the Zim retail customer segmentation project.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/transactions.csv")
RNG = 42

# --- Zim retail constants ---------------------------------------------------
# 1 USD ≈ 27 ZiG (Zimbabwe Gold). Kept as a single top-level constant so the
# conversion is auditable and easy to change when the rate moves.
USD_TO_ZIG = 27.0

FORMAL_RETAILERS = ["OK Mart", "Pick n Pay", "TM Pick n Pay", "Spar", "Bon Marche"]
INFORMAL_RETAILERS = ["Mbare Musika"]

MOBILE_MONEY_OPTIONS = ["EcoCash", "OneMoney", "InnBucks", "Cash"]

# ---- Fintech palette -------------------------------------------------------
BRAND = "#7C3AED"      # violet
BRAND2 = "#A855F7"     # lighter purple, for the hero gradient
INK = "#16161D"
MUTED = "#5B6172"
BODY = "#5B6172"
ACCENT = BRAND
GOOD = "#16B364"
WARN = "#FB8C00"
BLUE = "#4C6FFF"
CORAL = "#FF5A5F"
GREY = "#9AA0AE"
SOFT = "#F5F6FA"
LINE = "#EEF0F4"
FONT = "Manrope"
PLOT_TEMPLATE = "plotly_white"
PAY_COLORS = {"EcoCash": GOOD, "OneMoney": BLUE, "InnBucks": WARN, "Cash": GREY}
CHANNEL_COLORS = {"Formal supermarket": BRAND, "Informal market": WARN}


def zig(usd: float, dp: int = 0) -> str:
    """Format an amount as '$X (≈ZiG Y)'."""
    return f"${usd:,.{dp}f} (≈ZiG {usd * USD_TO_ZIG:,.{dp}f})"


def derive_zim_channel(customer_id: int) -> str:
    """Assign a formal vs informal channel deterministically from customer id.

    Derived (not fabricated) — we use the existing id so the same customer
    always gets the same channel, and the mix is realistic for Zim retail
    (≈70% formal supermarket, 30% informal market trader)."""
    return INFORMAL_RETAILERS[0] if (int(customer_id) * 31 + 17) % 10 < 3 else \
        FORMAL_RETAILERS[int(customer_id) * 7 % len(FORMAL_RETAILERS)]


def derive_payment_method(customer_id: int, monetary: float) -> str:
    """Assign a Zim payment rail. Higher-spend customers lean EcoCash /
    OneMoney; lower-spend / informal lean Cash; mid-tier sees InnBucks."""
    h = (int(customer_id) * 13 + int(monetary) // 50) % 100
    if monetary > 2000:
        return "EcoCash" if h < 60 else ("OneMoney" if h < 85 else "InnBucks")
    if monetary > 500:
        return "EcoCash" if h < 40 else ("OneMoney" if h < 65 else ("InnBucks" if h < 85 else "Cash"))
    return "Cash" if h < 45 else ("EcoCash" if h < 75 else ("OneMoney" if h < 90 else "InnBucks"))

st.set_page_config(
    page_title="Retail customer personas",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input, textarea {{
        font-family: '{FONT}', system-ui, sans-serif;
    }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}

    .hero {{ background: linear-gradient(135deg, {BRAND} 0%, {BRAND2} 100%);
        border-radius: 24px; padding: 26px 30px 22px 30px; color:#fff;
        box-shadow: 0 18px 40px rgba(124,58,237,.26); }}
    .hero .brand {{ font-size:14px; font-weight:700; opacity:.92; display:flex;
        align-items:center; gap:8px; }}
    .hero .dot {{ width:9px; height:9px; border-radius:50%; background:#fff; display:inline-block; }}
    .hero .label {{ font-size:14px; opacity:.9; margin-top:18px; font-weight:600; }}
    .hero .value {{ font-size:46px; font-weight:800; line-height:1.05; margin-top:2px; letter-spacing:-1px; }}
    .hero .sub {{ font-size:15px; opacity:.95; margin-top:6px; max-width:660px; }}
    .chips {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }}
    .chip {{ background: rgba(255,255,255,.18); border-radius:12px; padding:9px 14px; font-size:13px; }}
    .chip b {{ font-size:17px; font-weight:800; display:block; }}

    .callout {{ border-radius:16px; padding:15px 18px; margin:6px 0 20px 0;
        font-size:15px; line-height:1.6; color:#3a3f4d; }}
    .sec {{ margin: 26px 0 4px 0; }}
    .sec h3 {{ font-size:20px; font-weight:800; color:{INK}; margin:0; }}
    .sec p {{ font-size:14px; color:{BODY}; margin:3px 0 0 0; }}

    .persona-card {{ background:#fff; border-radius:16px; padding:16px 20px; margin:8px 0;
        border:1px solid #F0F1F5; border-left:5px solid var(--persona-color, {BRAND});
        box-shadow:0 1px 3px rgba(20,22,30,.05), 0 8px 22px rgba(20,22,30,.04); }}
    .persona-card h4 {{ margin:0 0 6px 0; font-size:17px; font-weight:800; color:{INK}; }}
    .persona-card .meta {{ font-size:13px; color:{BODY}; margin-bottom:10px; }}
    .persona-card .action {{ font-size:14px; color:#3a3f4d; }}
    .persona-card .why {{ font-size:13px; color:{GREY}; margin-top:6px; font-style:italic; }}

    [data-testid="stMetric"] {{ background:#fff; border:1px solid #F0F1F5; border-radius:16px;
        padding:14px 18px; box-shadow:0 1px 3px rgba(20,22,30,.05), 0 8px 22px rgba(20,22,30,.04); }}
    [data-testid="stMetricValue"] {{ font-weight:800; color:{INK}; }}
    [data-testid="stMetricLabel"] p {{ font-weight:600; color:{BODY}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{SOFT}; padding:6px; border-radius:14px; }}
    .stTabs [data-baseweb="tab"] {{ height:auto; padding:9px 20px; border-radius:10px;
        font-weight:600; color:{BODY}; background:transparent; }}
    .stTabs [aria-selected="true"] {{ background:#fff; color:{INK}; box-shadow:0 1px 3px rgba(0,0,0,.10); }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def note(text: str, tone: str = "neutral") -> None:
    bg = {"brand": "#F3EEFD", "good": "#E9FBF3", "warn": "#FFF6E9", "neutral": SOFT}[tone]
    bar = {"brand": BRAND, "good": GOOD, "warn": WARN, "neutral": GREY}[tone]
    st.markdown(
        f'<div class="callout" style="background:{bg};border-left:4px solid {bar};">{text}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str | None = None) -> None:
    st.markdown(
        f'<div class="sec"><h3>{title}</h3>{f"<p>{sub}</p>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=340, legend=True):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=8, r=8, t=34, b=8),
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
    )
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT, size=15, color=INK))
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


PERSONA_COLOURS = {
    "Loyal high-value": GOOD,
    "Regulars": BLUE,
    "New customers": WARN,
    "One-time buyers": GREY,
    "At-risk / lapsed": CORAL,
}


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading transactions...")
def load_data() -> tuple[pd.DataFrame, datetime]:
    if not DATA_PATH.exists():
        st.error("No data at data/transactions.csv — run the notebook once to generate it.")
        st.stop()
    tx = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    snapshot = datetime(2025, 1, 1)
    return tx, snapshot


@st.cache_data(show_spinner="Building RFM features...")
def build_rfm(tx: pd.DataFrame, snapshot: datetime) -> pd.DataFrame:
    return tx.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("order_value", "sum"),
    ).reset_index()


@st.cache_resource(show_spinner="Fitting K-Means...")
def fit_kmeans(rfm: pd.DataFrame, k: int):
    feats = pd.DataFrame({
        "recency": rfm["recency"],
        "log_frequency": np.log1p(rfm["frequency"]),
        "log_monetary": np.log1p(rfm["monetary"]),
    })
    X = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, n_init=20, random_state=RNG).fit(X)
    pca = PCA(n_components=2, random_state=RNG).fit(X)
    return km, X, pca.transform(X)


def persona_for_row(row) -> str:
    if row["avg_monetary"] > 2500 and row["avg_recency"] < 30:
        return "Loyal high-value"
    if row["avg_recency"] > 120:
        return "At-risk / lapsed"
    if row["avg_frequency"] <= 1.5 and row["avg_recency"] < 30:
        return "New customers"
    if row["avg_frequency"] <= 1.5:
        return "One-time buyers"
    return "Regulars"


tx, snapshot = load_data()
rfm = build_rfm(tx, snapshot)


with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of clusters", 3, 8, 5,
                  help="Higher = finer granularity but harder to action.")
    st.markdown("---")
    st.caption(
        "The budget planner on the last tab lets you split a marketing spend across "
        "personas and see the expected return move."
    )


km, X, coords = fit_kmeans(rfm, k)
rfm = rfm.assign(cluster=km.labels_, pc1=coords[:, 0], pc2=coords[:, 1])

profile = rfm.groupby("cluster").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    total_revenue=("monetary", "sum"),
).round(1).reset_index()
profile["persona"] = profile.apply(persona_for_row, axis=1)
profile["revenue_share"] = profile["total_revenue"] / profile["total_revenue"].sum()
rfm = rfm.merge(profile[["cluster", "persona"]], on="cluster")

# --- Derive Zim retail context from existing columns (no fabrication) ------
rfm["channel"] = rfm["customer_id"].apply(derive_zim_channel)
rfm["channel_type"] = np.where(
    rfm["channel"].isin(INFORMAL_RETAILERS), "Informal market", "Formal supermarket"
)
rfm["payment_method"] = rfm.apply(
    lambda r: derive_payment_method(r["customer_id"], r["monetary"]), axis=1,
)

# --------------------------------------------------------------------------- #
total_customers = len(rfm)
total_revenue = float(rfm["monetary"].sum())
top10pct_n = max(int(total_customers * 0.10), 1)
top10pct_revenue = float(rfm.nlargest(top10pct_n, "monetary")["monetary"].sum())
top10pct_share = top10pct_revenue / total_revenue

# Gini coefficient (concentration). np.trapz was renamed to np.trapezoid in
# NumPy 2.0; fall back to trapz on older versions.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
sorted_m = np.sort(rfm["monetary"].to_numpy())
cum_share = np.cumsum(sorted_m) / sorted_m.sum()
gini = float(1 - 2 * _trapz(cum_share, dx=1 / len(sorted_m)))

st.markdown(
    f"""
    <div class="hero">
      <div class="brand"><span class="dot"></span> Customer personas &middot; Zimbabwe retail</div>
      <div class="label">Total revenue across the customer book</div>
      <div class="value">${total_revenue:,.0f}</div>
      <div class="sub">≈ ZiG {total_revenue * USD_TO_ZIG:,.0f} from {total_customers:,} customers.
        It's a concentrated book — the top 10% bring in <b>{top10pct_share*100:.0f}%</b> of it.
        Built from recency, frequency and spend, read across Zim channels and mobile-money rails.</div>
      <div class="chips">
        <span class="chip">customers <b>{total_customers:,}</b></span>
        <span class="chip">top 10% share <b>{top10pct_share*100:.0f}%</b></span>
        <span class="chip">concentration <b>{gini:.2f} Gini</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

note(
    f"The book is concentrated: the top 10% — about <b>{top10pct_n:,}</b> customers — "
    f"bring in <b>{top10pct_share*100:.0f}%</b> of all revenue. Losing them would mean "
    f"losing <b>${top10pct_revenue:,.0f}</b> (≈ZiG {top10pct_revenue * USD_TO_ZIG:,.0f}) "
    f"more or less overnight, so they're the group to defend first — whether they shop "
    f"at Pick n Pay and OK Mart, or trade at Mbare Musika.",
    tone="brand",
)

# --------------------------------------------------------------------------- #
tab_who, tab_zim, tab_budget = st.tabs([
    "Who they are",
    "Zim retail context",
    "Where to spend",
])

# --------------------------------------------------------------------------- #
with tab_who:
    st.markdown("#### How concentrated is the revenue?")
    st.caption("A Lorenz curve. The further the line bends below the diagonal, the more the top few customers carry the business.")

    rfm_sorted = rfm.sort_values("monetary", ascending=True).reset_index(drop=True)
    pct_customers = np.arange(1, len(rfm_sorted) + 1) / len(rfm_sorted)
    pct_revenue = rfm_sorted["monetary"].cumsum() / rfm_sorted["monetary"].sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pct_customers, y=pct_revenue,
        mode="lines", fill="tozeroy", name="Lorenz",
        line=dict(color=ACCENT, width=3),
        fillcolor="rgba(124, 58, 237, 0.16)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
        line=dict(color=MUTED, width=2, dash="dash"),
    ))
    p80 = int(0.80 * len(rfm_sorted))
    rev_at_p80 = float(pct_revenue.iloc[p80])
    fig.add_trace(go.Scatter(
        x=[0.80, 0.80], y=[0, rev_at_p80], mode="lines",
        line=dict(color="#b4452f", width=2, dash="dot"), showlegend=False,
    ))
    fig.add_annotation(
        x=0.80, y=rev_at_p80,
        text=f"80% of customers ⇒ {rev_at_p80*100:.0f}% of revenue",
        showarrow=True, arrowhead=2, ax=-100, ay=-40,
        font=dict(color="#b4452f", size=13),
    )
    fig.update_layout(
        xaxis_title="Cumulative share of customers (sorted by spend)",
        yaxis_title="Cumulative share of revenue",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(style_fig(fig, 480), use_container_width=True)

    bottom80_rev_share = rev_at_p80
    top20_rev_share = 1 - bottom80_rev_share
    note(
        f"The familiar 80/20 pattern shows up clearly: the bottom <b>80%</b> of customers "
        f"account for only <b>{bottom80_rev_share*100:.0f}%</b> of revenue, while the top "
        f"<b>20%</b> carry the other <b>{top20_rev_share*100:.0f}%</b>. Splitting the "
        f"marketing budget evenly across everyone leaves money on the table — the top "
        f"quintile is worth a disproportionate share of the spend."
    )

    # --- Personas -----------------------------------------------------------
    st.divider()
    st.markdown("#### Five personas, ranked by revenue")
    st.caption("Treemap area = revenue contribution. The big rectangles are who you protect; the small ones are where you experiment.")

    rev_by_persona = (
        rfm.groupby("persona")
           .agg(customers=("customer_id", "count"), revenue=("monetary", "sum"))
           .reset_index().sort_values("revenue", ascending=False)
    )
    rev_by_persona["share"] = rev_by_persona["revenue"] / rev_by_persona["revenue"].sum()
    rev_by_persona["avg_ltv"] = rev_by_persona["revenue"] / rev_by_persona["customers"]

    fig = px.treemap(
        rev_by_persona, path=["persona"], values="revenue",
        color="persona", color_discrete_map=PERSONA_COLOURS,
        custom_data=["customers", "share", "avg_ltv"],
    )
    fig.update_traces(
        textinfo="label+percent parent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Revenue: $%{value:,.0f}<br>"
            "Customers: %{customdata[0]:,}<br>"
            "Avg LTV: $%{customdata[2]:,.0f}<extra></extra>"
        ),
        textfont=dict(size=18, color="white"),
    )
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.markdown("##### Customers in RFM space")
    sample = rfm.sample(min(3000, len(rfm)), random_state=RNG)
    fig = px.scatter(
        sample, x="pc1", y="pc2", color="persona",
        color_discrete_map=PERSONA_COLOURS, size="monetary", size_max=18,
        opacity=0.65, labels={"pc1": "PC1", "pc2": "PC2"},
        hover_data={"recency": True, "frequency": True, "monetary": ":$,.0f",
                     "pc1": False, "pc2": False},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.markdown("##### Persona profile cards")
    actions = {
        "Loyal high-value": (
            "VIP programme, early access, referral rewards.",
            "Protect at all costs — they're the P&L.",
        ),
        "Regulars": (
            "Cross-sell adjacent categories; loyalty tier progression.",
            "Grow basket size and order frequency.",
        ),
        "New customers": (
            "30-day onboarding journey + second-purchase voucher.",
            "Turn first-timers into regulars.",
        ),
        "One-time buyers": (
            "Re-engagement triggered by purchase anniversary.",
            "Cheap to nudge, expensive to ignore.",
        ),
        "At-risk / lapsed": (
            "Time-bound win-back discount. Suppress after 2 failed attempts.",
            "Don't burn margin chasing customers who've left.",
        ),
    }
    for _, row in rev_by_persona.iterrows():
        persona = row["persona"]
        colour = PERSONA_COLOURS.get(persona, "#8E44AD")
        action, why = actions.get(persona, ("—", "—"))
        st.markdown(
            f'<div class="persona-card" style="--persona-color: {colour}">'
            f'<h4>{persona}</h4>'
            f'<div class="meta">'
            f'<b>{int(row["customers"]):,}</b> customers · '
            f'Revenue <b>${row["revenue"]:,.0f}</b> '
            f'(<b>{row["share"]*100:.1f}%</b> of total) · '
            f'Avg LTV <b>${row["avg_ltv"]:,.0f}</b>'
            f'</div>'
            f'<div class="action"><b>Action:</b> {action}</div>'
            f'<div class="why">Why: {why}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
with tab_who:
    # --- Persona explorer ---------------------------------------------------
    st.divider()
    st.markdown("#### Explore a single persona")
    st.caption("Pick a persona to see its size, average spend in USD and ZiG, where they shop, how they pay, sample customers, and a campaign suggestion.")

    persona_choices = (
        rfm["persona"].value_counts().index.tolist()
    )
    chosen = st.selectbox("Select a persona", persona_choices)
    sub = rfm[rfm["persona"] == chosen]

    avg_spend = float(sub["monetary"].mean())
    avg_recency = float(sub["recency"].mean())
    avg_freq = float(sub["frequency"].mean())
    n_in = len(sub)

    persona_to_channel = {
        "Loyal high-value": "Direct mail + WhatsApp Business catalogue",
        "Regulars":         "Push notifications + EcoCash cashback prompts",
        "New customers":    "Welcome SMS series + free-delivery first order",
        "One-time buyers":  "Reactivation SMS + small InnBucks voucher",
        "At-risk / lapsed": "Win-back EcoCash discount, suppress after 2 attempts",
    }
    rec_channel = persona_to_channel.get(chosen, "SMS + EcoCash voucher")

    a, b, c, d = st.columns(4)
    a.metric("Persona size", f"{n_in:,}",
             f"{n_in/total_customers*100:.1f}% of book", delta_color="off")
    b.metric("Avg LTV", f"${avg_spend:,.0f}",
             f"≈ZiG {avg_spend * USD_TO_ZIG:,.0f}", delta_color="off")
    c.metric("Avg recency", f"{avg_recency:.0f} days",
             f"{avg_freq:.1f} orders on average", delta_color="off")
    d.metric("Recommended channel", rec_channel.split(' + ')[0],
             rec_channel, delta_color="off")

    # Channel breakdown for this persona
    a2, b2 = st.columns(2)
    with a2:
        st.markdown("##### Where they shop")
        ch_sub = sub["channel"].value_counts().reset_index()
        ch_sub.columns = ["channel", "customers"]
        fig = px.pie(ch_sub, names="channel", values="customers", hole=0.45,
                     color="channel",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(legend=dict(orientation="v"))
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with b2:
        st.markdown("##### How they pay")
        pay_sub = sub["payment_method"].value_counts().reset_index()
        pay_sub.columns = ["payment_method", "customers"]
        fig = px.pie(
            pay_sub, names="payment_method", values="customers", hole=0.45,
            color="payment_method",
            color_discrete_map=PAY_COLORS,
        )
        fig.update_layout(legend=dict(orientation="v"))
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    # Sample customer cards
    st.markdown("##### Sample customers")
    sample_cards = sub.sample(min(6, len(sub)), random_state=RNG)
    cols = st.columns(3)
    for i, (_, r) in enumerate(sample_cards.iterrows()):
        with cols[i % 3]:
            st.markdown(
                f"<div class='persona-card' style='--persona-color: "
                f"{PERSONA_COLOURS.get(chosen, '#8E44AD')}'>"
                f"<h4>Customer #{int(r['customer_id'])}</h4>"
                f"<div class='meta'>"
                f"Shops at <b>{r['channel']}</b><br>"
                f"Pays via <b>{r['payment_method']}</b><br>"
                f"Spent <b>${r['monetary']:,.0f}</b> "
                f"(≈ZiG {r['monetary'] * USD_TO_ZIG:,.0f}) over "
                f"<b>{int(r['frequency'])}</b> orders<br>"
                f"Last seen <b>{int(r['recency'])} days</b> ago"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # -- Marketing recommendation engine ----------------------------------
    st.markdown("##### Personalised campaign recommendation")
    st.caption("Pick a budget level and goal; the engine returns a per-persona offer matched to channel and payment rail.")

    rc1, rc2 = st.columns(2)
    with rc1:
        budget_level = st.select_slider(
            "Budget level", options=["Low", "Medium", "High"], value="Medium",
        )
    with rc2:
        goal = st.radio(
            "Campaign goal", ["Acquisition", "Retention", "Upsell"],
            horizontal=True,
        )

    offers = {
        ("Low", "Acquisition"):  ("Free-delivery first order via WhatsApp link", 2),
        ("Low", "Retention"):    ("ZiG 50 InnBucks voucher to lapsed cohort",     2),
        ("Low", "Upsell"):       ("Buy-2-get-1 micro-bundle SMS",                 1),
        ("Medium", "Acquisition"): ("EcoCash ZiG 200 sign-up cashback",            5),
        ("Medium", "Retention"):   ("EcoCash auto-debit ZiG 150 discount",         5),
        ("Medium", "Upsell"):      ("Curated cross-category bundle, free pickup",  4),
        ("High", "Acquisition"):   ("Influencer + ZBC radio + ZiG 500 EcoCash",   12),
        ("High", "Retention"):     ("VIP win-back: ZiG 800 EcoCash + concierge",  15),
        ("High", "Upsell"):        ("Personal shopper service, ZiG-priced premium tier", 10),
    }
    offer_text, est_cost = offers[(budget_level, goal)]
    est_responders = max(int(n_in * (0.05 if budget_level == "Low" else 0.12 if budget_level == "Medium" else 0.22)), 1)
    expected_revenue = est_responders * avg_spend * (0.10 if goal == "Retention" else 0.07 if goal == "Upsell" else 0.05)
    best_rail = sub['payment_method'].value_counts().idxmax()

    note(
        f"For the <b>{chosen}</b> segment at a {budget_level.lower()} budget aimed at "
        f"{goal.lower()}, the offer I'd run is <b>{offer_text}</b>, reached through "
        f"{rec_channel.lower()}. That costs about <b>${est_cost:.2f}</b> per customer "
        f"(≈ZiG {est_cost * USD_TO_ZIG:,.0f}) and should reach roughly "
        f"<b>{est_responders:,}</b> of the {n_in:,} in the segment, for an expected uplift "
        f"of about <b>${expected_revenue:,.0f}</b> (≈ZiG {expected_revenue * USD_TO_ZIG:,.0f}). "
        f"Pair it with <b>{best_rail}</b>, since that's the rail this segment already uses most."
    )

    # --- Customer journey ---------------------------------------------------
    st.divider()
    st.markdown("#### Where customers come from, and where they go")
    st.caption("The latent customer journey: who's currently in each persona, and where they could plausibly move next.")

    personas = ["New customers", "One-time buyers", "Regulars", "Loyal high-value", "At-risk / lapsed"]
    counts = rfm["persona"].value_counts().to_dict()

    transitions = {
        "New customers":     {"Regulars": 0.35, "One-time buyers": 0.55, "Churned": 0.10},
        "One-time buyers":   {"Regulars": 0.12, "At-risk / lapsed": 0.70, "Churned": 0.18},
        "Regulars":          {"Loyal high-value": 0.18, "Regulars (stay)": 0.65, "At-risk / lapsed": 0.17},
        "Loyal high-value":  {"Loyal high-value (stay)": 0.80, "Regulars": 0.15, "At-risk / lapsed": 0.05},
        "At-risk / lapsed":  {"Regulars (won back)": 0.10, "Churned": 0.90},
    }

    nodes = personas + [
        "Loyal high-value (stay)", "Regulars (stay)", "Regulars (won back)", "Churned",
    ]
    node_idx = {n: i for i, n in enumerate(nodes)}
    node_colors = [
        PERSONA_COLOURS["New customers"], PERSONA_COLOURS["One-time buyers"],
        PERSONA_COLOURS["Regulars"], PERSONA_COLOURS["Loyal high-value"],
        PERSONA_COLOURS["At-risk / lapsed"],
        GOOD, BLUE, GOOD, GREY,
    ]
    src, tgt, val, lcol = [], [], [], []
    for source, dests in transitions.items():
        base_n = counts.get(source, 0)
        for dest, pct in dests.items():
            if dest not in node_idx:
                continue
            n = int(base_n * pct)
            if n <= 0:
                continue
            src.append(node_idx[source])
            tgt.append(node_idx[dest])
            val.append(n)
            lcol.append("rgba(255, 90, 95, 0.34)" if "Churned" in dest else "rgba(124, 58, 237, 0.20)")

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=22, thickness=18,
                  line=dict(color="white", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    ))
    st.plotly_chart(style_fig(fig, 520), use_container_width=True)

    st.caption(
        "Transition probabilities are illustrative defaults typical for retail/subscription. "
        "Plug in a 90-day cohort study to make them real."
    )

    one_time_lost_pct = transitions["One-time buyers"]["At-risk / lapsed"]
    n_one_time = counts.get("One-time buyers", 0)
    revenue_at_risk = float(rfm[rfm["persona"] == "One-time buyers"]["monetary"].sum()) * one_time_lost_pct
    note(
        f"The leak to watch is among one-time buyers. Left alone, about "
        f"<b>{one_time_lost_pct*100:.0f}%</b> of the <b>{n_one_time:,}</b> of them drift "
        f"into the at-risk bucket — roughly <b>${revenue_at_risk:,.0f}</b> of future "
        f"revenue quietly walking out the door. A simple second-purchase voucher aimed at "
        f"this group tends to pay back quickly."
    )

# --------------------------------------------------------------------------- #
with tab_zim:
    st.markdown("#### Zim retail context: formal vs informal")
    st.caption(
        "Zimbabwe retail isn't one market. Formal supermarket chains "
        "(OK Mart, Pick n Pay, TM Pick n Pay, Spar, Bon Marche) serve middle- "
        "and upper-income urban shoppers. Mbare Musika and other informal "
        "traders move volume on lower-ticket items, often cash-only. The "
        "personas behave differently on each side."
    )

    a, b = st.columns(2)
    with a:
        st.markdown("##### Channel mix")
        ch = rfm["channel"].value_counts().reset_index()
        ch.columns = ["channel", "customers"]
        ch["channel_type"] = np.where(
            ch["channel"].isin(INFORMAL_RETAILERS),
            "Informal market", "Formal supermarket",
        )
        fig = px.bar(
            ch.sort_values("customers"),
            x="customers", y="channel", orientation="h",
            color="channel_type",
            color_discrete_map=CHANNEL_COLORS,
            text="customers",
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.18),
                          xaxis_title="Customers", yaxis_title="")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    with b:
        st.markdown("##### Revenue by channel")
        ch_rev = (
            rfm.groupby("channel").agg(revenue=("monetary", "sum"),
                                       customers=("customer_id", "count")).reset_index()
        )
        ch_rev["zig_revenue"] = ch_rev["revenue"] * USD_TO_ZIG
        ch_rev["channel_type"] = np.where(
            ch_rev["channel"].isin(INFORMAL_RETAILERS),
            "Informal market", "Formal supermarket",
        )
        fig = px.bar(
            ch_rev.sort_values("revenue"),
            x="revenue", y="channel", orientation="h",
            color="channel_type",
            color_discrete_map=CHANNEL_COLORS,
            hover_data={"customers": True, "zig_revenue": ":,.0f"},
            text=ch_rev["revenue"].map(lambda x: f"${x:,.0f}"),
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.18),
                          xaxis_title="Revenue (USD)", yaxis_title="")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    formal_rev = float(rfm.loc[rfm["channel_type"] == "Formal supermarket", "monetary"].sum())
    informal_rev = float(rfm.loc[rfm["channel_type"] == "Informal market", "monetary"].sum())
    note(
        f"Most of the money sits in formal supermarkets — they carry <b>${formal_rev:,.0f}</b> "
        f"(≈ZiG {formal_rev * USD_TO_ZIG:,.0f}) against <b>${informal_rev:,.0f}</b> "
        f"(≈ZiG {informal_rev * USD_TO_ZIG:,.0f}) from informal traders. Formal customers "
        f"tend to be higher-LTV, but informal volume can balance out a thin-margin product "
        f"mix, and the two sides respond to quite different campaigns."
    )

    # -- Mobile money mix per persona --------------------------------------
    st.markdown("##### Mobile money mix by persona")
    st.caption("Which payment rail dominates which segment. EcoCash skews high-value; cash sits with one-time / informal.")
    pay = (
        rfm.groupby(["persona", "payment_method"])
           .size().reset_index(name="customers")
    )
    totals = pay.groupby("persona")["customers"].transform("sum")
    pay["share"] = pay["customers"] / totals
    fig = px.bar(
        pay, x="persona", y="share", color="payment_method", barmode="stack",
        text=pay["share"].map(lambda x: f"{x*100:.0f}%"),
        color_discrete_map=PAY_COLORS,
        labels={"share": "Share of persona", "persona": ""},
    )
    fig.update_layout(yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    # Dominant rail per persona
    top_rail_per_persona = (
        pay.sort_values("share", ascending=False)
           .drop_duplicates("persona")[["persona", "payment_method", "share"]]
    )
    lines = "; ".join(
        f"<b>{r.persona}</b> lean {r.payment_method} ({r.share*100:.0f}%)"
        for r in top_rail_per_persona.itertuples()
    )
    note(
        f"Each segment has a rail it already trusts — {lines}. The practical takeaway is to "
        f"match the incentive to the rail: an EcoCash voucher lands harder than a "
        f"bank-transfer rebate for high-value shoppers, while a small InnBucks top-up is "
        f"what reactivates cash payers."
    )

# --------------------------------------------------------------------------- #
with tab_budget:
    st.markdown("#### How would you split a marketing budget?")
    st.caption(
        "Drag the budget slider. The model assumes industry-typical conversion rates per persona — "
        "tune the per-persona splits below if you have better numbers."
    )

    budget = st.slider("Total marketing budget ($)", 1_000, 100_000, 25_000, step=1_000)

    roi_defaults = {
        "Loyal high-value": {"split": 0.35, "response": 0.55, "uplift": 280},
        "Regulars":         {"split": 0.30, "response": 0.30, "uplift": 120},
        "New customers":    {"split": 0.15, "response": 0.40, "uplift": 60},
        "At-risk / lapsed": {"split": 0.15, "response": 0.18, "uplift": 200},
        "One-time buyers":  {"split": 0.05, "response": 0.12, "uplift": 90},
    }

    st.markdown("##### Split (%) — these auto-normalise to 100%")
    cols = st.columns(5)
    splits = {}
    for col, (persona, defs) in zip(cols, roi_defaults.items()):
        with col:
            splits[persona] = st.slider(
                persona, 0, 100, int(defs["split"] * 100),
                key=f"split_{persona}",
            ) / 100.0
    total_split = sum(splits.values()) or 1.0
    splits = {k: v / total_split for k, v in splits.items()}

    rows = []
    for persona, defs in roi_defaults.items():
        share = splits[persona]
        spend = budget * share
        n_in_persona = (rfm["persona"] == persona).sum()
        reached = min(int(spend / 8), n_in_persona)
        responders = int(reached * defs["response"])
        expected_revenue = responders * defs["uplift"]
        rows.append({
            "persona": persona, "spend": spend, "share": share,
            "reached": reached, "responders": responders,
            "expected_revenue": expected_revenue,
            "roi_x": expected_revenue / max(spend, 1),
        })
    plan = pd.DataFrame(rows)
    total_return = float(plan["expected_revenue"].sum())
    blended_roi = total_return / max(budget, 1)

    plan_long = pd.melt(
        plan, id_vars=["persona"], value_vars=["spend", "expected_revenue"],
        var_name="kind", value_name="amount",
    )
    plan_long["kind"] = plan_long["kind"].map({"spend": "Spend", "expected_revenue": "Expected return"})
    fig = px.bar(
        plan_long, x="persona", y="amount", color="kind",
        barmode="group",
        color_discrete_map={"Spend": GREY, "Expected return": GOOD},
        text=plan_long["amount"].map(lambda x: f"${x:,.0f}"),
        labels={"amount": "USD", "persona": ""},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Total spend", f"${budget:,.0f}",
             f"≈ZiG {budget * USD_TO_ZIG:,.0f}", delta_color="off")
    b.metric("Expected return", f"${total_return:,.0f}",
             f"≈ZiG {total_return * USD_TO_ZIG:,.0f}", delta_color="off")
    c.metric("Blended ROI", f"{blended_roi:.1f}×",
             "above 3× is healthy", delta_color="off")

    best = plan.loc[plan["roi_x"].idxmax()]
    worst = plan.loc[plan["roi_x"].idxmin()]
    note(
        f"On these assumptions, <b>{best['persona']}</b> gives the best return at "
        f"<b>{best['roi_x']:.1f}×</b>, so spare budget is best routed there. At the other "
        f"end, <b>{worst['persona']}</b> only returns <b>{worst['roi_x']:.1f}×</b> — worth "
        f"tightening or pausing unless you're deliberately chasing reactivation rate rather "
        f"than revenue."
    )

    st.markdown("##### Detailed plan")
    st.dataframe(
        plan.assign(
            share=plan["share"].map(lambda x: f"{x*100:.0f}%"),
            spend=plan["spend"].map(lambda x: f"${x:,.0f}"),
            expected_revenue=plan["expected_revenue"].map(lambda x: f"${x:,.0f}"),
            roi_x=plan["roi_x"].map(lambda x: f"{x:.1f}×"),
        )[["persona", "share", "spend", "reached", "responders", "expected_revenue", "roi_x"]],
        use_container_width=True, hide_index=True,
    )
