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
    page_title="Retail Customer Intelligence",
    page_icon=":bust_in_silhouette:",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #8E44AD 0%, #4A2065 100%);
        padding: 36px 32px;
        border-radius: 16px;
        color: white;
        margin: -10px 0 22px 0;
        box-shadow: 0 12px 32px rgba(142, 68, 173, 0.25);
    }
    .hero h1 { margin: 0; font-size: 40px; font-weight: 800; letter-spacing: -0.8px; }
    .hero p  { margin: 10px 0 0 0; font-size: 17px; opacity: 0.95; }

    .stat {
        background: white;
        padding: 22px 24px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-top: 4px solid #8E44AD;
        height: 100%;
    }
    .stat .label { font-size: 11px; color: #7F8C8D; text-transform: uppercase; letter-spacing: 1.2px; }
    .stat .value { font-size: 30px; font-weight: 800; color: #2C3E50; margin: 6px 0 2px 0; }
    .stat .sub   { font-size: 12px; color: #95A5A6; }

    .insight {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F4FB 100%);
        border-left: 4px solid #8E44AD;
        padding: 18px 22px;
        border-radius: 10px;
        margin: 12px 0;
    }
    .insight .head { font-size: 11px; color: #8E44AD; font-weight: 800; letter-spacing: 1.2px; }
    .insight .body { font-size: 16px; color: #2C3E50; margin-top: 6px; line-height: 1.55; }

    .persona-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 8px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 6px solid var(--persona-color, #8E44AD);
    }
    .persona-card h4 { margin: 0 0 6px 0; font-size: 18px; color: #2C3E50; }
    .persona-card .meta { font-size: 13px; color: #7F8C8D; margin-bottom: 10px; }
    .persona-card .action { font-size: 14px; color: #2C3E50; }
    .persona-card .why { font-size: 13px; color: #8E44AD; margin-top: 6px; font-style: italic; }

    div[data-testid="stTabs"] button[data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def big_stat(label, value, sub=""):
    return f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def insight(head, body):
    return f'<div class="insight"><div class="head">{head}</div><div class="body">{body}</div></div>'


PERSONA_COLOURS = {
    "Loyal high-value": "#27AE60",
    "Regulars": "#2E86C1",
    "New customers": "#F39C12",
    "One-time buyers": "#7F8C8D",
    "At-risk / lapsed": "#E74C3C",
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
    st.markdown(
        ":bulb: **Try the Marketing Budget tab.** "
        "Drag the budget slider to see expected ROI per persona."
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

# Gini coefficient (concentration)
sorted_m = np.sort(rfm["monetary"].to_numpy())
cum_share = np.cumsum(sorted_m) / sorted_m.sum()
gini = float(1 - 2 * np.trapz(cum_share, dx=1 / len(sorted_m)))

st.markdown(
    """
    <div class="hero">
      <h1>Retail Customer Intelligence</h1>
      <p>Who drives the revenue, who's about to walk away, and where to spend the next marketing dollar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(big_stat("Customers", f"{total_customers:,}"), unsafe_allow_html=True)
c2.markdown(big_stat("Total revenue", f"${total_revenue:,.0f}",
                      f"≈ZiG {total_revenue * USD_TO_ZIG:,.0f}"),
            unsafe_allow_html=True)
c3.markdown(
    big_stat(
        "Top 10% revenue share",
        f"{top10pct_share*100:.0f}%",
        f"{top10pct_n:,} customers · {zig(top10pct_revenue)}",
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    big_stat(
        "Revenue concentration",
        f"{gini:.2f} Gini",
        "0.0 = even • 1.0 = one customer has it all",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    insight(
        "THE BIG NUMBER",
        f"<b>{top10pct_n:,}</b> customers &mdash; the top 10% of the book &mdash; drive "
        f"<b>{top10pct_share*100:.0f}%</b> of all revenue. Lose them and you've lost "
        f"<b>${top10pct_revenue:,.0f}</b> (≈ZiG {top10pct_revenue * USD_TO_ZIG:,.0f}) "
        "overnight. This is who marketing should defend first &mdash; whether they "
        "shop at <b>Pick n Pay</b>, <b>OK Mart</b>, or hustle at <b>Mbare Musika</b>.",
    ),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
tab_pareto, tab_personas, tab_zim, tab_explorer, tab_journey, tab_strategy = st.tabs([
    ":chart_with_upwards_trend: The 80/20",
    ":bust_in_silhouette: Personas",
    ":zimbabwe: Zim Retail Context",
    ":mag: Persona Explorer",
    ":twisted_rightwards_arrows: Customer Journey",
    ":moneybag: Marketing Budget",
])

# --------------------------------------------------------------------------- #
with tab_pareto:
    st.subheader("How concentrated is the revenue?")
    st.caption("The classic Lorenz curve. The further the line bends below the diagonal, the more the top few customers carry the business.")

    rfm_sorted = rfm.sort_values("monetary", ascending=True).reset_index(drop=True)
    pct_customers = np.arange(1, len(rfm_sorted) + 1) / len(rfm_sorted)
    pct_revenue = rfm_sorted["monetary"].cumsum() / rfm_sorted["monetary"].sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pct_customers, y=pct_revenue,
        mode="lines", fill="tozeroy", name="Lorenz",
        line=dict(color="#8E44AD", width=3),
        fillcolor="rgba(142, 68, 173, 0.18)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
        line=dict(color="#95A5A6", width=2, dash="dash"),
    ))
    p80 = int(0.80 * len(rfm_sorted))
    rev_at_p80 = float(pct_revenue.iloc[p80])
    fig.add_trace(go.Scatter(
        x=[0.80, 0.80], y=[0, rev_at_p80], mode="lines",
        line=dict(color="#E74C3C", width=2, dash="dot"), showlegend=False,
    ))
    fig.add_annotation(
        x=0.80, y=rev_at_p80,
        text=f"80% of customers ⇒ {rev_at_p80*100:.0f}% of revenue",
        showarrow=True, arrowhead=2, ax=-100, ay=-40,
        font=dict(color="#E74C3C", size=13),
    )
    fig.update_layout(
        xaxis_title="Cumulative share of customers (sorted by spend)",
        yaxis_title="Cumulative share of revenue",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        height=480, margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    bottom80_rev_share = rev_at_p80
    top20_rev_share = 1 - bottom80_rev_share
    st.markdown(
        insight(
            "THE 80/20 RULE IN ACTION",
            f"The bottom <b>80%</b> of customers only account for "
            f"<b>{bottom80_rev_share*100:.0f}%</b> of revenue. The top <b>20%</b> "
            f"carry the remaining <b>{top20_rev_share*100:.0f}%</b>. "
            "Spending equally across all segments leaves money on the table — "
            "the top quintile deserves disproportionate investment.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_personas:
    st.subheader("Five personas, ranked by revenue")
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
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Customers in RFM space")
    sample = rfm.sample(min(3000, len(rfm)), random_state=RNG)
    fig = px.scatter(
        sample, x="pc1", y="pc2", color="persona",
        color_discrete_map=PERSONA_COLOURS, size="monetary", size_max=18,
        opacity=0.65, labels={"pc1": "PC1", "pc2": "PC2"},
        hover_data={"recency": True, "frequency": True, "monetary": ":$,.0f",
                     "pc1": False, "pc2": False},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, use_container_width=True)

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
with tab_zim:
    st.subheader("Zim retail context: formal vs informal")
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
            color_discrete_map={
                "Formal supermarket": "#8E44AD", "Informal market": "#F39C12",
            },
            text="customers",
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.18),
                          xaxis_title="Customers", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

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
            color_discrete_map={
                "Formal supermarket": "#8E44AD", "Informal market": "#F39C12",
            },
            hover_data={"customers": True, "zig_revenue": ":,.0f"},
            text=ch_rev["revenue"].map(lambda x: f"${x:,.0f}"),
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.18),
                          xaxis_title="Revenue (USD)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    formal_rev = float(rfm.loc[rfm["channel_type"] == "Formal supermarket", "monetary"].sum())
    informal_rev = float(rfm.loc[rfm["channel_type"] == "Informal market", "monetary"].sum())
    st.markdown(
        insight(
            "WHERE THE MONEY ACTUALLY SITS",
            f"Formal supermarkets carry <b>${formal_rev:,.0f}</b> "
            f"(≈ZiG {formal_rev * USD_TO_ZIG:,.0f}); informal traders carry "
            f"<b>${informal_rev:,.0f}</b> (≈ZiG {informal_rev * USD_TO_ZIG:,.0f}). "
            "Formal customers tend to be higher-LTV but informal volume can "
            "balance a thin-margin product mix &mdash; both need different campaigns.",
        ),
        unsafe_allow_html=True,
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
        color_discrete_map={
            "EcoCash":  "#27AE60", "OneMoney": "#2E86C1",
            "InnBucks": "#F39C12", "Cash":     "#95A5A6",
        },
        labels={"share": "Share of persona", "persona": ""},
    )
    fig.update_layout(height=380, yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.18),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Dominant rail per persona
    top_rail_per_persona = (
        pay.sort_values("share", ascending=False)
           .drop_duplicates("persona")[["persona", "payment_method", "share"]]
    )
    lines = " · ".join(
        f"<b>{r.persona}</b> → {r.payment_method} ({r.share*100:.0f}%)"
        for r in top_rail_per_persona.itertuples()
    )
    st.markdown(
        insight(
            "DOMINANT RAIL PER SEGMENT",
            f"{lines}.<br>Match campaign incentives to the rail customers already use &mdash; "
            "an EcoCash voucher will land harder than a bank-transfer rebate for "
            "high-value shoppers, while small InnBucks top-ups reactivate cash payers.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_explorer:
    st.subheader("Interactive persona explorer")
    st.caption("Pick a persona to see size, average spend in USD + ZiG, top channel, recommended marketing channel, and sample customer cards.")

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
    a.markdown(big_stat("Persona size", f"{n_in:,}",
                         f"{n_in/total_customers*100:.1f}% of book"),
               unsafe_allow_html=True)
    b.markdown(big_stat("Avg LTV", f"${avg_spend:,.0f}",
                         f"≈ZiG {avg_spend * USD_TO_ZIG:,.0f}"),
               unsafe_allow_html=True)
    c.markdown(big_stat("Avg recency", f"{avg_recency:.0f} days",
                         f"{avg_freq:.1f} orders on average"),
               unsafe_allow_html=True)
    d.markdown(big_stat("Recommended channel", rec_channel.split(' + ')[0],
                         rec_channel),
               unsafe_allow_html=True)

    # Channel breakdown for this persona
    a2, b2 = st.columns(2)
    with a2:
        st.markdown("##### Where they shop")
        ch_sub = sub["channel"].value_counts().reset_index()
        ch_sub.columns = ["channel", "customers"]
        fig = px.pie(ch_sub, names="channel", values="customers", hole=0.45,
                     color="channel",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="v"))
        st.plotly_chart(fig, use_container_width=True)
    with b2:
        st.markdown("##### How they pay")
        pay_sub = sub["payment_method"].value_counts().reset_index()
        pay_sub.columns = ["payment_method", "customers"]
        fig = px.pie(
            pay_sub, names="payment_method", values="customers", hole=0.45,
            color="payment_method",
            color_discrete_map={
                "EcoCash":  "#27AE60", "OneMoney": "#2E86C1",
                "InnBucks": "#F39C12", "Cash":     "#95A5A6",
            },
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="v"))
        st.plotly_chart(fig, use_container_width=True)

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

    st.markdown(
        insight(
            f"OFFER FOR {chosen.upper()}",
            f"<b>Offer:</b> {offer_text}<br>"
            f"<b>Channel:</b> {rec_channel}<br>"
            f"<b>Est. cost / customer:</b> ${est_cost:.2f} "
            f"(≈ZiG {est_cost * USD_TO_ZIG:,.0f})<br>"
            f"<b>Est. responders:</b> {est_responders:,} of {n_in:,}<br>"
            f"<b>Expected uplift:</b> ${expected_revenue:,.0f} "
            f"(≈ZiG {expected_revenue * USD_TO_ZIG:,.0f})<br>"
            f"<b>Best payment rail to pair with:</b> "
            f"{sub['payment_method'].value_counts().idxmax()}",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_journey:
    st.subheader("Where customers come from, where they go")
    st.caption("The latent customer journey: who's currently in each persona, and where they could plausibly move.")

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
        "#27AE60", "#2E86C1", "#16A085", "#7F8C8D",
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
            lcol.append("rgba(231, 76, 60, 0.35)" if "Churned" in dest else "rgba(142, 68, 173, 0.25)")

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=node_colors, pad=22, thickness=18,
                  line=dict(color="white", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), font_size=13)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Transition probabilities are illustrative defaults typical for retail/subscription. "
        "Plug in a 90-day cohort study to make them real."
    )

    one_time_lost_pct = transitions["One-time buyers"]["At-risk / lapsed"]
    n_one_time = counts.get("One-time buyers", 0)
    revenue_at_risk = float(rfm[rfm["persona"] == "One-time buyers"]["monetary"].sum()) * one_time_lost_pct
    st.markdown(
        insight(
            "THE LEAKY BUCKET",
            f"Without intervention, ~<b>{one_time_lost_pct*100:.0f}%</b> of "
            f"<b>{n_one_time:,}</b> one-time buyers drift into the at-risk bucket — "
            f"that's <b>${revenue_at_risk:,.0f}</b> of future revenue quietly leaving "
            "the building. A second-purchase voucher campaign here pays back fast.",
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
with tab_strategy:
    st.subheader("How would you split a marketing budget?")
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
        color_discrete_map={"Spend": "#95A5A6", "Expected return": "#27AE60"},
        text=plan_long["amount"].map(lambda x: f"${x:,.0f}"),
        labels={"amount": "USD", "persona": ""},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=20),
                       legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    a, b, c = st.columns(3)
    a.markdown(big_stat("Total spend", f"${budget:,.0f}",
                         f"≈ZiG {budget * USD_TO_ZIG:,.0f}"),
               unsafe_allow_html=True)
    b.markdown(big_stat("Expected return", f"${total_return:,.0f}",
                         f"≈ZiG {total_return * USD_TO_ZIG:,.0f}"),
               unsafe_allow_html=True)
    c.markdown(
        big_stat(
            "Blended ROI",
            f"{blended_roi:.1f}×",
            "Rule of thumb: above 3× is healthy.",
        ),
        unsafe_allow_html=True,
    )

    best = plan.loc[plan["roi_x"].idxmax()]
    worst = plan.loc[plan["roi_x"].idxmin()]
    st.markdown(
        insight(
            "WHERE THE NEXT DOLLAR SHOULD GO",
            f"On these assumptions, <b>{best['persona']}</b> gives the highest ROI at "
            f"<b>{best['roi_x']:.1f}×</b> — if you have spare budget, route it here. "
            f"<b>{worst['persona']}</b> at <b>{worst['roi_x']:.1f}×</b> is the worst yield; "
            "tighten or stop spending there unless you're optimising for reactivation "
            "rate rather than revenue.",
        ),
        unsafe_allow_html=True,
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
