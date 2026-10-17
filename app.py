"""
Coffeelab CFO-dashboard
AI miniopdracht 3: AI data applicatie (dashboard, cards, filters)

Doelgroep: de CFO van Coffeelab, die snel inzicht wil in omzet, openstaande
bedragen (cashflow) en spreiding over provincies/productcategorieen, om
strategische keuzes te onderbouwen (zie Coffeelab-casus).

De AI-functie in dit dashboard werkt via een directe API-call naar Groq
(geen los chatvenster): de knop "Genereer AI-inzicht" stuurt een samenvatting
van de HUIDIGE gefilterde data naar het taalmodel en toont het antwoord
binnen de app.
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------
DATA_DIR = "data"
GROQ_MODEL = "openai/gpt-oss-120b"  # zelfde model als in opdracht 1 (EDA)

# Koffiestijl kleurenpalet (gebruikt in grafieken)
COFFEE_COLORS = ["#6F4E37", "#C68B59", "#A9746E", "#8B5E3C", "#D9B48F", "#4B3621", "#E8C39E"]

MAAND_NAMEN = {
    1: "jan", 2: "feb", 3: "mrt", 4: "apr", 5: "mei", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "okt", 11: "nov", 12: "dec",
}

st.set_page_config(
    page_title="Coffeelab CFO-dashboard",
    page_icon="☕",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Koffiestijl (CSS)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #FFF8F0; }
    [data-testid="stSidebar"] { background-color: #F0E0CC; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    h1, h2, h3 { color: #4B3621; }
    p, label, span, div { color: #3B2317; }

    div[data-testid="stMetric"] {
        background-color: #FFF3E0;
        border: 1px solid #D9B48F;
        padding: 14px 14px 8px 14px;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] { color: #6F4E37; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #4B3621; }

    .stButton button {
        background-color: #6F4E37;
        color: #FFF8F0;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover { background-color: #4B3621; color: #FFF8F0; }

    [data-testid="stExpander"] {
        background-color: #FBEFE0;
        border-radius: 8px;
        border: 1px solid #E3C9A3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data laden en samenvoegen
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    verkoop = pd.read_parquet(os.path.join(DATA_DIR, "f_Verkooptransacties.parquet"))
    artikelen = pd.read_parquet(os.path.join(DATA_DIR, "d_Artikelen.parquet"))
    klanten = pd.read_parquet(os.path.join(DATA_DIR, "d_Klanten.parquet"))

    verkoop["Orderdatum"] = pd.to_datetime(verkoop["Orderdatum"], errors="coerce")

    df = verkoop.merge(
        artikelen[["Artikelnummer", "Product L1", "Product L2"]],
        on="Artikelnummer",
        how="left",
    )
    df = df.merge(
        klanten[["Klantnummer", "Provincie", "Type"]],
        on="Klantnummer",
        how="left",
    )

    df["_jaar"] = df["Orderdatum"].dt.year
    df["_jaarmaand"] = df["Orderdatum"].dt.to_period("M").astype(str)
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Hulpfunctie: aan-/afvinkbare filtergroep met "Alles" / "Geen" knoppen
# ---------------------------------------------------------------------------
def checkbox_filter(label, opties, key_prefix, expanded=True):
    with st.sidebar.expander(label, expanded=expanded):
        c1, c2 = st.columns(2)
        if c1.button("Alles", key=f"{key_prefix}_alles_btn", use_container_width=True):
            for optie in opties:
                st.session_state[f"{key_prefix}_{optie}"] = True
        if c2.button("Geen", key=f"{key_prefix}_geen_btn", use_container_width=True):
            for optie in opties:
                st.session_state[f"{key_prefix}_{optie}"] = False

        geselecteerd = []
        for optie in opties:
            k = f"{key_prefix}_{optie}"
            if k not in st.session_state:
                st.session_state[k] = True
            if st.checkbox(str(optie), key=k):
                geselecteerd.append(optie)
    return geselecteerd


# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
st.sidebar.header("☕ Filters")

# --- Periode: duidelijk kiezen tussen Dag / Maand / Jaar ---
st.sidebar.subheader("Periode")
granulariteit = st.sidebar.radio(
    "Type periode", ["Dag", "Maand", "Jaar"], horizontal=True, key="periode_type"
)

min_datum = df["Orderdatum"].min()
max_datum = df["Orderdatum"].max()

if granulariteit == "Jaar":
    jaar_opties = sorted(df["_jaar"].dropna().unique().astype(int).tolist())
    gekozen_jaren = st.sidebar.multiselect(
        "Jaren", jaar_opties, default=jaar_opties, key="jaren_select"
    )
    periode_mask = df["_jaar"].isin(gekozen_jaren)

elif granulariteit == "Maand":
    maand_opties = sorted(df["_jaarmaand"].dropna().unique().tolist())
    maand_labels = {
        ym: f"{MAAND_NAMEN[int(ym.split('-')[1])]} {ym.split('-')[0]}" for ym in maand_opties
    }
    gekozen_maanden = st.sidebar.multiselect(
        "Maanden",
        maand_opties,
        default=maand_opties,
        format_func=lambda ym: maand_labels[ym],
        key="maanden_select",
    )
    periode_mask = df["_jaarmaand"].isin(gekozen_maanden)

else:  # Dag
    periode = st.sidebar.date_input(
        "Exacte periode (van - tot)",
        value=(min_datum, max_datum),
        min_value=min_datum,
        max_value=max_datum,
        key="dag_select",
    )
    if isinstance(periode, tuple) and len(periode) == 2:
        start, eind = periode
    else:
        start, eind = min_datum, max_datum
    periode_mask = (df["Orderdatum"] >= pd.Timestamp(start)) & (
        df["Orderdatum"] <= pd.Timestamp(eind)
    )

# --- Provincie / productcategorie / status: aan-/afvinkbaar met Alles/Geen ---
provincies = sorted(df["Provincie"].dropna().unique().tolist())
gekozen_provincies = checkbox_filter("Provincie (d_Klanten)", provincies, "prov")

categorieen = sorted(df["Product L1"].dropna().unique().tolist())
gekozen_categorieen = checkbox_filter("Productcategorie (Product L1)", categorieen, "cat")

statussen = sorted(df["Status"].dropna().unique().tolist())
gekozen_status = checkbox_filter("Status (f_Verkooptransacties)", statussen, "stat")

# ---------------------------------------------------------------------------
# Filters toepassen
# ---------------------------------------------------------------------------
# 'gefilterd' = alles wat je links hebt aangevinkt, incl. status
mask = (
    periode_mask
    & df["Provincie"].isin(gekozen_provincies)
    & df["Product L1"].isin(gekozen_categorieen)
    & df["Status"].isin(gekozen_status)
)
gefilterd = df[mask]

# 'basis' = periode/provincie/categorie, maar bewust ZONDER statusfilter.
# Nodig om een betrouwbaar "% nog openstaand" te kunnen tonen, ongeacht welke
# statussen je links hebt aan-/uitgevinkt.
basis_mask = (
    periode_mask
    & df["Provincie"].isin(gekozen_provincies)
    & df["Product L1"].isin(gekozen_categorieen)
)
basis = df[basis_mask]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("☕ Coffeelab CFO-dashboard")
st.caption(
    "Gebaseerd op f_Verkooptransacties, d_Artikelen en d_Klanten "
    f"— {len(gefilterd):,} van {len(df):,} rijen zichtbaar na filtering."
)

# ---------------------------------------------------------------------------
# Cards (KPI's) — reageren op alle filters, incl. status
# ---------------------------------------------------------------------------
totale_omzet = gefilterd["Bedrag"].sum()
aantal_orders = gefilterd["Ordernummer"].nunique()
openstaand_bedrag = gefilterd.loc[gefilterd["Status"] == "Openstaand", "Bedrag"].sum()
gem_orderwaarde = totale_omzet / aantal_orders if aantal_orders else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Totale omzet", f"€ {totale_omzet:,.0f}")
col2.metric("Aantal orders", f"{aantal_orders:,}")
col3.metric("Openstaand bedrag", f"€ {openstaand_bedrag:,.0f}")
col4.metric("Gem. orderwaarde", f"€ {gem_orderwaarde:,.0f}")

# ---------------------------------------------------------------------------
# Nieuwe cards: percentage nog openstaand (los van de statusfilter)
# ---------------------------------------------------------------------------
omzet_basis = basis["Bedrag"].sum()
orders_basis = basis["Ordernummer"].nunique()
omzet_open_basis = basis.loc[basis["Status"] == "Openstaand", "Bedrag"].sum()
orders_open_basis = basis.loc[basis["Status"] == "Openstaand", "Ordernummer"].nunique()

pct_omzet_open = (omzet_open_basis / omzet_basis * 100) if omzet_basis else 0
pct_orders_open = (orders_open_basis / orders_basis * 100) if orders_basis else 0

st.caption(
    "De twee percentages en taartdiagrammen hieronder tellen altijd over "
    "**alle statussen samen** (los van de statusfilter links), zodat ze "
    "blijven kloppen ongeacht welke status je hebt aan-/uitgevinkt."
)
col5, col6 = st.columns(2)
col5.metric("% omzet nog openstaand", f"{pct_omzet_open:.1f}%")
col6.metric("% orders nog openstaand", f"{pct_orders_open:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Visualisaties
# ---------------------------------------------------------------------------
links, rechts = st.columns(2)

with links:
    per_maand = (
        gefilterd.dropna(subset=["Orderdatum"])
        .assign(Maand=gefilterd["Orderdatum"].dt.to_period("M").astype(str))
        .groupby("Maand", as_index=False)["Bedrag"].sum()
        .sort_values("Maand")
    )
    fig1 = px.line(per_maand, x="Maand", y="Bedrag", title="Omzet per maand")
    fig1.update_traces(line_color=COFFEE_COLORS[0])
    st.plotly_chart(fig1, use_container_width=True)

with rechts:
    per_provincie = (
        gefilterd.groupby("Provincie", as_index=False)["Bedrag"].sum()
        .sort_values("Bedrag", ascending=False)
    )
    fig2 = px.bar(
        per_provincie, x="Provincie", y="Bedrag", title="Omzet per provincie",
        color_discrete_sequence=[COFFEE_COLORS[0]],
    )
    st.plotly_chart(fig2, use_container_width=True)

per_categorie = (
    gefilterd.groupby("Product L1", as_index=False)["Bedrag"].sum()
    .sort_values("Bedrag", ascending=False)
)
fig3 = px.pie(
    per_categorie, names="Product L1", values="Bedrag",
    title="Omzet per productcategorie (Product L1)",
    color_discrete_sequence=COFFEE_COLORS,
)
st.plotly_chart(fig3, use_container_width=True)

# --- Betaald vs. openstaand: omzet en orders (los van de statusfilter) ---
st.subheader("Betaald vs. openstaand")
p1, p2 = st.columns(2)

omzet_split = basis.groupby("Status", as_index=False)["Bedrag"].sum()
fig4 = px.pie(
    omzet_split, names="Status", values="Bedrag",
    title="Omzet: betaald vs. openstaand",
    color_discrete_sequence=COFFEE_COLORS,
)
p1.plotly_chart(fig4, use_container_width=True)

orders_split = (
    basis.groupby("Status")["Ordernummer"].nunique().reset_index(name="Aantal")
)
fig5 = px.pie(
    orders_split, names="Status", values="Aantal",
    title="Orders: betaald vs. openstaand",
    color_discrete_sequence=COFFEE_COLORS,
)
p2.plotly_chart(fig5, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# AI-inzicht via API-call (Groq)
# ---------------------------------------------------------------------------
st.subheader("🤖 AI-inzicht (via Groq API)")
st.caption(
    "Deze knop stuurt een samenvatting van de huidige gefilterde cijfers "
    "rechtstreeks naar de Groq API en toont het antwoord hieronder — er wordt "
    "geen los chatvenster gebruikt."
)

if st.button("Genereer AI-inzicht"):
    # Lokaal: via 'set GROQ_API_KEY=...' in de terminal (omgevingsvariabele).
    # Op Streamlit Community Cloud: via Settings -> Secrets (st.secrets).
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        st.error(
            "Geen GROQ_API_KEY gevonden. Lokaal: zet 'set GROQ_API_KEY=jouw_sleutel' "
            "in je terminal voordat je de app start. Op Streamlit Cloud: zet hem bij "
            "Settings -> Secrets van de app."
        )
    else:
        samenvatting = {
            "aantal_rijen": int(len(gefilterd)),
            "totale_omzet": round(float(totale_omzet), 2),
            "aantal_orders": int(aantal_orders),
            "openstaand_bedrag": round(float(openstaand_bedrag), 2),
            "gem_orderwaarde": round(float(gem_orderwaarde), 2),
            "pct_omzet_nog_openstaand": round(float(pct_omzet_open), 1),
            "pct_orders_nog_openstaand": round(float(pct_orders_open), 1),
            "omzet_per_provincie": per_provincie.round(2).to_dict(orient="records"),
            "omzet_per_categorie": per_categorie.round(2).to_dict(orient="records"),
        }

        with st.spinner("AI analyseert de gefilterde cijfers..."):
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Je bent een data-analist die de CFO van Coffeelab "
                            "ondersteunt. Je krijgt samengevatte cijfers van het "
                            "huidige, gefilterde deel van het verkoopdashboard. "
                            "Geef in het Nederlands: (1) de belangrijkste "
                            "observatie, (2) een risico of aandachtspunt "
                            "(bijvoorbeeld rond het openstaande bedrag), en "
                            "(3) een concrete aanbeveling. Maximaal 150 woorden, "
                            "duidelijke taal, geen jargon zonder uitleg."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Cijfers van het huidige dashboard-filter: {samenvatting}",
                    },
                ],
            )
            st.success(response.choices[0].message.content)

st.divider()

# ---------------------------------------------------------------------------
# Gefilterde data (raw preview)
# ---------------------------------------------------------------------------
with st.expander("Bekijk de gefilterde data (eerste 100 rijen)"):
    st.caption(
        "Dit is een voorbeeld van de onderliggende, rij-voor-rij verkoopdata "
        "(uit f_Verkooptransacties) die na jouw filters overblijft — dus elke "
        "rij is één artikelregel op een order, met de gekoppelde "
        "productcategorie (uit d_Artikelen) en provincie (uit d_Klanten). "
        "De cards en grafieken hierboven zijn samenvattingen van precies "
        "deze rijen."
    )
    st.dataframe(gefilterd.head(100))
