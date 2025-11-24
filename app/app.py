import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

# =========================
#   CONFIG AZURE ML
# =========================
# ⚠️ REMPLIS BIEN CES 3 VARIABLES AVEC TES VALEURS AZURE ML
ENDPOINT_URL = "https://stock-anomaly-ml-workspac-amklv.eastus.inference.ml.azure.com/score"   # <-- à remplacer
API_KEY = "PUT_YOUR_KEY_HERE_AFTER_PUSH"                           # <-- à remplacer
DEPLOYMENT_NAME = "stock-anomaly-detection-model-3"  # nom du déploiement


def call_azure_ml_batch(features_list):
    """
    Appelle l'endpoint Azure ML avec une liste de features :
    features_list = [[Open, High, Low, Close, Volume], ...]
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "azureml-model-deployment": DEPLOYMENT_NAME
    }

    payload = {
        "input_data": {
            "data": features_list
        }
    }

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        st.error(f"Erreur API Azure ML: {response.status_code} - {response.text}")
        return None

    return response.json()


# =========================
#    CHARGEMENT DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("stock_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


df = load_data()

# =========================
#  CONFIG INTERFACE STREAMLIT
# =========================
st.set_page_config(
    page_title="Stock Anomaly Detection Dashboard",
    layout="wide"
)

st.title("📈 Stock Dashboard — Détection d'anomalies avec Azure ML")

st.markdown(
    """
    **Dataset chargé : `stock_data.csv`**  
    Visualisation des prix, volumes et intégration du modèle de détection d’anomalies
    déployé sur **Azure Machine Learning**.
    """
)

# =========================
#       SIDEBAR
# =========================
st.sidebar.header("Filtres")

tickers = sorted(df["Ticker"].unique())
selected_ticker = st.sidebar.selectbox("Sélectionner une action :", tickers)

min_date = df["Date"].min()
max_date = df["Date"].max()

start_date, end_date = st.sidebar.date_input(
    "Sélectionner la période :",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Filtrage des données
mask = (
    (df["Ticker"] == selected_ticker) &
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
)
df_filtered = df[mask].sort_values("Date")

if df_filtered.empty:
    st.warning("⚠ Aucune donnée trouvée pour ces filtres.")
    st.stop()

# =========================
#        GRAPHIQUES
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📉 Prix de clôture — {selected_ticker}")
    fig_close = px.line(
        df_filtered,
        x="Date",
        y="Close",
        title="Évolution du prix de clôture"
    )
    st.plotly_chart(fig_close, use_container_width=True)

with col2:
    st.subheader(f"📊 Volume — {selected_ticker}")
    fig_vol = px.bar(
        df_filtered,
        x="Date",
        y="Volume",
        title="Volume échangé"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# =========================
#   TABLEAU DE DONNÉES
# =========================
st.subheader("📄 Données filtrées")
st.dataframe(df_filtered.reset_index(drop=True))

# =========================
#  DÉTECTION D'ANOMALIES
# =========================
st.markdown("---")
st.subheader("🔍 Détection d'anomalies avec le modèle Azure ML")

st.markdown(
    """
    Le bouton ci-dessous envoie les données filtrées (Open, High, Low, Close, Volume)  
    au **service Azure ML** pour détecter les anomalies avec le modèle IsolationForest.
    """
)

if st.checkbox("Afficher les features envoyées au modèle"):
    st.dataframe(df_filtered[["Date", "Open", "High", "Low", "Close", "Volume"]].head())

if st.button("🚀 Lancer la détection d'anomalies (Azure ML)"):
    # Préparation des features pour l'API
    features_list = df_filtered[["Open", "High", "Low", "Close", "Volume"]].values.tolist()

    with st.spinner("Appel du modèle Azure ML en cours..."):
        result = call_azure_ml_batch(features_list)

    if result is not None:
        # Récupération des résultats
        # On privilégie la clé "is_anomaly" si elle existe, sinon on reconstruit à partir de anomaly_predictions
        is_anomaly = result.get("is_anomaly", None)
        if is_anomaly is None:
            preds = result.get("anomaly_predictions", [])
            is_anomaly = [1 if p == -1 else 0 for p in preds]

        scores = result.get("anomaly_scores", [None] * len(is_anomaly))

        # Ajout au DataFrame
        df_results = df_filtered.copy().reset_index(drop=True)
        df_results["is_anomaly"] = is_anomaly
        df_results["anomaly_score"] = scores

        anomalies = df_results[df_results["is_anomaly"] == 1]

        st.success(
            f"Analyse terminée ✅ — {len(anomalies)} anomalies détectées "
            f"sur {len(df_results)} points."
        )

        # Tableau complet avec colonnes d'anomalie
        st.subheader("📊 Résultats avec colonnes d'anomalie")
        st.dataframe(df_results)

        # Tableau des anomalies uniquement
        if len(anomalies) > 0:
            st.subheader("🚨 Points détectés comme anomalies")
            st.dataframe(anomalies)
        else:
            st.info("Aucune anomalie détectée sur cette période et ce ticker.")

        # Graphique avec anomalies en rouge
        st.subheader("📉 Graphique avec anomalies mises en évidence")

        fig_anom = px.scatter(
            df_results,
            x="Date",
            y="Close",
            color=df_results["is_anomaly"].map({0: "Normal", 1: "Anomalie"}),
            color_discrete_map={"Normal": "blue", "Anomalie": "red"},
            title="Prix de clôture avec anomalies"
        )
        fig_anom.update_traces(mode="lines+markers")
        st.plotly_chart(fig_anom, use_container_width=True)
