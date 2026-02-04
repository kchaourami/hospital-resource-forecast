import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from prophet import Prophet # type: ignore

st.set_page_config(page_title="Hospital Resource Forecast", layout="wide")

DATA_PATH = "data/synthetic/hospital_daily_activity.csv"

# Capacités 
CAPACITY = {
    "Urgences": {"beds": 200, "staff": 120},
    "Reanimation": {"beds": 150, "staff": 180},
    "Cardiologie": {"beds": 100, "staff": 80},
}

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def add_future_scenario(future: pd.DataFrame, scenario: str, duration_days: int, event_cols: list, last_date: pd.Timestamp) -> pd.DataFrame:
    for c in event_cols:
        future[c] = 0

    if scenario == "Normal":
        return future

    scenario_map = {
        "Grippe": "event_flu",
        "COVID-19": "event_covid",
        "Canicule": "event_heatwave",
    }
    col = scenario_map.get(scenario)

    if col and col in future.columns:
        start = last_date + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=duration_days - 1)
        mask = (future["ds"] >= start) & (future["ds"] <= end)
        future.loc[mask, col] = 1

    return future

def make_recommendations(service: str, forecast_h: pd.DataFrame, df_hist: pd.DataFrame) -> list:
    cap_beds = CAPACITY[service]["beds"]
    base_staff = CAPACITY[service]["staff"]

    # Estimation ratio lits/admissions à partir de l'historique
    hist_ratio = (df_hist["beds_occupied"] / df_hist["y"].replace(0, np.nan)).dropna()
    bed_ratio = float(hist_ratio.median()) if len(hist_ratio) else 0.75
    bed_ratio = min(max(bed_ratio, 0.5), 0.9)  

    hist_staff_ratio = (df_hist["y"] / df_hist["staff_on_duty"].replace(0, np.nan)).dropna()
    adm_per_staff = float(hist_staff_ratio.median()) if len(hist_staff_ratio) else 2.5
    adm_per_staff = min(max(adm_per_staff, 0.5), 10)

    peak_row = forecast_h.loc[forecast_h["yhat"].idxmax()]
    peak_date = peak_row["ds"].date()
    peak_yhat = float(peak_row["yhat"])

    beds_needed = int(np.ceil(peak_yhat * bed_ratio))
    staff_needed = int(np.ceil(peak_yhat / adm_per_staff))

    recos = []
    # Alerte lits
    if beds_needed >= int(0.9 * cap_beds):
        extra_beds = max(0, beds_needed - cap_beds)
        recos.append(f"Risque de saturation lits (pic ~ {peak_yhat:.0f} admissions le {peak_date}). "
                     f"Lits estimés requis ≈ {beds_needed} / capacité {cap_beds}. "
                     f"Recommandation: prévoir +{extra_beds} lits (ou délestage / transfert).")
    else:
        recos.append(f"Lits: pas de saturation attendue sur l’horizon (pic estimé {beds_needed}/{cap_beds}).")

    # Alerte staff
    if staff_needed >= int(1.05 * base_staff):
        extra_staff = max(0, staff_needed - base_staff)
        recos.append(f"Tension personnel possible (staff requis ≈ {staff_needed} / baseline {base_staff}). "
                     f"Recommandation: prévoir +{extra_staff} soignants (renfort, astreintes, ajustement planning).")
    else:
        recos.append(f"Personnel: niveau baseline a priori suffisant (requis ≈ {staff_needed}/{base_staff}).")

    return recos

@st.cache_resource
def fit_prophet_model(df_service: pd.DataFrame, use_regressors: bool):
    df_service = df_service.sort_values("ds").copy()

    if not use_regressors:
        train_df = df_service[["ds", "y"]].copy()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(train_df)
        return m, df_service, [] 

    # Regressors
    reg = df_service[["ds", "y", "is_winter", "event"]].copy()
    reg = pd.get_dummies(reg, columns=["event"], drop_first=False)

    all_event_cols = [c for c in reg.columns if c.startswith("event_")]
    reg[all_event_cols] = reg[all_event_cols].astype(int)
    reg["is_winter"] = reg["is_winter"].astype(int)

    event_cols = [c for c in all_event_cols if c != "event_none"]


    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_regressor("is_winter")
    for c in event_cols:
        m.add_regressor(c)

    m.fit(reg)
    return m, df_service, event_cols

def predict(m, df_service: pd.DataFrame, horizon_days: int, use_regressors: bool, scenario: str, scenario_days: int, event_cols: list):
    last_date = df_service["ds"].max()

    if not use_regressors:
        future = m.make_future_dataframe(periods=horizon_days, freq="D")
        fc = m.predict(future)
        return fc, last_date

    # Future dataframe + regressors
    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    future["is_winter"] = future["ds"].dt.month.isin([12, 1, 2]).astype(int)
    for c in event_cols:
        future[c] = 0

    future = add_future_scenario(future, scenario, scenario_days, event_cols, last_date)
    fc = m.predict(future)
    return fc, last_date


df = load_data(DATA_PATH)

st.title("Hospital Resource Forecast")

col1, col2, col3, col4 = st.columns(4)

with col1:
    service = st.selectbox("Service", sorted(df["service"].unique()))

with col2:
    horizon = st.selectbox("Horizon de prévision (jours)", [14, 30, 90], index=1)

with col3:
    use_regressors = st.toggle("Utiliser regressors (hiver + événements)", value=True)

with col4:
    scenario = st.selectbox("Scénario", ["Normal", "Grippe", "COVID-19", "Canicule"])
    scenario_days = st.slider("Durée scénario (jours)", min_value=7, max_value=60, value=30, step=1)

df_s = df[df["service"] == service].sort_values("ds").copy()

# Affichage historique simple
with st.expander("Afficher l’historique (admissions)", expanded=False):
    st.line_chart(df_s.set_index("ds")["y"])

# Entraînement + prédiction
m, df_used, event_cols = fit_prophet_model(df_s, use_regressors)
forecast, last_date = predict(m, df_s, horizon, use_regressors, scenario, scenario_days, event_cols)

start_view = last_date - pd.Timedelta(days=365)  
fc_view = forecast[forecast["ds"] >= start_view].copy()

# KPI
future_only = forecast[forecast["ds"] > last_date].head(horizon).copy()
peak_row = future_only.loc[future_only["yhat"].idxmax()]
peak_date = peak_row["ds"].date()
peak_yhat = float(peak_row["yhat"])

k1, k2, k3 = st.columns(3)
k1.metric("Dernière date historique", str(last_date.date()))
k2.metric(f"Pic prévu (J+{horizon})", f"{peak_yhat:.0f} admissions")
k3.metric("Date du pic", str(peak_date))

# Plot forecast
st.subheader("Prévision (Prophet)")

fig = plt.figure(figsize=(12,4))
plt.plot(df_s["ds"], df_s["y"], label="Historique", alpha=0.6)
plt.plot(forecast["ds"], forecast["yhat"], label="Prévision (yhat)")
plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="Incertitude")
plt.axvline(last_date, linestyle="--", linewidth=1)
plt.title(f"{service} – Scénario: {scenario} – Horizon: {horizon} jours")
plt.xlabel("Date")
plt.ylabel("Admissions (y)")
plt.legend()
st.pyplot(fig)

st.subheader("Alertes & recommandations")
recs = make_recommendations(service, future_only, df_s)
for r in recs:
    st.write("-", r)

st.caption("Note: Les recommandations sont basées sur des règles simples et sur des ratios estimés depuis l’historique.")
