import streamlit as st # type: ignore
import pandas as pd # type: ignore

df = pd.read_csv("data/synthetic/hospital_daily_activity.csv")

st.title("Hospital Resource Forecast")

service = st.selectbox("Service", df["service"].unique())
df_service = df[df["service"] == service]

st.line_chart(df_service.set_index("ds")["y"])
