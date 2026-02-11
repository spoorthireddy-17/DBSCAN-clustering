import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="NYC Taxi Hotspot Detection", layout="wide")

st.title("🚕 NYC Taxi Pickup Hotspot Detection using DBSCAN")

# -----------------------------
# Load dataset directly
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

st.write("Dataset Shape:", df.shape)
st.write(df.head())

# -----------------------------
# Feature Selection
# -----------------------------
x = df[['pickup_latitude', 'pickup_longitude']]

# -----------------------------
# Sampling (avoid memory error)
# -----------------------------
sample_size = st.slider("Select Sample Size", 10000, 100000, 50000, step=10000)
x_sample = x.sample(n=min(sample_size, len(x)), random_state=42)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_sample)

st.success("Data Prepared Successfully ✅")

# -----------------------------
# DBSCAN Experiments
# -----------------------------
eps_values = [0.2, 0.3, 0.5]
results = {}

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(x_scaled)
    
    n_clusters = len(set(labels) - {-1})
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)
    
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        sil_score = silhouette_score(x_scaled[mask], labels[mask])
    else:
        sil_score = None
    
    results[eps] = {
        "labels": labels,
        "clusters": n_clusters,
        "noise": n_noise,
        "noise_ratio": noise_ratio,
        "silhouette": sil_score
    }

# -----------------------------
# Display Evaluation
# -----------------------------
st.subheader("📊 Cluster Evaluation Results")

for eps in eps_values:
    st.write(f"### eps = {eps}")
    st.write("Clusters:", results[eps]["clusters"])
    st.write("Noise Points:", results[eps]["noise"])
    st.write("Noise Ratio:", round(results[eps]["noise_ratio"], 4))
    
    if results[eps]["silhouette"] is not None:
        st.write("Silhouette Score:", round(results[eps]["silhouette"], 4))
    else:
        st.write("Silhouette Score: Not Applicable")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📍 Cluster Visualization")

selected_eps = st.selectbox("Select eps for Visualization", eps_values)
labels = results[selected_eps]["labels"]

fig, ax = plt.subplots(figsize=(8,6))

unique_labels = set(labels)

for label in unique_labels:
    if label == -1:
        ax.scatter(
            x_sample[labels == -1]['pickup_longitude'],
            x_sample[labels == -1]['pickup_latitude'],
            c='black',
            s=5,
            label='Noise'
        )
    else:
        ax.scatter(
            x_sample[labels == label]['pickup_longitude'],
            x_sample[labels == label]['pickup_latitude'],
            s=5,
            label=f'Cluster {label}'
        )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"DBSCAN Clustering (eps={selected_eps})")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Best Model Selection
# -----------------------------
best_eps = None
best_score = -1

for eps in eps_values:
    score = results[eps]["silhouette"]
    if score is not None and score > best_score:
        best_score = score
        best_eps = eps

st.subheader("🏆 Best Model Selection")

if best_eps is not None:
    st.success(f"Best eps value = {best_eps}")
else:
    st.warning("Silhouette score not applicable.")
