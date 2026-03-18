import os

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

st.set_page_config(
    page_title="Iris Classifier - ML System",
    page_icon="🌸",
    layout="wide",
)

st.title("Iris Classifier - ML System")


# --- Health check ---
def check_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


# --- Sidebar ---
api_url = st.sidebar.text_input("API URL", value=API_URL)
if api_url != API_URL:
    API_URL = api_url

page = st.sidebar.radio("Navigation", ["Prediction", "Metrics"])

healthy = check_health()
if healthy:
    st.sidebar.success("API: Connected")
else:
    st.sidebar.error("API: Unavailable")

# ==================== Prediction Page ====================
if page == "Prediction":
    st.header("Predict Iris Species")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

    if st.button("Predict", type="primary"):
        if not healthy:
            st.error("API is unavailable. Please check the backend server.")
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"features": [sepal_length, sepal_width, petal_length, petal_width]},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                st.subheader(f"Predicted Species: **{data['prediction'].title()}**")

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ["#2ecc71", "#3498db", "#9b59b6"]
                bars = ax.barh(CLASS_NAMES, data["probability"], color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.bar_label(bars, fmt="%.3f", padding=3)
                fig.tight_layout()
                st.pyplot(fig)

            except requests.RequestException as e:
                st.error(f"API request failed: {e}")

# ==================== Metrics Page ====================
elif page == "Metrics":
    st.header("Model Metrics")

    if not healthy:
        st.error("API is unavailable. Please check the backend server.")
    else:
        try:
            resp = requests.get(f"{API_URL}/metrics", timeout=10)
            resp.raise_for_status()
            metrics = resp.json()

            # Accuracy
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")

            # Classification report as table
            st.subheader("Classification Report")
            report = metrics.get("classification_report", {})
            rows = []
            for label in CLASS_NAMES:
                if label in report:
                    r = report[label]
                    rows.append({
                        "Class": label.title(),
                        "Precision": f"{r['precision']:.3f}",
                        "Recall": f"{r['recall']:.3f}",
                        "F1-Score": f"{r['f1-score']:.3f}",
                        "Support": int(r["support"]),
                    })
            if rows:
                st.table(rows)

            # Confusion matrix heatmap
            cm = metrics.get("confusion_matrix")
            if cm:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4))
                cm_array = np.array(cm)
                im = ax.imshow(cm_array, cmap="Blues")
                ax.set_xticks(range(len(CLASS_NAMES)))
                ax.set_yticks(range(len(CLASS_NAMES)))
                ax.set_xticklabels([n.title() for n in CLASS_NAMES])
                ax.set_yticklabels([n.title() for n in CLASS_NAMES])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                for i in range(len(CLASS_NAMES)):
                    for j in range(len(CLASS_NAMES)):
                        ax.text(j, i, str(cm_array[i, j]),
                                ha="center", va="center",
                                color="white" if cm_array[i, j] > cm_array.max() / 2 else "black")
                fig.colorbar(im)
                fig.tight_layout()
                st.pyplot(fig)

        except requests.RequestException as e:
            st.error(f"Failed to fetch metrics: {e}")
