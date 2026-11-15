"""
Financial Statement Fraud Detection System
Deep Learning + SHAP Explainability

Run:
    pip install -r requirements.txt
    streamlit run app.py

Optional:
    Keep data_processor.py and financialreport.csv in the same folder.
    If data_processor.py is unavailable, upload a processed CSV containing
    a binary `Fraud` target column in the Data Explorer page.

Important:
    A model risk score is not proof of fraud. Dataset labels and methodology
    must be independently validated before any real-world use.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    from data_processor import process_full_dataset
    DATA_PROCESSOR_AVAILABLE = True
except Exception:
    process_full_dataset = None
    DATA_PROCESSOR_AVAILABLE = False


# ---------------------------------------------------------------------
# App configuration and styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        color: #64B5F6 !important;
        text-align: center;
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: .4rem;
    }
    .sub-header {
        color: #64B5F6 !important;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #B8C7D9 !important;
        margin-bottom: 1.6rem;
    }
    .metric-card {
        background: #E3F2FD !important;
        border: 1px solid #BBDEFB;
        border-radius: 14px;
        padding: 1.25rem .9rem;
        min-height: 190px;
        height: 100%;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,.16);
    }
    .metric-card h3, .metric-card h4, .metric-card p {
        color: #172554 !important;
        opacity: 1 !important;
    }
    .metric-card h3 { font-size: 2rem; margin: .15rem 0 .6rem 0; }
    .metric-card h4 { font-size: 1.05rem; font-weight: 800; margin: .3rem 0; }
    .metric-card p { font-size: .9rem; line-height: 1.45; margin: .35rem 0 0 0; }
    .risk-high {
        background: #FFEBEE;
        color: #B71C1C !important;
        border-left: 5px solid #E53935;
        border-radius: 8px;
        padding: 1rem;
    }
    .risk-low {
        background: #E8F5E9;
        color: #1B5E20 !important;
        border-left: 5px solid #43A047;
        border-radius: 8px;
        padding: 1rem;
    }
    .risk-high h3, .risk-high p, .risk-low h3, .risk-low p {
        color: inherit !important;
    }
    .small-note { color: #9CA3AF; font-size: .85rem; }
    [data-testid="stMetric"] {
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.18);
        padding: 12px;
        border-radius: 10px;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 1.7rem; }
        .metric-card { min-height: 155px; }
    }
</style>
""", unsafe_allow_html=True)

np.random.seed(42)
tf.random.set_seed(42)

TARGET = "Fraud"
EXCLUDED_COLUMNS = {
    "company", "period", "Fraud", "FraudProbability",
    "Fraud_Probability", "Prediction", "Risk_Level"
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def clean_feature_frame(frame, feature_names=None):
    """Keep numeric model features, preserve training column order."""
    data = frame.copy()
    if feature_names is not None:
        missing = [c for c in feature_names if c not in data.columns]
        if missing:
            raise ValueError(
                "Uploaded file is missing required model features: "
                + ", ".join(missing[:15])
                + (" ..." if len(missing) > 15 else "")
            )
        data = data[feature_names]
    else:
        data = data.select_dtypes(include=[np.number])
        data = data[[c for c in data.columns if c not in EXCLUDED_COLUMNS]]

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def load_local_dataset():
    """Use the project's data_processor if available, otherwise local CSV."""
    if DATA_PROCESSOR_AVAILABLE:
        return process_full_dataset()
    for candidate in ("financialreport.csv", "financial_report.csv", "financial_reports.csv"):
        try:
            return pd.read_csv(candidate)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Could not find data_processor.py or a local financialreport.csv. "
        "Upload a processed CSV containing a Fraud column instead."
    )


def safe_binary_target(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError("The Fraud target contains missing or non-numeric values.")
    unique = set(values.unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError("Fraud target must contain only 0 and 1 (0=Non-Fraud, 1=Fraud).")
    if len(unique) < 2:
        raise ValueError("The Fraud target must contain both classes: 0 and 1.")
    return values.astype("int32").to_numpy()


def make_model(input_dim, learning_rate=0.001):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.15),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


def train_detector(df, test_fraction, validation_fraction, epochs,
                   batch_size, learning_rate, patience, use_class_weights,
                   threshold):
    """Split first; fit imputer/scaler on training data only (no leakage)."""
    if TARGET not in df.columns:
        raise ValueError("Dataset must contain a 'Fraud' target column.")

    y = safe_binary_target(df[TARGET])
    X = clean_feature_frame(df.drop(columns=[TARGET]))

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns were found.")
    if len(X) < 20:
        raise ValueError("Dataset is too small for a reliable train/test split.")
    if X.isna().all().any():
        all_missing = X.columns[X.isna().all()].tolist()
        raise ValueError("These columns contain no usable values: " + ", ".join(all_missing[:10]))

    # Hold out test set before fitting any preprocessing objects.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=y
    )

    # validation_fraction is the fraction of the remaining training set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=validation_fraction,
        random_state=42, stratify=y_train_full
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    model = make_model(X_train_scaled.shape[1], learning_rate)

    class_weight = None
    if use_class_weights:
        counts = np.bincount(y_train, minlength=2)
        class_weight = {
            0: len(y_train) / (2 * max(counts[0], 1)),
            1: len(y_train) / (2 * max(counts[1], 1))
        }

    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(3, patience // 3), min_lr=1e-6, verbose=0
        )
    ]

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callback_list,
        verbose=0
    )

    probabilities = model.predict(X_test_scaled, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(int)

    results = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "history": history,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "y_train": y_train,
        "y_val": y_val,
        "threshold": threshold,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities)
                if len(np.unique(y_test)) == 2 else np.nan,
            "PR-AUC": average_precision_score(y_test, probabilities)
        }
    }
    return results


def prepare_uploaded_features(frame, detector):
    """Validate full feature schema and apply fitted preprocessing."""
    X = clean_feature_frame(frame, detector["feature_names"])
    X_imp = detector["imputer"].transform(X)
    return detector["scaler"].transform(X_imp)


def plot_training_history(history):
    hist = history.history
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "Accuracy", "AUC", "Precision / Recall")
    )
    for key, label, row, col in [
        ("loss", "Training", 1, 1),
        ("accuracy", "Training", 1, 2),
        ("auc", "Training", 2, 1),
        ("precision", "Training", 2, 2)
    ]:
        if key in hist:
            fig.add_trace(go.Scatter(y=hist[key], name=f"{label} {key}"), row=row, col=col)
        val_key = "val_" + key
        if val_key in hist:
            fig.add_trace(go.Scatter(y=hist[val_key], name=f"Validation {key}"), row=row, col=col)
    fig.update_layout(height=620, title="Training History", template="plotly_dark")
    fig.update_xaxes(title_text="Epoch")
    return fig


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Non-Fraud", "Predicted Fraud"],
        y=["Actual Non-Fraud", "Actual Fraud"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
        template="plotly_dark",
        height=480
    )
    return fig


def plot_roc(y_true, probabilities):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    score = roc_auc_score(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC curve (AUC={score:.3f})",
        fill="tozeroy"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random baseline", line=dict(dash="dash")
    ))
    fig.update_layout(
        title="ROC Curve", xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate", template="plotly_dark",
        height=450
    )
    return fig


def calculate_shap_values(detector, sample_count=40):
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not installed. Run: pip install shap")
    background = detector["X_train_scaled"]
    explain_data = detector["X_test_scaled"][:sample_count]
    background = background[:min(50, len(background))]
    # KernelExplainer is slower but works with a broad range of Keras versions.
    explainer = shap.KernelExplainer(
        lambda values: detector["model"].predict(
            np.asarray(values, dtype=np.float32), verbose=0
        ).reshape(-1),
        background
    )
    values = explainer.shap_values(explain_data, nsamples=100)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values.squeeze()
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values, explain_data


def show_risk_result(probability, threshold):
    st.subheader("Prediction Result")
    st.metric("Model risk score", f"{probability:.1%}")
    if probability >= threshold:
        st.markdown(
            f"""<div class="risk-high">
            <h3>⚠️ Higher model-risk score — review recommended</h3>
            <p>Score: <strong>{probability:.2%}</strong>. Review the underlying
            financial records and supporting evidence.</p></div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class="risk-low">
            <h3>Lower model-risk score</h3>
            <p>Score: <strong>{probability:.2%}</strong>. A lower score does not
            prove that the financial statement is legitimate.</p></div>""",
            unsafe_allow_html=True
        )
    st.caption(
        "This is a model-generated screening score, not a finding of fraud. "
        "It depends on the dataset labels, feature quality and model calibration."
    )


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = None
if "detector" not in st.session_state:
    st.session_state.detector = None


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.markdown(
    "<h2 style='text-align:center;color:#64B5F6'>💰 Fraud Detection</h2>",
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Explorer",
        "🤖 Model Training",
        "🔍 SHAP Explainability",
        "🎯 Prediction"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Financial Statement Fraud Detection")
st.sidebar.caption("Deep Learning + SHAP Explainability")


# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        "<h1 class='main-header'>Financial Statement Fraud Detection System</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 class='hero-subtitle'>Deep Learning + SHAP Explainability</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center'>Screen financial statements for potential "
        "anomalies using neural networks and explainable AI.</p>",
        unsafe_allow_html=True
    )

    card_cols = st.columns(4)
    cards = [
        ("🧠", "Deep Learning", "Neural network for binary classification"),
        ("📊", "SHAP XAI", "Explain the factors behind model predictions"),
        ("📈", "Financial Data", "Explore financial statements and ratios"),
        ("🎯", "Risk Scoring", "Screen records and review model-risk scores")
    ]
    for col, (emoji, title, description) in zip(card_cols, cards):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                <h3>{emoji}</h3><h4>{title}</h4><p>{description}</p>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
    st.info(
        "Dataset reference: [Financial Reports Fraud Detection Data on Kaggle]"
        "(https://www.kaggle.com/datasets/ziya07/financial-reports-fraud-detection-data)\n\n"
        "Load the dataset in Data Explorer. The app can use the optional "
        "`data_processor.py` pipeline or an uploaded processed CSV."
    )

    st.markdown("<h2 class='sub-header'>Model Architecture</h2>", unsafe_allow_html=True)
    st.code(
        "Input features\n"
        "   ↓\n"
        "Dense(128) + BatchNorm + Dropout\n"
        "   ↓\n"
        "Dense(64) + BatchNorm + Dropout\n"
        "   ↓\n"
        "Dense(32) + Dropout\n"
        "   ↓\n"
        "Sigmoid output: model score from 0 to 1\n\n"
        "Training: Adam + Binary Cross-Entropy + Early Stopping\n"
        "Evaluation: Precision, Recall, F1, ROC-AUC and PR-AUC",
        language="text"
    )
    st.warning(
        "Important: do not describe the output as proof of fraud. "
        "It is a screening score based on the labels and patterns in the training data."
    )


# ---------------------------------------------------------------------
# DATA EXPLORER
# ---------------------------------------------------------------------
elif page == "📊 Data Explorer":
    st.markdown("<h1 class='main-header'>📊 Data Explorer</h1>", unsafe_allow_html=True)
    source = st.radio(
        "Choose data source",
        ["Load Local Dataset", "Upload CSV"],
        horizontal=True
    )

    if source == "Load Local Dataset":
        if st.button("Load Financial Dataset", type="primary"):
            try:
                with st.spinner("Loading dataset..."):
                    st.session_state.data = load_local_dataset()
                st.success(
                    f"Loaded {len(st.session_state.data):,} rows and "
                    f"{len(st.session_state.data.columns):,} columns."
                )
            except Exception as exc:
                st.error(str(exc))
    else:
        uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"], key="data_upload")
        if uploaded is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded)
                st.success("CSV uploaded successfully.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    df = st.session_state.data
    if df is not None:
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Fraud cases", f"{int(pd.to_numeric(df[TARGET], errors='coerce').sum()):,}"
                  if TARGET in df.columns else "Not available")
        c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        st.markdown("### Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### Missing Values")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(25)
        if len(missing):
            st.plotly_chart(
                px.bar(
                    x=missing.values, y=missing.index, orientation="h",
                    labels={"x": "Missing values", "y": "Column"},
                    title="Columns with the most missing values"
                ),
                use_container_width=True
            )
        else:
            st.success("No missing values detected.")

        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns):
            st.markdown("### Numeric Feature Distributions")
            selected = st.multiselect(
                "Choose numeric columns",
                numeric.columns.tolist(),
                default=numeric.columns.tolist()[:3]
            )
            if selected:
                selected_col = st.selectbox("View distribution for", selected)
                st.plotly_chart(
                    px.histogram(df, x=selected_col, title=f"Distribution: {selected_col}"),
                    use_container_width=True
                )

        if TARGET in df.columns:
            st.markdown("### Target Distribution")
            target_counts = df[TARGET].value_counts(dropna=False).rename_axis("Fraud label").reset_index(name="Count")
            st.plotly_chart(
                px.bar(target_counts, x="Fraud label", y="Count", title="Fraud label counts"),
                use_container_width=True
            )


# ---------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------
elif page == "🤖 Model Training":
    st.markdown("<h1 class='main-header'>🤖 Model Training</h1>", unsafe_allow_html=True)
    df = st.session_state.data

    if df is None:
        st.warning("Load or upload a dataset first in Data Explorer.")
    elif TARGET not in df.columns:
        st.error("The dataset must contain a binary 'Fraud' column (0 or 1).")
    else:
        st.markdown("<h2 class='sub-header'>Training Parameters</h2>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a:
            test_size = st.slider("Test set (%)", 10, 40, 20) / 100
            epochs = st.slider("Maximum epochs", 10, 200, 80)
        with b:
            batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)
            learning_rate = st.select_slider(
                "Learning rate",
                options=[0.0001, 0.0005, 0.001, 0.005],
                value=0.001
            )
        with c:
            validation_size = st.slider("Validation split (%) of training data", 10, 30, 20) / 100
            threshold = st.slider("Classification threshold", 0.10, 0.90, 0.50, 0.05)

        with st.expander("Advanced settings"):
            use_weights = st.checkbox("Use automatically calculated class weights", value=True)
            patience = st.slider("Early stopping patience", 5, 30, 12)

        st.info(
            "Preprocessing is fitted only on the training split. This prevents "
            "test-set information from leaking into imputation and scaling."
        )

        if st.button("🚀 Train Model", type="primary"):
            try:
                with st.spinner("Training neural network. Please wait..."):
                    trained = train_detector(
                        df=df,
                        test_fraction=test_size,
                        validation_fraction=validation_size,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        patience=patience,
                        use_class_weights=use_weights,
                        threshold=threshold
                    )
                    st.session_state.detector = trained
                st.success("Training completed successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        detector = st.session_state.detector
        if detector is not None:
            st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
            metric_cols = st.columns(5)
            for col, (name, value) in zip(metric_cols, detector["metrics"].items()):
                col.metric(name, f"{value:.4f}" if pd.notna(value) else "N/A")

            st.plotly_chart(plot_training_history(detector["history"]), use_container_width=True)
            st.plotly_chart(
                plot_confusion_matrix(detector["y_test"], detector["predictions"]),
                use_container_width=True
            )
            st.plotly_chart(
                plot_roc(detector["y_test"], detector["probabilities"]),
                use_container_width=True
            )

            st.markdown("### Classification Report")
            report = classification_report(
                detector["y_test"],
                detector["predictions"],
                labels=[0, 1],
                target_names=["Non-Fraud", "Fraud"],
                output_dict=True,
                zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            st.caption(
                f"Features used: {len(detector['feature_names'])}. "
                f"Decision threshold: {detector['threshold']:.2f}."
            )


# ---------------------------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------------------------
elif page == "🔍 SHAP Explainability":
    st.markdown("<h1 class='main-header'>🔍 SHAP Explainability</h1>", unsafe_allow_html=True)
    detector = st.session_state.detector

    if detector is None:
        st.warning("Train a model first in Model Training.")
    elif not SHAP_AVAILABLE:
        st.error("SHAP is not installed. Run: pip install shap")
    else:
        st.write(
            "SHAP estimates how each input feature contributes to the model output. "
            "Positive values push the model score upward; negative values push it downward."
        )
        sample_count = st.slider(
            "Number of test records to explain",
            1, min(50, len(detector["X_test_scaled"])),
            min(20, len(detector["X_test_scaled"]))
        )

        if st.button("Generate SHAP Explanation", type="primary"):
            try:
                with st.spinner("Calculating SHAP values. This may take some time..."):
                    shap_values, explain_data = calculate_shap_values(detector, sample_count)

                mean_abs = np.abs(shap_values).mean(axis=0)
                importance = pd.DataFrame({
                    "Feature": detector["feature_names"],
                    "Mean absolute SHAP": mean_abs
                }).sort_values("Mean absolute SHAP", ascending=False)

                st.markdown("### Global Feature Importance")
                st.plotly_chart(
                    px.bar(
                        importance.head(20).sort_values("Mean absolute SHAP"),
                        x="Mean absolute SHAP",
                        y="Feature",
                        orientation="h",
                        title="Top 20 features by mean absolute SHAP value"
                    ),
                    use_container_width=True
                )
                st.dataframe(importance.head(30), use_container_width=True)

                st.markdown("### Individual Record Explanation")
                record_index = st.number_input(
                    "Choose a record from the explained sample",
                    min_value=0,
                    max_value=sample_count - 1,
                    value=0,
                    step=1
                )
                individual = pd.DataFrame({
                    "Feature": detector["feature_names"],
                    "SHAP value": shap_values[int(record_index)],
                    "Scaled feature value": explain_data[int(record_index)]
                })
                individual["Absolute impact"] = individual["SHAP value"].abs()
                individual = individual.sort_values("Absolute impact", ascending=False).head(20)

                st.plotly_chart(
                    px.bar(
                        individual.sort_values("SHAP value"),
                        x="SHAP value",
                        y="Feature",
                        orientation="h",
                        color="SHAP value",
                        color_continuous_scale="RdBu",
                        title="Top feature contributions for selected record"
                    ),
                    use_container_width=True
                )
                st.dataframe(individual, use_container_width=True)
                st.caption(
                    "SHAP values explain model behaviour; they do not establish "
                    "that a feature caused fraud."
                )
            except Exception as exc:
                st.error(f"SHAP analysis failed: {exc}")


# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------
elif page == "🎯 Prediction":
    st.markdown("<h1 class='main-header'>🎯 Fraud Risk Screening</h1>", unsafe_allow_html=True)
    detector = st.session_state.detector

    if detector is None:
        st.warning("Train a model first in Model Training.")
    else:
        st.info(
            "Upload a CSV containing all features used during training. "
            "The app will reject missing required columns instead of silently "
            "filling them with zero."
        )
        mode = st.radio(
            "Prediction mode",
            ["Single record (CSV)", "Batch prediction (CSV)"],
            horizontal=True
        )

        uploaded = st.file_uploader(
            "Upload CSV with the model's required feature columns",
            type=["csv"],
            key="prediction_upload"
        )

        if uploaded is not None:
            try:
                prediction_df = pd.read_csv(uploaded)
                st.markdown("### Uploaded data preview")
                st.dataframe(prediction_df.head(10), use_container_width=True)

                if st.button("🔍 Run Prediction", type="primary"):
                    X_scaled = prepare_uploaded_features(prediction_df, detector)
                    probs = detector["model"].predict(X_scaled, verbose=0).ravel()
                    output = prediction_df.copy()
                    output["Fraud_Probability"] = probs
                    output["Prediction"] = np.where(
                        probs >= detector["threshold"], "Review Required", "Lower Model Risk"
                    )
                    output["Risk_Level"] = pd.cut(
                        probs,
                        bins=[-0.000001, 0.30, 0.50, 0.70, 1.000001],
                        labels=["Low", "Low-Medium", "Medium-High", "High"]
                    )

                    if mode == "Single record (CSV)":
                        if len(output) != 1:
                            st.error("For single-record mode, upload a CSV containing exactly one row.")
                        else:
                            show_risk_result(float(probs[0]), detector["threshold"])
                            st.markdown("### Record details")
                            st.dataframe(output, use_container_width=True)
                    else:
                        st.markdown("### Batch prediction results")
                        st.dataframe(output, use_container_width=True)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total records", len(output))
                        m2.metric(
                            "Records flagged for review",
                            int((probs >= detector["threshold"]).sum())
                        )
                        m3.metric("Average model score", f"{probs.mean():.2%}")

                        st.plotly_chart(
                            px.histogram(
                                output,
                                x="Fraud_Probability",
                                nbins=20,
                                title="Distribution of model-risk scores"
                            ),
                            use_container_width=True
                        )

                    csv_bytes = output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download prediction results",
                        data=csv_bytes,
                        file_name="fraud_prediction_results.csv",
                        mime="text/csv"
                    )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Financial Fraud Detection System")
st.sidebar.caption("For educational and screening purposes only.")