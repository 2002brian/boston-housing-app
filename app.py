import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide")

st.markdown(
    """
    <style>
        .stMetric {
            border: 1px solid #E4E8EE;
            border-radius: 12px;
            padding: 8px;
            background: #F9FBFF;
        }
        .block-container {
            padding-top: 1.2rem;
        }
        h1, h2, h3 {
            letter-spacing: 0.2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/housing.csv")
    if "B" in df.columns:
        df = df.drop(columns=["B"])  # Ethical exclusion per requirement
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    target_col = "MEDV"
    feature_cols = [c for c in df.columns if c != target_col]
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df, feature_cols, target_col


@st.cache_data
def split_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float,
    random_state: int = 42,
):
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@dataclass
class ModelResult:
    model_name: str
    mae: float
    rmse: float
    r2: float
    cv_mean_r2: float
    cv_std_r2: float


def make_model(model_name: str, random_state: int = 42):
    if model_name == "Linear Regression":
        estimator = LinearRegression()
    elif model_name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=400, random_state=random_state, n_jobs=-1
        )
    elif model_name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        estimator = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    elif model_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")
        estimator = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            random_state=random_state,
        )
    elif model_name == "CatBoost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed")
        estimator = CatBoostRegressor(
            iterations=600,
            learning_rate=0.03,
            depth=6,
            random_seed=random_state,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            slice(0, None),
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", estimator),
        ]
    )


@st.cache_resource
def train_and_score_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
):
    model = make_model(model_name, random_state=random_state)

    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=None,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    result = ModelResult(
        model_name=model_name,
        mae=mean_absolute_error(y_test, preds),
        rmse=np.sqrt(mean_squared_error(y_test, preds)),
        r2=r2_score(y_test, preds),
        cv_mean_r2=float(np.mean(cv_scores)),
        cv_std_r2=float(np.std(cv_scores)),
    )
    return model, preds, result


def infer_top_driver(model_pipeline: Pipeline, feature_cols: List[str]) -> str:
    estimator = model_pipeline.named_steps["model"]

    if hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_).flatten()
        if len(coefs) == len(feature_cols):
            return feature_cols[int(np.argmax(np.abs(coefs)))]
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_).flatten()
        if len(importances) == len(feature_cols):
            return feature_cols[int(np.argmax(importances))]

    return ""


@st.cache_resource
def train_torch_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    input_dim: int,
    random_state: int = 42,
    epochs: int = 500,
    lr: float = 0.01,
):
    if torch is None or nn is None:
        raise ImportError("torch is not installed")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X_train.values)
    y_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

    return model, x_scaler, y_scaler


def torch_predict(model, x_scaler, y_scaler, X: pd.DataFrame):
    X_scaled = x_scaler.transform(X.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
    pred = y_scaler.inverse_transform(pred_scaled).flatten()
    return pred


def extract_feature_importance(model_pipeline: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    estimator = model_pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        vals = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        vals = np.abs(estimator.coef_)
    else:
        vals = np.zeros(len(feature_cols))

    return pd.DataFrame({"Feature": feature_cols, "Importance": vals}).sort_values(
        by="Importance", ascending=False
    )


def sidebar_controls(df: pd.DataFrame, feature_cols: List[str]):
    st.sidebar.header("Global Controls")

    st.sidebar.subheader("1) Data Settings")
    train_ratio = st.sidebar.slider(
        "Train Ratio",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.05,
    )
    test_size = 1.0 - train_ratio
    st.sidebar.caption(f"Global test_size used by all models: `{test_size:.2f}`")

    st.sidebar.subheader("2) What-If Simulator")
    sim_values = {}
    for col in feature_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_median = float(df[col].median())

        if col == "CHAS":
            sim_values[col] = float(
                st.sidebar.selectbox("CHAS (Charles River Dummy)", options=[0, 1], index=0)
            )
        else:
            sim_values[col] = st.sidebar.slider(
                col,
                min_value=col_min,
                max_value=col_max,
                value=col_median,
                step=(col_max - col_min) / 200 if col_max > col_min else 0.1,
            )

    single_instance = pd.DataFrame([sim_values])[feature_cols]
    return train_ratio, test_size, single_instance


def show_missing_library_notices():
    missing = []
    if XGBRegressor is None:
        missing.append("xgboost")
    if LGBMRegressor is None:
        missing.append("lightgbm")
    if CatBoostRegressor is None:
        missing.append("catboost")
    if shap is None:
        missing.append("shap")
    if LimeTabularExplainer is None:
        missing.append("lime")
    if torch is None:
        missing.append("torch")

    if missing:
        st.warning(
            "Optional packages missing in this environment: "
            + ", ".join(missing)
            + ". The app will run available functionality."
        )


def main():
    st.title("Boston Housing Price Prediction")
    st.caption(
        "Interactive ML workbench with AutoML leaderboard, explainability, and what-if simulation"
    )

    show_missing_library_notices()

    df_raw = load_data()
    df, feature_cols, target_col = preprocess_data(df_raw.copy())

    train_ratio, test_size, single_instance = sidebar_controls(df, feature_cols)

    X_train, X_test, y_train, y_test = split_dataset(
        df, feature_cols, target_col, test_size=test_size, random_state=42
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "AutoML Leaderboard",
            "Diagnostics",
            "Explainability",
            "Neural Net + What-If",
        ]
    )

    available_models = ["Linear Regression", "Random Forest"]
    if XGBRegressor is not None:
        available_models.append("XGBoost")
    if LGBMRegressor is not None:
        available_models.append("LightGBM")
    if CatBoostRegressor is not None:
        available_models.append("CatBoost")

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = pd.DataFrame()
    if "pred_store" not in st.session_state:
        st.session_state.pred_store = {}

    with tab1:
        st.subheader("Model Comparison")
        st.write(
            "Train/evaluate multiple models using the global split ratio from the sidebar."
        )

        if st.button("Run All Models", type="primary"):
            leaderboard_rows = []
            trained_models = {}
            pred_store = {}

            with st.spinner("Training all models..."):
                for model_name in available_models:
                    model, preds, result = train_and_score_model(
                        model_name, X_train, X_test, y_train, y_test, random_state=42
                    )
                    trained_models[model_name] = model
                    pred_store[model_name] = preds
                    leaderboard_rows.append(
                        {
                            "Model": result.model_name,
                            "MAE": result.mae,
                            "RMSE": result.rmse,
                            "R2": result.r2,
                            "CV Mean R2": result.cv_mean_r2,
                            "CV Std Dev": result.cv_std_r2,
                        }
                    )

            leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by="R2", ascending=False)
            st.session_state.trained_models = trained_models
            st.session_state.leaderboard = leaderboard_df
            st.session_state.pred_store = pred_store

        if not st.session_state.leaderboard.empty:
            st.dataframe(st.session_state.leaderboard, use_container_width=True)

            fig = px.bar(
                st.session_state.leaderboard,
                x="Model",
                y="R2",
                color="R2",
                color_continuous_scale="Tealgrn",
                title="Leaderboard by R2",
            )
            st.plotly_chart(fig, use_container_width=True)

            best_row = st.session_state.leaderboard.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Model", best_row["Model"])
            c2.metric("Best R2", f"{best_row['R2']:.4f}")
            c3.metric("Global Train Ratio", f"{train_ratio:.2f}")
        else:
            st.info("Click 'Run All Models' to generate the leaderboard.")

    with tab2:
        st.subheader("Prediction Diagnostics")

        if st.session_state.leaderboard.empty:
            st.info("Run AutoML Leaderboard first.")
        else:
            model_choice = st.selectbox(
                "Select model", st.session_state.leaderboard["Model"].tolist()
            )
            y_pred = st.session_state.pred_store[model_choice]
            model_perf = st.session_state.leaderboard[
                st.session_state.leaderboard["Model"] == model_choice
            ].iloc[0]

            st.markdown("**Model Performance**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Test R2", f"{model_perf['R2']:.4f}")
            m2.metric("CV Mean R2", f"{model_perf['CV Mean R2']:.4f}")
            m3.metric("CV Std Dev", f"{model_perf['CV Std Dev']:.4f}")

            diag_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            diag_df["Residual"] = diag_df["Actual"] - diag_df["Predicted"]

            f1 = px.scatter(
                diag_df,
                x="Actual",
                y="Predicted",
                trendline="ols",
                title="Actual vs Predicted",
            )
            st.plotly_chart(f1, use_container_width=True)

            f2 = px.histogram(diag_df, x="Residual", nbins=30, title="Residual Distribution")
            st.plotly_chart(f2, use_container_width=True)

            fi_df = extract_feature_importance(
                st.session_state.trained_models[model_choice], feature_cols
            )
            f3 = px.bar(
                fi_df.head(12),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (Model-Specific)",
            )
            f3.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(f3, use_container_width=True)

    with tab3:
        st.subheader("Model Explainability (SHAP + LIME)")

        if st.session_state.leaderboard.empty:
            st.info("Run AutoML Leaderboard first.")
        else:
            model_choice = st.selectbox(
                "Model for explainability",
                st.session_state.leaderboard["Model"].tolist(),
                key="xai_model",
            )
            model_pipe = st.session_state.trained_models[model_choice]

            if shap is None:
                st.warning("Install `shap` to enable SHAP explainability.")
            else:
                st.markdown("**SHAP Summary Plot (test sample)**")
                X_sample = X_test.sample(min(150, len(X_test)), random_state=42)
                X_trans = model_pipe.named_steps["preprocess"].transform(X_sample)
                estimator = model_pipe.named_steps["model"]

                try:
                    if hasattr(estimator, "feature_importances_"):
                        explainer = shap.TreeExplainer(estimator)
                        shap_values = explainer.shap_values(X_trans)
                    else:
                        background = model_pipe.named_steps["preprocess"].transform(
                            X_train.sample(min(80, len(X_train)), random_state=42)
                        )
                        explainer = shap.KernelExplainer(estimator.predict, background)
                        shap_values = explainer.shap_values(X_trans[:40], nsamples=120)
                        X_trans = X_trans[:40]

                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    shap.summary_plot(
                        shap_values,
                        features=X_trans,
                        feature_names=feature_cols,
                        show=False,
                    )
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.error(f"SHAP failed for this model in current environment: {e}")

                st.markdown("**SHAP Waterfall (Current What-If Input)**")
                try:
                    preprocessor = model_pipe.named_steps["preprocess"]
                    estimator = model_pipe.named_steps["model"]

                    X_train_trans = preprocessor.transform(X_train)
                    single_trans = preprocessor.transform(single_instance)

                    if hasattr(estimator, "feature_importances_"):
                        explainer = shap.TreeExplainer(estimator)
                        explanation = explainer(single_trans)
                        single_explanation = shap.Explanation(
                            values=np.array(explanation.values[0]),
                            base_values=float(np.array(explanation.base_values).flatten()[0]),
                            data=np.array(single_trans[0]),
                            feature_names=feature_cols,
                        )
                    elif hasattr(estimator, "coef_"):
                        explainer = shap.LinearExplainer(estimator, X_train_trans)
                        explanation = explainer(single_trans)
                        single_explanation = shap.Explanation(
                            values=np.array(explanation.values[0]),
                            base_values=float(np.array(explanation.base_values).flatten()[0]),
                            data=np.array(single_trans[0]),
                            feature_names=feature_cols,
                        )
                    else:
                        background = X_train_trans[: min(80, len(X_train_trans))]
                        explainer = shap.KernelExplainer(estimator.predict, background)
                        shap_vals = explainer.shap_values(single_trans, nsamples=150)
                        shap_array = np.array(shap_vals)
                        if shap_array.ndim > 1:
                            shap_array = shap_array[0]
                        base_val = float(np.array(explainer.expected_value).flatten()[0])
                        single_explanation = shap.Explanation(
                            values=shap_array,
                            base_values=base_val,
                            data=np.array(single_trans[0]),
                            feature_names=feature_cols,
                        )

                    fig_wf = plt.figure(figsize=(9, 4.8))
                    shap.plots.waterfall(single_explanation, show=False)
                    st.pyplot(fig_wf, clear_figure=True)
                except Exception as e:
                    st.error(f"SHAP waterfall failed for this model in current environment: {e}")

            st.markdown("**LIME Local Explanation**")
            if LimeTabularExplainer is None:
                st.warning("Install `lime` to enable LIME explanations.")
            else:
                try:
                    X_train_trans = model_pipe.named_steps["preprocess"].transform(X_train)
                    X_test_trans = model_pipe.named_steps["preprocess"].transform(X_test)
                    estimator = model_pipe.named_steps["model"]

                    explainer = LimeTabularExplainer(
                        X_train_trans,
                        feature_names=feature_cols,
                        mode="regression",
                    )
                    row_idx = st.slider(
                        "Select test row index",
                        min_value=0,
                        max_value=len(X_test_trans) - 1,
                        value=0,
                    )
                    explanation = explainer.explain_instance(
                        X_test_trans[row_idx],
                        estimator.predict,
                        num_features=min(8, len(feature_cols)),
                    )
                    lime_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Weight"])
                    st.dataframe(lime_df, use_container_width=True)
                except Exception as e:
                    st.error(f"LIME failed for this model in current environment: {e}")

    with tab4:
        st.subheader("Neural Network + What-If Prediction")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**PyTorch Neural Network**")
            if torch is None or nn is None:
                st.warning("Install `torch` to enable neural network training.")
            else:
                if st.button("Train Neural Network"):
                    with st.spinner("Training neural network..."):
                        nn_model, x_scaler, y_scaler = train_torch_regressor(
                            X_train,
                            y_train,
                            input_dim=len(feature_cols),
                            random_state=42,
                            epochs=600,
                            lr=0.01,
                        )
                        st.session_state.nn_model = nn_model
                        st.session_state.nn_x_scaler = x_scaler
                        st.session_state.nn_y_scaler = y_scaler

                        nn_preds = torch_predict(nn_model, x_scaler, y_scaler, X_test)
                        st.session_state.nn_metrics = {
                            "MAE": mean_absolute_error(y_test, nn_preds),
                            "RMSE": np.sqrt(mean_squared_error(y_test, nn_preds)),
                            "R2": r2_score(y_test, nn_preds),
                        }

                if "nn_metrics" in st.session_state:
                    m = st.session_state.nn_metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("NN MAE", f"{m['MAE']:.3f}")
                    c2.metric("NN RMSE", f"{m['RMSE']:.3f}")
                    c3.metric("NN R2", f"{m['R2']:.3f}")

        with col_right:
            st.markdown("**Single-Instance Price Prediction**")
            source_options = []
            if not st.session_state.leaderboard.empty:
                source_options.extend(st.session_state.leaderboard["Model"].tolist())
            if "nn_model" in st.session_state:
                source_options.append("Neural Network")

            if not source_options:
                st.info("Train models first (Run All Models or Train Neural Network).")
            else:
                pred_model = st.selectbox("Prediction model", source_options)

                if pred_model == "Neural Network":
                    pred_value = torch_predict(
                        st.session_state.nn_model,
                        st.session_state.nn_x_scaler,
                        st.session_state.nn_y_scaler,
                        single_instance,
                    )[0]
                else:
                    pred_value = st.session_state.trained_models[pred_model].predict(single_instance)[0]

                st.metric("Predicted MEDV (in $1000s)", f"{pred_value:.2f}")

                indicator = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=float(pred_value),
                        title={"text": "Predicted Home Value"},
                        gauge={
                            "axis": {"range": [0, max(55.0, float(df[target_col].max()))]},
                            "bar": {"color": "#2A9D8F"},
                            "steps": [
                                {"range": [0, 20], "color": "#F4F1DE"},
                                {"range": [20, 35], "color": "#E9C46A"},
                                {"range": [35, 55], "color": "#E76F51"},
                            ],
                        },
                    )
                )
                st.plotly_chart(indicator, use_container_width=True)

                with st.expander("💰 Business & Market Analysis", expanded=True):
                    train_mean_price = float(y_train.mean())
                    if pred_value > train_mean_price:
                        st.success(
                            "📈 Premium Listing: This property is valued above the market average."
                        )
                    elif pred_value < train_mean_price:
                        st.info(
                            "📉 Value Deal: This property is valued below the market average."
                        )
                    else:
                        st.info("This property is valued at the market average.")

                    st.caption(
                        f"Training market average (MEDV, $1000s): {train_mean_price:.2f}"
                    )

                    if pred_model != "Neural Network":
                        top_driver = infer_top_driver(
                            st.session_state.trained_models[pred_model], feature_cols
                        )
                        if top_driver:
                            st.write(
                                f"The biggest factor affecting this price is **{top_driver}**."
                            )
                        else:
                            st.write("The biggest factor affecting this price is not available.")
                    else:
                        st.write(
                            "The biggest factor affecting this price is not available for the neural network model."
                        )


if __name__ == "__main__":
    main()
