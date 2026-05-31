"""
Student Dropout Prediction — Interactive Streamlit Demo
Course    : Programming, Data Science & Statistics 2
Author    : Celestin Hakorimana
Instructor: Pr. Ikram Chairi — UM6P

─── Deploy on Streamlit Cloud ────────────────────────────────────────────────
1. Push this folder to GitHub as a new repo
2. Go to https://share.streamlit.io  →  New app
3. Repo: your-username/student-dropout-app
   Branch: main  |  Main file: streamlit_app.py
4. Click Deploy — live in ~2 minutes

─── Run locally ──────────────────────────────────────────────────────────────
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # required on Linux cloud servers — must be before pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.feature_selection  import RFE
from sklearn.decomposition      import PCA
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
CLR_BLUE   = "#4C72B0"
CLR_ORANGE = "#E07B54"
CLR_GREEN  = "#55A868"
CLR_RED    = "#C44E52"

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .section-hdr {
    font-size: 1.3rem; font-weight: 700; color: #1A3A5C;
    border-bottom: 3px solid #4C72B0;
    padding-bottom: 0.3rem; margin-top: 1.2rem; margin-bottom: 0.8rem;
  }
  .pred-grad {
    background:#d4edda; border:2px solid #28a745; border-radius:10px;
    padding:1.2rem; text-align:center; font-size:1.3rem; font-weight:700;
  }
  .pred-drop {
    background:#f8d7da; border:2px solid #dc3545; border-radius:10px;
    padding:1.2rem; text-align:center; font-size:1.3rem; font-weight:700;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA  (cached — downloads once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="📥  Loading dataset …")
def load_data():
    """Load UCI Student Dropout dataset from GitHub mirror or ucimlrepo."""
    URL = (
        "https://raw.githubusercontent.com/"
        "ranga4all1/student-dropout-and-success-prediction/main/data/dataset.csv"
    )
    try:
        df_raw = pd.read_csv(URL, sep=";")
        df_raw.columns = [c.strip() for c in df_raw.columns]
        target_col = next(c for c in df_raw.columns if c.lower() in ("target", "status"))
        source = "UCI Student Dropout (GitHub mirror)"
    except Exception:
        from ucimlrepo import fetch_ucirepo
        raw = fetch_ucirepo(id=697)
        target_col = raw.data.targets.columns[0]
        df_raw = pd.concat([raw.data.features, raw.data.targets], axis=1)
        source = "UCI Student Dropout (ucimlrepo)"

    mask = df_raw[target_col].isin(["Graduate", "Dropout"])
    df   = df_raw[mask].reset_index(drop=True)
    y    = (df[target_col] == "Graduate").astype(int)
    X    = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    return X, y, source


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE  (cached resource — survives reruns)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🤖  Training models … (~30 s first time)")
def train_all(n_samples: int):
    """StandardScaler → RFE (best k by 5-fold CV) → 4 models + PCA 2D."""
    X, y, _ = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    scaler     = StandardScaler()
    Xtr_sc     = scaler.fit_transform(X_train)
    Xte_sc     = scaler.transform(X_test)

    # ── RFE: find best k ─────────────────────────────────────────────────────
    k_range   = range(3, min(X.shape[1] + 1, 16))
    cv_scores = []
    for k in k_range:
        pipe = Pipeline([
            ("rfe", RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                        n_features_to_select=k)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
        cv_scores.append(
            cross_val_score(pipe, Xtr_sc, y_train, cv=5, scoring="accuracy").mean()
        )
    best_k  = list(k_range)[np.argmax(cv_scores)]
    rfe_obj = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                  n_features_to_select=best_k)
    rfe_obj.fit(Xtr_sc, y_train)

    sel    = list(X.columns[rfe_obj.support_])
    Xtr_r  = Xtr_sc[:, rfe_obj.support_]
    Xte_r  = Xte_sc[:, rfe_obj.support_]

    # ── Models ────────────────────────────────────────────────────────────────
    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                      random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                          learning_rate=0.1,
                                                          random_state=RANDOM_STATE),
    }
    results = {}
    for name, m in model_defs.items():
        m.fit(Xtr_r, y_train)
        preds = m.predict(Xte_r)
        probs = m.predict_proba(Xte_r)[:, 1]
        results[name] = dict(
            model=m, preds=preds, probs=probs,
            accuracy=accuracy_score(y_test, preds),
            f1=f1_score(y_test, preds),
            auc=roc_auc_score(y_test, probs),
        )

    # ── PCA 2D ────────────────────────────────────────────────────────────────
    pca2   = PCA(n_components=2, random_state=RANDOM_STATE)
    Xpca2  = pca2.fit_transform(Xtr_r)

    return dict(
        scaler=scaler, rfe=rfe_obj, sel=sel, best_k=best_k,
        X_cols=list(X.columns), X=X, y=y,
        Xtr_sc=Xtr_sc, Xte_sc=Xte_sc,
        Xtr_r=Xtr_r, Xte_r=Xte_r,
        y_train=y_train, y_test=y_test,
        results=results,
        pca2=pca2, Xpca2=Xpca2,
        k_range=list(k_range), cv_scores=cv_scores,
    )


# ── Bootstrap ─────────────────────────────────────────────────────────────────
try:
    X_raw, y_raw, data_source = load_data()
    arts = train_all(len(X_raw))
except Exception as exc:
    st.error(f"**Failed to load data:** {exc}")
    st.info("Check your internet connection and that all packages are installed.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Navigation")
    page = st.radio("", [
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "🔬 Feature Selection (RFE)",
        "📉 PCA Visualisation",
        "🤖 Model Comparison",
        "🎯 Live Prediction",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"**Dataset:** {data_source}")
    st.caption(f"Students: **{len(X_raw):,}**  ·  Features: **{X_raw.shape[1]}**")
    st.caption(f"Graduate: **{y_raw.mean()*100:.1f}%**  ·  "
               f"Dropout: **{(1-y_raw.mean())*100:.1f}%**")
    st.caption(f"RFE selected: **{arts['best_k']} features**")
    st.markdown("---")
    st.caption("*Course: Data Science & Statistics 2*")
    st.caption("*Pr. Ikram Chairi — UM6P*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🎓 Student Dropout Prediction")
    st.markdown("**Final Project · Data Science & Statistics 2 · UM6P · 2026**")
    st.markdown("---")
    st.markdown("""
This app is the **interactive demo** for the final project.
It implements the complete machine learning pipeline from the semester:

| Step | Method | Lab / Session |
|------|--------|--------------|
| 🔎 Feature Selection | Recursive Feature Elimination (RFE) | Lab 1, Lab 3 |
| 📉 Dimensionality Reduction | PCA | Lab 2, Lab 3 |
| 📈 Statistical Model | GLM — Logistic Regression | Lab 7 |
| ⚡ Interaction Effects | Additive vs Interaction GLM | Lab 6 |
| 🌳 Tree Model | CART Decision Tree | Session 7 |
| 🌲 Ensemble | Random Forest + Gradient Boosting | Session 8 |

### Research Question
> *Can we predict whether a student will **drop out or graduate** based on
> demographic, socio-economic, and academic characteristics?*
    """)
    best_name = max(arts["results"], key=lambda k: arts["results"][k]["auc"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students",     f"{len(X_raw):,}")
    c2.metric("Features",     X_raw.shape[1])
    c3.metric("Best AUC",     f"{arts['results'][best_name]['auc']:.3f}", best_name)
    c4.metric("RFE Features", arts["best_k"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown('<div class="section-hdr">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)
    X, y = arts["X"], arts["y"]

    col1, col2 = st.columns(2)
    with col1:
        counts = y.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(["Dropout", "Graduate"], [counts[0], counts[1]],
               color=[CLR_ORANGE, CLR_BLUE], edgecolor="white", width=0.5)
        for i, v in enumerate([counts[0], counts[1]]):
            ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
        ax.set_title("Class Distribution", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        ct = X.corrwith(y).abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = [CLR_BLUE if v > 0.25 else "#A8BFD8" for v in ct.values]
        ax.barh(ct.index[::-1], ct.values[::-1], color=colors[::-1], edgecolor="white")
        ax.axvline(0.25, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Top 10 Feature Correlations with Target", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**Explore a feature distribution by outcome:**")
    feat = st.selectbox("Select feature", X.columns.tolist())
    fig, ax = plt.subplots(figsize=(9, 3))
    for cls, lbl, c in [(0, "Dropout", CLR_ORANGE), (1, "Graduate", CLR_BLUE)]:
        ax.hist(X[y == cls][feat], bins=30, alpha=0.6, label=lbl, density=True, color=c)
    ax.set_title(f"Distribution of '{feat}' by Outcome", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**Correlation heatmap (top 12 features):**")
    corr  = X.corr()
    top12 = corr.abs().mean().nlargest(12).index
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr.loc[top12, top12],
                mask=np.triu(np.ones((12, 12), dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap — Top 12 Features", fontweight="bold")
    st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RFE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Selection (RFE)":
    st.markdown('<div class="section-hdr">🔬 Recursive Feature Elimination (RFE)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**RFE** wraps a Logistic Regression and removes the least important features iteratively. "
        "Optimal `k` is chosen by **5-fold cross-validation** — exactly as in Lab 1 and Lab 3."
    )
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(arts["k_range"], arts["cv_scores"], "o-", color=CLR_BLUE, lw=2)
        ax.axvline(arts["best_k"], color="red", linestyle="--",
                   label=f"Best k = {arts['best_k']}")
        ax.set_xlabel("Number of Features (k)"); ax.set_ylabel("5-fold CV Accuracy")
        ax.set_title("RFE — Choosing Optimal k", fontweight="bold")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        rank_df = pd.DataFrame({
            "feature":  arts["X"].columns,
            "rank":     arts["rfe"].ranking_,
            "selected": arts["rfe"].support_,
        }).sort_values("rank")
        fig, ax = plt.subplots(figsize=(5, max(4, len(rank_df) * 0.25)))
        bar_c = [CLR_BLUE if s else "#C0C8D8" for s in rank_df["selected"]]
        ax.barh(rank_df["feature"][::-1], rank_df["rank"][::-1],
                color=bar_c[::-1], edgecolor="white")
        ax.axvline(1, color="red", linestyle="--", label="Selected (rank = 1)")
        ax.set_xlabel("RFE Rank"); ax.set_title("Feature Rankings", fontweight="bold")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.metric("Optimal k", arts["best_k"])
    st.markdown("**Selected features:**")
    for i, f in enumerate(arts["sel"], 1):
        st.markdown(f"&nbsp;&nbsp;**{i}.** `{f}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PCA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 PCA Visualisation":
    st.markdown('<div class="section-hdr">📉 PCA — Dimensionality Reduction</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**PCA** projects data onto directions of maximum variance. "
        "Scree plot follows the Lab 3 pattern (`plt.bar + plt.step`)."
    )
    col1, col2 = st.columns(2)

    with col1:
        pca_full = PCA()
        pca_full.fit(arts["Xtr_r"])
        evr = pca_full.explained_variance_ratio_
        cum = np.cumsum(evr)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(range(1, len(evr)+1), evr, alpha=0.7, color=CLR_BLUE,
               label="Explained variance")
        ax.step(range(1, len(cum)+1), cum, where="mid", color=CLR_ORANGE, lw=2,
                label="Cumulative")
        ax.axhline(0.90, color="red", linestyle="--", alpha=0.7, label="90% threshold")
        ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Scree Plot", fontweight="bold"); ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        Xpca2, y_train, pca2 = arts["Xpca2"], arts["y_train"], arts["pca2"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for lab, lbl, c in [(0, "Dropout", CLR_ORANGE), (1, "Graduate", CLR_BLUE)]:
            m = y_train.values == lab
            ax.scatter(Xpca2[m, 0], Xpca2[m, 1], c=c, label=lbl,
                       alpha=0.45, s=18, edgecolors="none")
        ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("PCA 2D Projection", fontweight="bold"); ax.legend()
        st.pyplot(fig, use_container_width=True); plt.close()

    k_90 = int(np.argmax(cum >= 0.90)) + 1
    st.info(f"**{k_90} components** explain ≥ 90% of variance.  "
            f"PC1: {evr[0]*100:.1f}%  ·  PC2: {evr[1]*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<div class="section-hdr">🤖 Model Comparison & Evaluation</div>',
                unsafe_allow_html=True)
    results, y_test = arts["results"], arts["y_test"]

    summary = pd.DataFrame([
        {"Model": k, "Accuracy": v["accuracy"], "F1-Score": v["f1"], "ROC-AUC": v["auc"]}
        for k, v in results.items()
    ]).set_index("Model").round(4).sort_values("ROC-AUC", ascending=False)
    st.dataframe(summary.style.highlight_max(axis=0, color="#d4edda"),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    clrs = [CLR_BLUE, CLR_ORANGE, CLR_GREEN, CLR_RED]

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, r), c in zip(results.items(), clrs):
            fpr, tpr, _ = roc_curve(y_test, r["probs"])
            ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} ({r['auc']:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", fontweight="bold"); ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        best_name = summary["ROC-AUC"].idxmax()
        cm = confusion_matrix(y_test, results[best_name]["preds"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Dropout","Graduate"],
                    yticklabels=["Dropout","Graduate"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**Random Forest — Feature Importance:**")
    rf_fi = pd.Series(
        results["Random Forest"]["model"].feature_importances_,
        index=arts["sel"]
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3))
    rf_fi.plot(kind="bar", color=CLR_BLUE, edgecolor="white", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("RF Feature Importance (Gini)", fontweight="bold")
    st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Prediction":
    st.markdown('<div class="section-hdr">🎯 Live Student Dropout Prediction</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Fill in a student profile. All four models will predict their graduation probability."
    )

    X, sel     = arts["X"], arts["sel"]
    scaler     = arts["scaler"]
    rfe_obj    = arts["rfe"]
    results    = arts["results"]
    cols       = arts["X_cols"]

    with st.form("pred_form"):
        st.markdown("#### Student Features")
        input_vals = {}
        rows       = [st.columns(3) for _ in range(-(-len(cols)//3))]   # ceil division
        flat       = [c for row in rows for c in row]

        for i, col_name in enumerate(cols):
            lo, hi, mu = (float(X[col_name].min()),
                          float(X[col_name].max()),
                          float(X[col_name].mean()))
            label = col_name.replace("_", " ")
            with flat[i]:
                if X[col_name].nunique() <= 10:
                    input_vals[col_name] = st.slider(
                        label, int(lo), int(hi), int(round(mu)), key=col_name
                    )
                else:
                    input_vals[col_name] = st.number_input(
                        label, lo, hi, round(mu, 2), key=col_name
                    )

        submitted = st.form_submit_button(
            "🔮 Predict", type="primary", use_container_width=True
        )

    if submitted:
        inp_df  = pd.DataFrame([input_vals])[cols]
        inp_sc  = scaler.transform(inp_df)
        inp_rfe = inp_sc[:, rfe_obj.support_]

        st.markdown("---")
        st.markdown("### Predictions")
        pred_cols = st.columns(len(results))
        for (name, r), col in zip(results.items(), pred_cols):
            prob = r["model"].predict_proba(inp_rfe)[0][1]
            box  = "pred-grad" if prob >= 0.5 else "pred-drop"
            lbl  = "🎓 Graduate" if prob >= 0.5 else "⚠️ Dropout Risk"
            with col:
                st.markdown(
                    f'<div class="{box}">{name}<br>'
                    f'<span style="font-size:1.5rem">{lbl}</span><br>'
                    f'P(Graduate) = <b>{prob*100:.1f}%</b></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### Graduation Probability — All Models")
        names_l = list(results.keys())
        probs_l = [results[m]["model"].predict_proba(inp_rfe)[0][1] for m in names_l]
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.barh(names_l, probs_l,
                color=[CLR_GREEN if p >= 0.5 else CLR_RED for p in probs_l],
                edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold (0.5)")
        ax.set_xlim(0, 1); ax.set_xlabel("P(Graduate)")
        ax.set_title("Model Agreement", fontweight="bold"); ax.legend()
        for i, p in enumerate(probs_l):
            ax.text(min(p + 0.02, 0.91), i, f"{p*100:.1f}%", va="center", fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("#### RFE-selected features used")
        for f, val in zip(sel, inp_rfe[0]):
            st.markdown(f"- **{f}** → raw: `{input_vals[f]}` · standardised: `{val:.3f}`")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Student Dropout Prediction · Final Project · UM6P · 2026 · "
    "Celestin Hakorimana · Pr. Ikram Chairi"
)
