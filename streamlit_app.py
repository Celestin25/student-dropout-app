"""
Student Dropout Prediction — Professional Streamlit Demo
Academic Machine Learning Project · 2026
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Student Dropout Predictor",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

RS = 42

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
NAVY   = "#0B1F3A"
BLUE   = "#1D6FA4"
LBLUE  = "#3B9DD4"
ACCENT = "#F0A500"
GREEN  = "#0E9F6E"
RED    = "#E02424"
LIGHT  = "#F7FAFC"
MUTED  = "#64748B"
WHITE  = "#FFFFFF"

st.markdown(f"""
<style>
  /* ── Reset & Base ── */
  .block-container {{ padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1400px; }}
  section[data-testid="stSidebar"] {{ background: {NAVY} !important; }}
  section[data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
  section[data-testid="stSidebar"] .stRadio label {{
    color: #94A3B8 !important; font-size: 0.9rem; padding: 0.3rem 0;
  }}
  section[data-testid="stSidebar"] .stRadio label:hover {{ color: {WHITE} !important; }}

  /* ── Page header ── */
  .page-hero {{
    background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
    border-radius: 16px; padding: 2.2rem 2.5rem; margin-bottom: 2rem;
    color: {WHITE};
  }}
  .page-hero h1 {{ font-size: 2rem; font-weight: 800; margin: 0 0 0.4rem 0; color: {WHITE}; }}
  .page-hero p  {{ margin: 0; opacity: 0.75; font-size: 1rem; color: #CBD5E1; }}

  /* ── Section title ── */
  .sec-title {{
    font-size: 1.2rem; font-weight: 700; color: {NAVY};
    border-left: 4px solid {BLUE}; padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
  }}

  /* ── KPI cards ── */
  .kpi-row {{ display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }}
  .kpi-card {{
    flex:1; min-width:140px; background:{WHITE};
    border-radius:12px; padding:1.2rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border-top: 4px solid {BLUE};
  }}
  .kpi-card .kpi-val {{
    font-size: 2rem; font-weight: 800; color: {NAVY}; line-height: 1.1;
  }}
  .kpi-card .kpi-lbl {{
    font-size: 0.78rem; color: {MUTED}; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 0.3rem;
  }}

  /* ── Method table ── */
  .method-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin:1rem 0; }}
  .method-card {{
    background: {WHITE}; border-radius:10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    border-left: 3px solid {LBLUE};
  }}
  .method-card .mc-step {{ font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.1em; color:{MUTED}; margin-bottom:0.25rem; }}
  .method-card .mc-name {{ font-size:0.95rem; font-weight:700; color:{NAVY}; }}
  .method-card .mc-lab  {{ font-size:0.78rem; color:{BLUE}; margin-top:0.2rem; }}

  /* ── Prediction outcome ── */
  .pred-wrapper {{
    display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem;
    margin: 1.5rem 0;
  }}
  .pred-card {{
    border-radius: 14px; padding: 1.4rem 1rem;
    text-align: center; transition: transform 0.15s;
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
  }}
  .pred-card:hover {{ transform: translateY(-2px); }}
  .pred-card .pc-model {{
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; opacity: 0.7; margin-bottom: 0.5rem;
  }}
  .pred-card .pc-verdict {{
    font-size: 1.55rem; font-weight: 900; margin: 0.3rem 0;
  }}
  .pred-card .pc-prob {{
    font-size: 0.9rem; opacity: 0.85; margin-top: 0.4rem;
  }}
  .pred-grad {{
    background: linear-gradient(135deg, #065F46, #059669);
    color: {WHITE};
  }}
  .pred-drop {{
    background: linear-gradient(135deg, #7F1D1D, #DC2626);
    color: {WHITE};
  }}

  /* ── Big outcome banner ── */
  .outcome-banner {{
    border-radius: 16px; padding: 2rem 2.5rem; text-align: center;
    margin: 1.5rem 0; box-shadow: 0 6px 30px rgba(0,0,0,0.15);
  }}
  .outcome-banner.grad {{
    background: linear-gradient(135deg, #064E3B 0%, #059669 100%);
    color: {WHITE};
  }}
  .outcome-banner.drop {{
    background: linear-gradient(135deg, #7F1D1D 0%, #DC2626 100%);
    color: {WHITE};
  }}
  .outcome-banner .ob-label {{
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em;
    opacity: 0.75; margin-bottom: 0.5rem;
  }}
  .outcome-banner .ob-verdict {{
    font-size: 3rem; font-weight: 900; letter-spacing: -0.02em;
  }}
  .outcome-banner .ob-sub {{
    font-size: 1rem; opacity: 0.8; margin-top: 0.5rem;
  }}

  /* ── Gauge track ── */
  .gauge-wrap {{ background:{LIGHT}; border-radius:12px; padding:1.2rem 1.5rem;
    box-shadow:0 1px 4px rgba(0,0,0,0.06); }}
  .gauge-label {{ font-size:0.75rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.08em; color:{MUTED}; margin-bottom:0.6rem; }}
  .gauge-track {{ background:#E2E8F0; border-radius:999px; height:14px;
    overflow:hidden; }}
  .gauge-fill {{ height:100%; border-radius:999px;
    transition: width 0.6s ease; }}

  /* ── Feature pill ── */
  .feat-pill {{
    display:inline-block; background:{LIGHT};
    border:1px solid #CBD5E1; border-radius:8px;
    padding:0.35rem 0.7rem; font-size:0.82rem;
    color:{NAVY}; margin:0.2rem;
  }}
  .feat-pill span {{ color:{BLUE}; font-weight:700; }}

  /* ── Data table override ── */
  .dataframe {{ font-size: 0.85rem !important; }}

  /* ── Sidebar brand ── */
  .sb-brand {{
    font-size: 1.1rem; font-weight: 800; color: {WHITE} !important;
    letter-spacing: -0.01em; margin-bottom: 0.2rem;
  }}
  .sb-tag {{
    font-size: 0.72rem; color: #64748B !important;
    text-transform: uppercase; letter-spacing: 0.1em;
  }}
  .sb-stat {{
    background: rgba(255,255,255,0.07); border-radius:8px;
    padding: 0.5rem 0.75rem; margin: 0.3rem 0; font-size:0.8rem;
  }}
  .sb-stat b {{ color: {WHITE} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    URL = (
        "https://raw.githubusercontent.com/"
        "ranga4all1/student-dropout-and-success-prediction/main/data/dataset.csv"
    )
    try:
        df_raw = pd.read_csv(URL, sep=";")
        df_raw.columns = [c.strip() for c in df_raw.columns]
        tc = next(c for c in df_raw.columns if c.lower() in ("target", "status"))
    except Exception:
        from ucimlrepo import fetch_ucirepo
        raw = fetch_ucirepo(id=697)
        tc  = raw.data.targets.columns[0]
        df_raw = pd.concat([raw.data.features, raw.data.targets], axis=1)

    mask = df_raw[tc].isin(["Graduate", "Dropout"])
    df   = df_raw[mask].reset_index(drop=True)
    y    = (df[tc] == "Graduate").astype(int)
    X    = df.drop(columns=[tc]).select_dtypes(include=[np.number])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models … (first load ~30 s)")
def train_all(n: int):
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=RS, stratify=y)
    sc   = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xte_sc = sc.transform(Xte)

    # RFE best k
    k_range = range(3, min(X.shape[1]+1, 16))
    cvs = []
    for k in k_range:
        p = Pipeline([
            ("rfe", RFE(LogisticRegression(max_iter=1000, random_state=RS),
                        n_features_to_select=k)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RS)),
        ])
        cvs.append(cross_val_score(p, Xtr_sc, ytr, cv=5, scoring="accuracy").mean())

    best_k  = list(k_range)[np.argmax(cvs)]
    rfe_obj = RFE(LogisticRegression(max_iter=1000, random_state=RS),
                  n_features_to_select=best_k)
    rfe_obj.fit(Xtr_sc, ytr)

    sel    = list(X.columns[rfe_obj.support_])
    Xtr_r  = Xtr_sc[:, rfe_obj.support_]
    Xte_r  = Xte_sc[:, rfe_obj.support_]

    mdefs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RS),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=RS),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                      random_state=RS, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                          learning_rate=0.1,
                                                          random_state=RS),
    }
    results = {}
    for name, m in mdefs.items():
        m.fit(Xtr_r, ytr)
        preds = m.predict(Xte_r)
        probs = m.predict_proba(Xte_r)[:, 1]
        results[name] = dict(model=m, preds=preds, probs=probs,
                             accuracy=accuracy_score(yte, preds),
                             f1=f1_score(yte, preds),
                             auc=roc_auc_score(yte, probs))

    pca2  = PCA(n_components=2, random_state=RS)
    Xpca2 = pca2.fit_transform(Xtr_r)

    pca_full = PCA()
    pca_full.fit(Xtr_r)
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)

    return dict(sc=sc, rfe=rfe_obj, sel=sel, best_k=best_k,
                X_cols=list(X.columns), X=X, y=y,
                Xtr_sc=Xtr_sc, Xte_sc=Xte_sc,
                Xtr_r=Xtr_r, Xte_r=Xte_r,
                ytr=ytr, yte=yte,
                results=results,
                pca2=pca2, Xpca2=Xpca2,
                evr=evr, cum=cum,
                k_range=list(k_range), cvs=cvs)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
try:
    X_raw, y_raw = load_data()
    A = train_all(len(X_raw))
except Exception as e:
    st.error(f"**Could not load data:** {e}")
    st.stop()

best_name = max(A["results"], key=lambda k: A["results"][k]["auc"])


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-brand">Student Dropout</div>'
        '<div class="sb-tag">Prediction Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1E3A5F; margin:0.8rem 0'>",
                unsafe_allow_html=True)

    page = st.radio("", [
        "Overview",
        "Data Explorer",
        "Feature Selection",
        "Dimensionality Reduction",
        "Model Performance",
        "Live Prediction",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E3A5F; margin:0.8rem 0'>",
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-stat"><b>{len(X_raw):,}</b> students</div>'
        f'<div class="sb-stat"><b>{X_raw.shape[1]}</b> features</div>'
        f'<div class="sb-stat"><b>{y_raw.mean()*100:.1f}%</b> graduation rate</div>'
        f'<div class="sb-stat"><b>{A["best_k"]}</b> features selected by RFE</div>'
        f'<div class="sb-stat">Best AUC: <b>{A["results"][best_name]["auc"]:.3f}</b></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: clean matplotlib figure
# ─────────────────────────────────────────────────────────────────────────────
def clean_fig(fig, ax=None):
    fig.patch.set_facecolor("none")
    axes = [ax] if ax else fig.axes
    for a in axes:
        a.set_facecolor("none")
        a.spines[["top","right"]].set_visible(False)
        a.tick_params(colors="#475569")
        a.xaxis.label.set_color("#475569")
        a.yaxis.label.set_color("#475569")
        a.title.set_color(NAVY)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(
        '<div class="page-hero">'
        '<h1>Student Dropout Prediction</h1>'
        '<p>A complete machine learning pipeline,from raw data to actionable predictions</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    r = A["results"]
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi-card"><div class="kpi-val">{len(X_raw):,}</div>'
        f'<div class="kpi-lbl">Students</div></div>'
        f'<div class="kpi-card"><div class="kpi-val">{X_raw.shape[1]}</div>'
        f'<div class="kpi-lbl">Raw Features</div></div>'
        f'<div class="kpi-card"><div class="kpi-val">{A["best_k"]}</div>'
        f'<div class="kpi-lbl">RFE Selected</div></div>'
        f'<div class="kpi-card"><div class="kpi-val">{r[best_name]["auc"]:.3f}</div>'
        f'<div class="kpi-lbl">Best AUC ({best_name.split()[0]})</div></div>'
        f'<div class="kpi-card"><div class="kpi-val">{y_raw.mean()*100:.0f}%</div>'
        f'<div class="kpi-lbl">Graduation Rate</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Research question
    st.markdown('<div class="sec-title">Research Question</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{LIGHT}; border-radius:12px; padding:1.2rem 1.5rem; '
        f'border-left:4px solid {ACCENT}; font-size:1.05rem; color:{NAVY}; '
        f'font-style:italic;">'
        f'Can we predict whether a university student will <strong>drop out or graduate</strong> '
        f'based on their demographic background, socio-economic status, and academic '
        f'performance in the first two semesters?'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Methods grid
    st.markdown('<div class="sec-title">Pipeline Methods</div>', unsafe_allow_html=True)
    methods = [
        ("01", "Recursive Feature Elimination", "RFE", "Lab 1 · Lab 3"),
        ("02", "Principal Component Analysis",  "PCA", "Lab 2 · Lab 3"),
        ("03", "t-SNE Visualisation",           "t-SNE", "Lab 5"),
        ("04", "GLM Logistic Regression",        "statsmodels", "Lab 7"),
        ("05", "Interaction Effects",            "LRT · AIC · BIC", "Lab 6"),
        ("06", "CART Decision Tree",             "CV-tuned", "Session 7"),
        ("07", "Random Forest",                  "Bagging · 200 trees", "Session 8"),
        ("08", "Gradient Boosting",              "Sequential · Best AUC", "Session 8"),
    ]
    html = '<div class="method-grid">'
    for step, name, sub, lab in methods:
        html += (f'<div class="method-card">'
                 f'<div class="mc-step">Step {step}</div>'
                 f'<div class="mc-name">{name}</div>'
                 f'<div class="mc-lab">{sub}</div>'
                 f'</div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Dataset info
    st.markdown('<div class="sec-title">Dataset</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f'<div style="background:{LIGHT}; border-radius:12px; padding:1.2rem 1.5rem;">'
            f'<b>Source:</b> UCI Machine Learning Repository — '
            f'<em>Predict Students\' Dropout and Academic Success</em> (ID 697)<br>'
            f'<b>Institution:</b> Portuguese higher-education (single institution)<br>'
            f'<b>Target:</b> Graduate vs Dropout (binary classification)'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        counts = y_raw.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        wedges, texts, autotexts = ax.pie(
            [counts[0], counts[1]],
            labels=["Dropout", "Graduate"],
            colors=[RED, GREEN],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": WHITE, "linewidth": 3},
        )
        for t in autotexts: t.set_color(WHITE); t.set_fontweight("bold")
        ax.set_facecolor("none"); fig.patch.set_facecolor("none")
        st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown(
        '<div class="page-hero"><h1>Data Explorer</h1>'
        '<p>Understand feature distributions, correlations, and class separability</p></div>',
        unsafe_allow_html=True,
    )
    X, y = A["X"], A["y"]

    # Top correlations
    st.markdown('<div class="sec-title">Feature Correlation with Target</div>',
                unsafe_allow_html=True)
    ct = X.corrwith(y).abs().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [GREEN if v > 0.30 else BLUE if v > 0.20 else "#94A3B8" for v in ct.values]
    bars = ax.barh(ct.index[::-1], ct.values[::-1], color=colors[::-1],
                   edgecolor="none", height=0.65)
    ax.axvline(0.30, color=ACCENT, linestyle="--", lw=1.5,
               label="Strong threshold (0.30)")
    ax.axvline(0.20, color=MUTED,  linestyle=":",  lw=1,
               label="Moderate threshold (0.20)")
    for bar, val in zip(bars, ct.values[::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8.5, color=NAVY, fontweight="600")
    ax.set_xlabel("|Pearson Correlation| with Target")
    ax.legend(fontsize=9)
    clean_fig(fig, ax); st.pyplot(fig, use_container_width=True); plt.close()

    # Feature distribution explorer
    st.markdown('<div class="sec-title">Feature Distribution by Outcome</div>',
                unsafe_allow_html=True)
    feat = st.selectbox("Select a feature to explore", X.columns.tolist(),
                        label_visibility="collapsed")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    # Histogram
    for cls, lbl, c in [(0, "Dropout", RED), (1, "Graduate", GREEN)]:
        axes[0].hist(X[y == cls][feat], bins=35, alpha=0.65,
                     label=lbl, density=True, color=c, edgecolor="none")
    axes[0].set_title(f"Density Distribution — {feat}", fontweight="bold", fontsize=11)
    axes[0].legend(fontsize=9)

    # Box plot
    bp = axes[1].boxplot(
        [X[y == 0][feat].values, X[y == 1][feat].values],
        patch_artist=True, notch=True,
        medianprops=dict(color=WHITE, linewidth=2),
    )
    bp["boxes"][0].set_facecolor(RED   + "BB")
    bp["boxes"][1].set_facecolor(GREEN + "BB")
    axes[1].set_xticklabels(["Dropout", "Graduate"])
    axes[1].set_title("Box Plot by Outcome", fontweight="bold", fontsize=11)

    for ax in axes: clean_fig(fig, ax)
    st.pyplot(fig, use_container_width=True); plt.close()

    # Correlation heatmap
    st.markdown('<div class="sec-title">Correlation Heatmap — Top 12 Features</div>',
                unsafe_allow_html=True)
    corr  = X.corr()
    top12 = corr.abs().mean().nlargest(12).index
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(corr.loc[top12, top12],
                mask=np.triu(np.ones((12,12), dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 7.5},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation — Top 12 Features", fontweight="bold", pad=12)
    clean_fig(fig, ax); st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE SELECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Feature Selection":
    st.markdown(
        '<div class="page-hero"><h1>Recursive Feature Elimination</h1>'
        '<p>Identifying the most predictive features through iterative elimination</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="sec-title">Cross-Validation Accuracy vs k</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.fill_between(A["k_range"], A["cvs"], alpha=0.12, color=BLUE)
        ax.plot(A["k_range"], A["cvs"], "o-", color=BLUE, lw=2.5, ms=6)
        ax.axvline(A["best_k"], color=ACCENT, linestyle="--", lw=2,
                   label=f"Optimal k = {A['best_k']}")
        ax.set_xlabel("Number of Features (k)"); ax.set_ylabel("5-fold CV Accuracy")
        ax.set_title("RFE — Selecting Optimal k", fontweight="bold")
        ax.legend(); clean_fig(fig, ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="sec-title">Feature Rankings</div>',
                    unsafe_allow_html=True)
        rank_df = pd.DataFrame({
            "feature":  A["X"].columns,
            "rank":     A["rfe"].ranking_,
            "selected": A["rfe"].support_,
        }).sort_values("rank")
        fig, ax = plt.subplots(figsize=(6, max(4, len(rank_df)*0.26)))
        colors_r = [GREEN if s else "#CBD5E1" for s in rank_df["selected"]]
        ax.barh(rank_df["feature"][::-1], rank_df["rank"][::-1],
                color=colors_r[::-1], edgecolor="none", height=0.7)
        ax.axvline(1, color=ACCENT, linestyle="--", lw=1.5, label="Selected (rank 1)")
        ax.set_xlabel("RFE Rank"); ax.set_title("Feature Rankings", fontweight="bold")
        ax.legend(fontsize=9); clean_fig(fig, ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Selected features pills
    st.markdown('<div class="sec-title">Selected Features</div>', unsafe_allow_html=True)
    pills = "".join(
        f'<span class="feat-pill"><span>{i}.</span> {f}</span>'
        for i, f in enumerate(A["sel"], 1)
    )
    st.markdown(pills, unsafe_allow_html=True)

    # Comparison table
    st.markdown('<div class="sec-title">Baseline vs RFE Comparison</div>',
                unsafe_allow_html=True)
    base_lr = LogisticRegression(max_iter=1000, random_state=RS)
    base_lr.fit(A["Xtr_sc"], A["ytr"])
    acc_base = accuracy_score(A["yte"], base_lr.predict(A["Xte_sc"]))
    rfe_lr   = LogisticRegression(max_iter=1000, random_state=RS)
    rfe_lr.fit(A["Xtr_r"], A["ytr"])
    acc_rfe  = accuracy_score(A["yte"], rfe_lr.predict(A["Xte_r"]))

    cmp = pd.DataFrame({
        "Method":   ["Baseline (all features)", f"RFE ({A['best_k']} features)"],
        "Features": [X_raw.shape[1], A["best_k"]],
        "Accuracy": [f"{acc_base:.4f}", f"{acc_rfe:.4f}"],
        "Reduction": ["—", f"{(1 - A['best_k']/X_raw.shape[1])*100:.0f}% fewer features"],
    })
    st.dataframe(cmp.set_index("Method"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DIMENSIONALITY REDUCTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Dimensionality Reduction":
    st.markdown(
        '<div class="page-hero"><h1>Dimensionality Reduction</h1>'
        '<p>PCA scree analysis and 2D class visualisation</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    evr, cum = A["evr"], A["cum"]

    with col1:
        st.markdown('<div class="sec-title">Scree Plot — Explained Variance</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(1, len(evr)+1), evr*100, alpha=0.75, color=BLUE,
               edgecolor="none", label="Per component")
        ax.step(range(1, len(cum)+1), cum*100, where="mid",
                color=ACCENT, lw=2.5, label="Cumulative")
        ax.axhline(90, color=RED, linestyle="--", lw=1.5, label="90% threshold")
        k90 = int(np.argmax(cum >= 0.90)) + 1
        ax.axvline(k90, color=GREEN, linestyle=":", lw=1.5, label=f"k={k90} (90%)")
        ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance (%)")
        ax.set_title("PCA Scree Plot", fontweight="bold")
        ax.legend(fontsize=8.5); clean_fig(fig, ax)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.info(f"**{k90} components** capture ≥ 90% of variance. "
                f"PC1: {evr[0]*100:.1f}% · PC2: {evr[1]*100:.1f}%")

    with col2:
        st.markdown('<div class="sec-title">2D PCA Projection by Outcome</div>',
                    unsafe_allow_html=True)
        Xpca2, ytr = A["Xpca2"], A["ytr"]
        pca2 = A["pca2"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for lab, lbl, c, m in [(0,"Dropout",RED,"X"),(1,"Graduate",GREEN,"o")]:
            mask = ytr.values == lab
            ax.scatter(Xpca2[mask,0], Xpca2[mask,1],
                       c=c, label=lbl, alpha=0.45, s=20,
                       edgecolors="none", marker=m)
        ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("PCA 2D Projection — Graduate vs Dropout", fontweight="bold")
        ax.legend(fontsize=9, markerscale=1.5); clean_fig(fig, ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Cumulative variance progress bars
    st.markdown('<div class="sec-title">Variance Captured by First N Components</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (nc, label) in enumerate([(1,"PC1"),(2,"PC1-2"),(3,"PC1-3"),(k90,f"PC1-{k90}")]):
        pct = cum[nc-1]*100
        bar_color = GREEN if pct >= 90 else BLUE
        with cols[i]:
            st.markdown(
                f'<div class="gauge-wrap">'
                f'<div class="gauge-label">{label}</div>'
                f'<div style="font-size:1.6rem;font-weight:800;color:{NAVY}">{pct:.1f}%</div>'
                f'<div class="gauge-track"><div class="gauge-fill" '
                f'style="width:{pct:.1f}%;background:{bar_color}"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(
        '<div class="page-hero"><h1>Model Performance</h1>'
        '<p>Comparative evaluation of all four classifiers on the held-out test set</p></div>',
        unsafe_allow_html=True,
    )
    results, yte = A["results"], A["yte"]

    # Summary metrics
    summary = pd.DataFrame([
        {"Model": k, "Accuracy": v["accuracy"],
         "F1-Score": v["f1"], "ROC-AUC": v["auc"]}
        for k, v in results.items()
    ]).set_index("Model").round(4).sort_values("ROC-AUC", ascending=False)

    # KPI cards for best model
    bm = results[best_name]
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi-card" style="border-top-color:{GREEN}">'
        f'<div class="kpi-val">{bm["auc"]:.3f}</div>'
        f'<div class="kpi-lbl">Best ROC-AUC ({best_name.split()[0]})</div></div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-val">{bm["accuracy"]*100:.1f}%</div>'
        f'<div class="kpi-lbl">Best Accuracy</div></div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-val">{bm["f1"]*100:.1f}%</div>'
        f'<div class="kpi-lbl">Best F1-Score</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-title">Full Comparison Table</div>', unsafe_allow_html=True)
    st.dataframe(
        summary.style.highlight_max(axis=0, color="#D1FAE5")
                     .format("{:.4f}"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    clrs = [BLUE, ACCENT, GREEN, RED]

    with col1:
        st.markdown('<div class="sec-title">ROC Curves</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, r), c in zip(results.items(), clrs):
            fpr, tpr, _ = roc_curve(yte, r["probs"])
            ax.plot(fpr, tpr, lw=2.5, color=c, label=f"{name.split()[0]} ({r['auc']:.3f})")
        ax.plot([0,1],[0,1], color="#CBD5E1", lw=1.5, linestyle="--")
        ax.fill_between([0,1],[0,1], alpha=0.04, color=MUTED)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9); clean_fig(fig, ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown(f'<div class="sec-title">Confusion Matrix — {best_name}</div>',
                    unsafe_allow_html=True)
        cm = confusion_matrix(yte, results[best_name]["preds"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.light_palette(GREEN, as_cmap=True),
                    xticklabels=["Dropout","Graduate"],
                    yticklabels=["Dropout","Graduate"],
                    ax=ax, linewidths=2, linecolor=WHITE,
                    annot_kws={"size":14, "weight":"bold"})
        ax.set_xlabel("Predicted", fontweight="600")
        ax.set_ylabel("Actual",    fontweight="600")
        ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
        clean_fig(fig, ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Metric bars
    st.markdown('<div class="sec-title">Metric Comparison</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["Accuracy", "F1-Score", "ROC-AUC"]
    for ax, metric in zip(axes, metrics):
        vals = summary[metric].sort_values(ascending=True)
        bars = ax.barh(range(len(vals)), vals.values,
                       color=[GREEN if n == best_name else BLUE
                               for n in vals.index],
                       edgecolor="none", height=0.55)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([n.split()[0] for n in vals.index], fontsize=9)
        ax.set_xlim(0.6, 1.0)
        ax.set_title(metric, fontweight="bold")
        for bar, v in zip(bars, vals.values):
            ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8.5)
        clean_fig(fig, ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # Feature importance
    st.markdown('<div class="sec-title">Random Forest — Feature Importance</div>',
                unsafe_allow_html=True)
    rf_fi = pd.Series(results["Random Forest"]["model"].feature_importances_,
                      index=A["sel"]).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(3, len(rf_fi)*0.35)))
    colors_fi = [GREEN if v == rf_fi.max() else BLUE for v in rf_fi.values]
    ax.barh(rf_fi.index, rf_fi.values, color=colors_fi, edgecolor="none", height=0.6)
    for i, v in enumerate(rf_fi.values):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_xlabel("Gini Importance"); ax.set_title("Feature Importance — Random Forest",
                                                     fontweight="bold")
    clean_fig(fig, ax); st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Live Prediction":
    st.markdown(
        '<div class="page-hero"><h1>Live Prediction</h1>'
        '<p>Input a student profile and instantly see graduation probability from all four models</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    X       = A["X"]
    sel     = A["sel"]
    scaler  = A["sc"]
    rfe_obj = A["rfe"]
    results = A["results"]
    cols    = A["X_cols"]

    with st.form("pred_form"):
        st.markdown(
            f'<div style="background:{LIGHT}; border-radius:12px; padding:1.2rem 1.5rem; '
            f'margin-bottom:1rem; font-size:0.9rem; color:{MUTED};">'
            f'Adjust the sliders and inputs to match a student\'s profile, '
            f'then click <strong>Run Prediction</strong>.</div>',
            unsafe_allow_html=True,
        )

        input_vals = {}
        n_cols = 4
        rows   = [st.columns(n_cols) for _ in range(-(-len(cols) // n_cols))]
        flat   = [c for row in rows for c in row]

        for i, col_name in enumerate(cols):
            lo   = float(X[col_name].min())
            hi   = float(X[col_name].max())
            mu   = float(X[col_name].mean())
            lbl  = col_name.replace("_", " ")
            with flat[i]:
                if X[col_name].nunique() <= 10:
                    input_vals[col_name] = st.slider(
                        lbl, int(lo), int(hi), int(round(mu)), key=col_name
                    )
                else:
                    input_vals[col_name] = st.number_input(
                        lbl, lo, hi, round(mu, 2), key=col_name
                    )

        submitted = st.form_submit_button(
            "Run Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        inp_df  = pd.DataFrame([input_vals])[cols]
        inp_sc  = scaler.transform(inp_df)
        inp_rfe = inp_sc[:, rfe_obj.support_]

        # Collect all probabilities
        model_probs = {
            name: r["model"].predict_proba(inp_rfe)[0][1]
            for name, r in results.items()
        }
        avg_prob    = np.mean(list(model_probs.values()))
        consensus   = avg_prob >= 0.5
        conf_pct    = max(avg_prob, 1 - avg_prob) * 100

        # ── BIG OUTCOME BANNER ──────────────────────────────────────────────
        verdict    = "WILL GRADUATE" if consensus else "AT RISK OF DROPOUT"
        banner_cls = "grad" if consensus else "drop"
        icon       = "✓" if consensus else "✕"

        st.markdown(
            f'<div class="outcome-banner {banner_cls}">'
            f'<div class="ob-label">Ensemble Consensus Prediction</div>'
            f'<div class="ob-verdict">{icon}  {verdict}</div>'
            f'<div class="ob-sub">Average graduation probability: '
            f'<strong>{avg_prob*100:.1f}%</strong> &nbsp;·&nbsp; '
            f'Model confidence: <strong>{conf_pct:.1f}%</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── INDIVIDUAL MODEL CARDS ───────────────────────────────────────────
        st.markdown('<div class="sec-title">Individual Model Predictions</div>',
                    unsafe_allow_html=True)
        cards_html = '<div class="pred-wrapper">'
        for name, prob in model_probs.items():
            cls     = "pred-grad" if prob >= 0.5 else "pred-drop"
            verdict2 = "Graduate" if prob >= 0.5 else "Dropout Risk"
            icon2    = "✓" if prob >= 0.5 else "✕"
            cards_html += (
                f'<div class="pred-card {cls}">'
                f'<div class="pc-model">{name}</div>'
                f'<div class="pc-verdict">{icon2} {verdict2}</div>'
                f'<div class="pc-prob">P(Graduate) = {prob*100:.1f}%</div>'
                f'</div>'
            )
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # ── PROBABILITY GAUGE BARS ───────────────────────────────────────────
        st.markdown('<div class="sec-title">Graduation Probability — All Models</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3))
        names_l = list(model_probs.keys())
        probs_l = list(model_probs.values())
        bar_c   = [GREEN if p >= 0.5 else RED for p in probs_l]
        bars    = ax.barh(names_l, probs_l, color=bar_c, edgecolor="none", height=0.55)
        ax.axvline(0.5, color=NAVY, linestyle="--", lw=2, label="Decision threshold (0.5)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(Graduate)")
        ax.set_title("Model Agreement on Graduation Probability", fontweight="bold")
        ax.legend(fontsize=9)
        for bar, p in zip(bars, probs_l):
            x_pos = min(p + 0.015, 0.88)
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f"{p*100:.1f}%", va="center", fontsize=11,
                    fontweight="bold", color=NAVY)
        clean_fig(fig, ax)
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── PROBABILITY GAUGES (HTML) ─────────────────────────────────────────
        gcols = st.columns(4)
        for (name, prob), gcol in zip(model_probs.items(), gcols):
            pct      = prob * 100
            bar_col  = GREEN if prob >= 0.5 else RED
            with gcol:
                st.markdown(
                    f'<div class="gauge-wrap">'
                    f'<div class="gauge-label">{name.split()[0]} {name.split()[-1]}</div>'
                    f'<div style="font-size:1.8rem;font-weight:900;color:{bar_col}">'
                    f'{pct:.1f}%</div>'
                    f'<div class="gauge-track"><div class="gauge-fill" '
                    f'style="width:{pct:.1f}%;background:{bar_col}"></div></div>'
                    f'<div style="font-size:0.72rem;color:{MUTED};margin-top:0.3rem">'
                    f'P(Graduate)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── FEATURE VALUES ───────────────────────────────────────────────────
        st.markdown('<div class="sec-title">Key Features Used by Models (RFE-selected)</div>',
                    unsafe_allow_html=True)
        pills = "".join(
            f'<span class="feat-pill"><span>{f.replace("_"," ")}</span> = {input_vals[f]}</span>'
            for f in sel
        )
        st.markdown(pills, unsafe_allow_html=True)
