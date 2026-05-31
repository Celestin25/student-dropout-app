# 🎓 Student Dropout Prediction

**Final Project · Programming, Data Science & Statistics 2 · UM6P · 2026**
**Instructor: Pr. Ikram Chairi**

A machine learning app that predicts whether a university student will **drop out or graduate**
using the UCI Student Dropout dataset (3,630 students, 36 features).

**Best model:** Gradient Boosting — ROC-AUC > 0.91

---

## 🚀 Deploy on Streamlit Cloud (free, ~2 minutes)

```bash
# 1. Push this folder to a new GitHub repo
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/student-dropout-app.git
git push -u origin main
```

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**
3. Fill in:
   - Repository: `YOUR_USERNAME/student-dropout-app`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Click **Deploy** ✅

---

## 💻 Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📁 Files

```
student-dropout-app/
├── streamlit_app.py       # Main app — all 6 pages
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Theme (navy + orange palette)
└── README.md
```

---

## 👤 Author

Celestin Hakorimana · celestinhakorimana25@gmail.com
