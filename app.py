import streamlit as st
import pandas as pd
import requests
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="GitHub Portfolio Analyzer", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("xgb_model.pkl")
    le = joblib.load("label_encoder.pkl")
    columns = joblib.load("model_columns.pkl")
    means = joblib.load("feature_means.pkl")
    explainer = shap.TreeExplainer(model)
    return model, le, columns, means, explainer

model, le, model_columns, feature_means, explainer = load_assets()

TOKEN = st.secrets["GITHUB_TOKEN"]
headers = {"Authorization": f"token {TOKEN}"}

st.markdown("""
<h1 style='text-align:center;'>🔍 CheckMyGit</h1>
<p style='text-align:center;font-size:20px;'>
AI-powered GitHub Strength Analyzer 🚀
</p>

""", unsafe_allow_html=True)

st.divider()

c1, c2, c3 = st.columns(3)
c1.markdown("### 1️⃣ Enter Profile\nPaste your GitHub link")
c2.markdown("### 2️⃣ AI Analysis\nWe evaluate your activity")
c3.markdown("### 3️⃣ Get Insights\nStrength + improvements")

st.divider()

st.markdown("### 📊 What We Analyze")

f1, f2, f3, f4 = st.columns(4)
f1.info("📦 Repositories")
f2.info("⭐ Stars")
f3.info("👥 Followers")
f4.info("🔥 Activity")

st.divider()

s1, s2, s3 = st.columns(3)
s1.error("Weak → Needs more activity")
s2.warning("Moderate → Good but improvable")
s3.success("Strong → Industry-ready")

st.divider()

sample_url = "https://github.com/torvalds"

if "github_url" not in st.session_state:
    st.session_state.github_url = ""

colB, colC = st.columns([2,1])
with colB:
    if st.button("✨ Try Sample Profile"):
        st.session_state.github_url = sample_url

st.markdown("""
<div style='padding:15px;border-radius:15px;t'>
<h3>Enter your GitHub profile</h3>
</div>
""", unsafe_allow_html=True)

github_url = st.text_input(
    "",
    value=st.session_state.github_url,
    placeholder="https://github.com/your-username"
)

analyze = st.button("🚀 Analyze", use_container_width=True)

def extract_username(url):
    return url.rstrip("/").split("/")[-1]

@st.cache_data
def fetch_data(username):

    user = requests.get(f"https://api.github.com/users/{username}", headers=headers).json()
    repos = requests.get(f"https://api.github.com/users/{username}/repos?per_page=100", headers=headers).json()

    repo_count = len(repos)
    total_stars = sum(r.get("stargazers_count", 0) for r in repos)
    total_forks = sum(r.get("forks_count", 0) for r in repos)

    active_repos = sum(
        (datetime.now() - datetime.strptime(r["pushed_at"], "%Y-%m-%dT%H:%M:%SZ")).days <= 90
        for r in repos if r.get("pushed_at")
    )

    created = datetime.strptime(user["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    account_age = (datetime.now() - created).days

    languages = [r["language"] for r in repos if r.get("language")]
    languages_used = len(set(languages))
    top_language = max(set(languages), key=languages.count) if languages else "None"

    engagement_score = user.get("followers", 0) + total_stars

    row = {
        "public_repos": user.get("public_repos", 0),
        "followers": user.get("followers", 0),
        "following": user.get("following", 0),
        "account_age_days": account_age,
        "repo_count": repo_count,
        "total_stars": total_stars,
        "avg_stars": total_stars / (repo_count + 1),
        "total_forks": total_forks,
        "avg_forks": total_forks / (repo_count + 1),
        "total_issues": 0,
        "active_repos": active_repos,
        "languages_used": languages_used,
        "has_company": 1 if user.get("company") else 0,
        "has_blog": 1 if user.get("blog") else 0,
        "bio_length": len(user.get("bio") or ""),
        "engagement_score": engagement_score,
        "top_language": top_language
    }

    df = pd.DataFrame([row])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    metrics = {
        "Public Repos": row["public_repos"],
        "Followers": row["followers"],
        "Total Stars": total_stars,
        "Active Repos": active_repos
    }

    return df, metrics, user

if analyze and github_url:

    username = extract_username(github_url)

    with st.spinner("Analyzing GitHub profile..."):

        features, metrics, user_raw = fetch_data(username)

        pred = model.predict(features)
        probs = model.predict_proba(features)

        label = le.inverse_transform(pred)[0]
        confidence = probs.max() * 100

        shap_values = explainer(features)
        class_id = pred[0]
        user_shap = shap_values.values[0][:, class_id]

    st.divider()

    profile_name = user_raw.get("name") or username
    profile_url = user_raw.get("html_url")

    st.markdown(f"## {profile_name}")
    st.markdown(f"[🔗 View GitHub Profile]({profile_url})")

    color = {"Weak":"#ff4b4b","Moderate":"#ffa500","Strong":"#00c853"}

    st.markdown(f"""
    <div style="padding:20px;border-radius:20px;background:{color[label]}15;border:2px solid {color[label]};">
    <h2 style="color:{color[label]};">Strength: {label}</h2>
    <h4>Confidence: {confidence:.2f}%</h4>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Public Repos", metrics["Public Repos"])
    c2.metric("Followers", metrics["Followers"])
    c3.metric("Total Stars", metrics["Total Stars"])
    c4.metric("Active Repos", metrics["Active Repos"])

    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame(probs, columns=le.inverse_transform([0,1,2])).T
    st.bar_chart(prob_df)

    shap_df = pd.DataFrame({
        "Feature": features.columns,
        "Impact": user_shap
    }).sort_values(by="Impact", key=abs, ascending=False).head(6)

    st.dataframe(shap_df, use_container_width=True)

    shap.plots.bar(shap_values[0][:, class_id], max_display=5, show=False)
    fig = plt.gcf()
    fig.set_size_inches(6, 2.5)
    st.pyplot(fig)

    st.subheader("💡 Personalized Suggestions")

    for _, row in shap_df.iterrows():
        f = row["Feature"]
        impact = row["Impact"]
        val = features.iloc[0][f]

        if impact < 0:
            st.warning(f"Improve **{f.replace('_',' ')}** (current: {round(val,2)})")
        else:
            st.success(f"Strong **{f.replace('_',' ')}** (current: {round(val,2)})")