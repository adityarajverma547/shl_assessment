import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("output.csv")
    df = df.dropna(subset=["Job Solution", "Link"])
    return df

df = load_data()

# Combine all relevant fields for text matching
df["combined_text"] = df["Job Solution"].fillna("") + " " + df["Test Types"].fillna("")

# Vectorize the job solutions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined_text"])

# --- Streamlit UI ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommender")
st.write("Enter a job description or query below. We’ll recommend the most relevant **SHL Individual Test Solutions** based on your input.")

query = st.text_input("Enter job description or query...")

if st.button("Recommend"):
    if not query:
        st.warning("Please enter a query.")
    else:
        query_vec = vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, X).flatten()
        top_indices = sim_scores.argsort()[::-1][:10]

        recommendations = df.iloc[top_indices]

        st.subheader("📘 Recommended SHL Assessments")
        for _, row in recommendations.iterrows():
            st.markdown(f"### [{row['Job Solution']}]({row['Link']})")
            st.markdown(f"- **Remote Testing:** {row['Remote Testing']}")
            st.markdown(f"- **Adaptive/IRT:** {row['Adaptive/IRT']}")
            st.markdown(f"- **Duration:** {row['Duration']}")
            st.markdown(f"- **Test Types:** {row['Test Types']}")
            st.markdown("---")
