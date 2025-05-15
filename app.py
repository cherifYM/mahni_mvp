import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="Mahni.ai – CV Analyzer",
    layout="wide"
)

# Load necessary libraries
from pathlib import Path
import tempfile, fitz
from sentence_transformers import SentenceTransformer
from cv_analyzer import analyze_cv

# Initialize SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
import faiss
from job_finder import fetch_jobs
from course_recommender import recommend_courses

# Set up language toggle
lang = st.radio("🌐 اختر اللغة / Choose language", ["English", "العربية"], horizontal=True)

# Set up file upload
uploaded = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

# Define function to extract PDF text
def extract_pdf_text(uploaded_file):
    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    tmp.write_bytes(uploaded_file.read())
    doc  = fitz.open(tmp)
    text = " ".join(p.get_text() for p in doc)
    return text

# Analyze CV and display results
if uploaded:
    cv_text = extract_pdf_text(uploaded)
    result  = analyze_cv(cv_text, lang=lang)

    # Display detected skills
    st.subheader("🧠 " + ("المهارات المكتشفة" if lang == "العربية" else "Detected Skills"))
    if result["skills"]:
        st.success(result["notes"])
        st.write(result["skills"])
    else:
        st.warning("لم يتم العثور على مهارات." if lang == "العربية"
                   else "No skills detected.")

    # Display recommended jobs
    st.subheader("🔎 Live jobs that match your top skills")
    top_skills = result["skills"][:3]
    for sk in top_skills:
        jobs = fetch_jobs(sk, location="Saudi Arabia", lang="en", max_hits=3)
        st.markdown(f"#### 💼 {sk.title()} jobs in KSA")
        if not jobs:
            st.write("No fresh listings.")
        else:
            for j in jobs:
                st.markdown(f"**{j['title']}** — _{j['company']}_  \n"
                            f"{j['where']}  \n{j['snippet']}  \n"
                            f"[Apply]({j['link']})")
                st.markdown("---")

    # Display recommended courses
    st.subheader("🎓 Recommended courses")
    courses = recommend_courses(result["skills"], max_hits=5)
    if not courses:
        st.write("No course suggestions yet.")
    else:
        for c in courses:
            st.markdown(f"**{c['title']}** — {c['platform']}  \n"
                        f"[Open course]({c['url']})")

    # Run FAISS similarity demo
    if result["skills"]:
        if st.button("🔍 تجربة تشابه المهارات" if lang == "العربية"
                     else "🔍 Run similarity demo"):
            skill_names_en = [txt.split("/")[0].strip() for txt in result["skills"]]
            vecs = embedder.encode(skill_names_en)
            index = faiss.IndexFlatL2(vecs.shape[1]); index.add(vecs)
            st.success("Index built ✅")

            q = st.text_input("اكتب مهارة للمقارنة" if lang == "العربية"
                              else "Type a skill to compare")
            if q:
                qvec, _ = faiss.normalize_L2(embedder.encode([q]))
                D, I = index.search(qvec, k=min(3, len(skill_names_en)))
                st.write("🧭 أقرب المهارات:" if lang == "العربية" else "🧭 Closest matches:")
                st.write([result["skills"][i] for i in I[0]])
