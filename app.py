import streamlit as st
import fitz  # PyMuPDF
import spacy
from pathlib import Path
import tempfile
import faiss
from sentence_transformers import SentenceTransformer

# ----- Setup (cache so Streamlit Cloud doesn’t redownload) -----
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # First run inside container – download model
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")
        
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

nlp = load_spacy()
embedder = load_embedder()

# ----- UI -----
st.set_page_config(page_title="Mahni.ai – CV Analyzer", layout="wide")
st.title("📄 Mahni.ai • CV Analyzer MVP")

uploaded = st.file_uploader("Upload your CV (PDF, DOCX coming soon)", type=["pdf"])
if uploaded:
    # Persist file to temp for PyMuPDF
    tmp_path = Path(tempfile.mkstemp(suffix=".pdf")[1])
    tmp_path.write_bytes(uploaded.read())
    
    with st.spinner("🔍 Extracting text..."):
        doc = fitz.open(tmp_path)
        text = " ".join(page.get_text() for page in doc)
    
    st.subheader("Raw text")
    st.text_area("CV contents", text, height=200)
    
    # ----- Basic skill extraction -----
    st.subheader("👇 Detected skills")
    doc_spacy = nlp(text)
    skill_list = [ent.text for ent in doc_spacy.ents if ent.label_ in {"ORG", "PRODUCT", "SKILL"}]
    
    if not skill_list:
        st.warning("No obvious skills found – try a different CV or refine rules.")
    else:
        st.success(f"Found {len(skill_list)} potential skills")
        st.write(skill_list)
    
    # ----- Embedding + FAISS demo (toy search) -----
    if st.button("Build FAISS index and run similarity demo"):
        with st.spinner("Crunching vectors…"):
            vecs = embedder.encode(skill_list)
            index = faiss.IndexFlatL2(vecs.shape[1])
            index.add(vecs)
        st.info("Index built ✅ Type any skill to find nearest match.")
        q = st.text_input("Query skill")
        if q:
            qvec = embedder.encode([q])
            D, I = index.search(qvec, k=3)
            st.write([skill_list[i] for i in I[0]])
