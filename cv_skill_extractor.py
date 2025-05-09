from pathlib import Path
from typing import Union, List
import streamlit as st
import fitz  # PyMuPDF
import re
import spacy

# Set page config
st.set_page_config(page_title="CV Skill Extractor", layout="wide")

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.warning(f"Failed to load spaCy model: {e}. Using rule-based fallback.")
        return None

# Initialize spaCy
nlp = load_spacy_model()

# Expanded list of common skills
common_skills = [
    "Python", "Java", "JavaScript", "HTML", "CSS", "SQL", "React", "Angular", "Vue",
    "Node.js", "Django", "Flask", "AWS", "Azure", "Docker", "Kubernetes", "Git",
    "Machine Learning", "AI", "Data Analysis", "Data Science", "Big Data",
    "Project Management", "Agile", "Scrum", "DevOps", "CI/CD", "C++", "C#",
    "Ruby", "PHP", "TypeScript", "Linux", "TensorFlow", "PyTorch", "Excel",
    "Tableau", "Power BI", "MongoDB", "PostgreSQL", "MySQL", "REST API", "GraphQL",
    "Web Development", "Software Engineering", "Cloud Computing", "Cybersecurity",
    "Blockchain", "IoT", "AR/VR", "Mobile Development", "UI/UX Design",
    "Programming", "Coding", "Development", "Analytics", "Networking"
]

def safe_regex_search(text: str, skill: str) -> bool:
    """Safely search for a skill in text using regex."""
    try:
        # Escape special regex characters in the skill name
        escaped_skill = re.escape(skill)
        pattern = rf'\b{escaped_skill}\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    except Exception:
        return False

def extract_skills(text: str) -> List[str]:
    """Extract skills using spaCy NER or fallback to rule-based method."""
    found_skills = set()
    
    # Rule-based extraction
    for skill in common_skills:
        if safe_regex_search(text, skill):
            found_skills.add(skill)
    
    # spaCy NER fallback
    if nlp and not found_skills:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "NORP"] or len(ent.text.split()) <= 3:
                    found_skills.add(ent.text)
        except Exception as e:
            st.error(f"Error during spaCy skill extraction: {e}")
    
    return list(found_skills)

def parse_cv(file: Union[str, Path, bytes]) -> str:
    """Return raw text from a PDF file-like object."""
    try:
        if isinstance(file, (str, Path)):
            doc = fitz.open(str(file))
        else:  # bytes from Streamlit uploader
            doc = fitz.open(stream=file, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        st.error(f"Failed to extract text from CV: {str(e)}")
        return ""

# Streamlit UI
st.title("CV Skill Extractor")
st.write("Upload a PDF CV to extract skills and insights.")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Analyzing CV..."):
        cv_text = parse_cv(uploaded_file)
        if cv_text:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("CV Content")
                st.text_area("Extracted Text", cv_text, height=400)
            
            with col2:
                st.subheader("Analysis Results")
                
                # Extract skills
                extracted_skills = extract_skills(cv_text)
                if extracted_skills:
                    st.write("### Extracted Skills")
                    st.success(f"Found {len(extracted_skills)} skill(s):")
                    for skill in extracted_skills:
                        st.markdown(f"- **{skill}**")
                else:
                    st.info("No skills identified in the CV. Try adding a clear 'Skills' section or using common skill keywords like 'Python', 'SQL', or 'Project Management'.")
                
                # Basic statistics
                st.write("### Basic Statistics")
                st.write(f"- **Total characters**: {len(cv_text)}")
                st.write(f"- **Number of words**: {len(cv_text.split())}")
                st.write(f"- **Number of lines**: {len(cv_text.splitlines())}")
        else:
            st.warning("No text extracted. Ensure your PDF is text-based and not encrypted.")
else:
    st.info("Please upload a PDF CV to begin.")