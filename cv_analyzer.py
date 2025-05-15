"""
Mahni.ai – CV skill‑extraction core (UI‑free).
Import this from Streamlit, FastAPI, or unit‑tests.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import torch
from sentence_transformers import SentenceTransformer, util

# ── defaults ────────────────────────────────────────────────────────────────
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_NAME   = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SKILLS_MASTER        = Path("skills_master.json")
SKILLS_CLEAN         = Path("skills_cleaned.json")
SKILLS_EMB           = Path("skills_embeddings.pt")

# ── lazy loaders (only used if caller doesn’t pass objects) ────────────────
def _lazy_embedder(name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(name, device=DEVICE)

def _lazy_assets() -> Dict[str, Any]:
    bilingual      = json.loads(SKILLS_MASTER.read_text(encoding="utf-8"))
    skills_en      = json.loads(SKILLS_CLEAN.read_text(encoding="utf-8"))
    emb            = torch.load(SKILLS_EMB, map_location="cpu").to(DEVICE)
    translations   = {row["en"].strip().title(): row["ar"].strip()
                      for row in bilingual}
    return {"skills_en": skills_en, "emb": emb, "translations": translations}

# ── main API ────────────────────────────────────────────────────────────────
def analyze_cv(
    text: str,
    lang: str = "English",
    *,
    embedder: Optional[SentenceTransformer] = None,
    skills_en: Optional[List[str]]         = None,
    skill_embeddings: Optional[torch.Tensor] = None,
    skill_translations: Optional[Dict[str, str]] = None,
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - skills (list[str]): formatted bilingual strings for display
      - skills_en_for_reco (list[str]): lowercase English skills (for jobs, courses)
      - notes (str): summary string
    """
    if embedder is None:
        embedder = _lazy_embedder()

    if skills_en is None or skill_embeddings is None or skill_translations is None:
        assets            = _lazy_assets()
        skills_en         = assets["skills_en"]
        skill_embeddings  = assets["emb"]
        skill_translations= assets["translations"]

    if not text.strip():
        return {"skills": [], "skills_en_for_reco": [], "notes": "⚠️ Empty CV text."}

    if len(skills_en) != skill_embeddings.shape[0]:
        return {"skills": [], "skills_en_for_reco": [],
                "notes": "⚠️ Mismatch between skills list and embeddings."}

    sentences = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sent_emb  = embedder.encode(sentences, convert_to_tensor=True)
    cos       = util.cos_sim(sent_emb, skill_embeddings).cpu()

    best: Dict[str, float] = {}
    for row in cos:
        for idx, score in enumerate(row):
            score = float(score)
            if score >= threshold and score > best.get(skills_en[idx], 0.0):
                best[skills_en[idx]] = score

    ordered             = sorted(best.items(), key=lambda x: x[1], reverse=True)
    skills_display      = []
    skills_en_lowercase = []
    for en, _ in ordered:
        ar = skill_translations.get(en, "—")
        skills_display.append(f"{ar} / {en}" if lang == "العربية" else f"{en} / {ar}")
        skills_en_lowercase.append(en.lower())

    note = (f"✔️ تم التعرف على {len(skills_display)} مهارة بدقة ≥ {threshold}"
            if lang == "العربية"
            else f"✔️ Detected {len(skills_display)} skills with similarity ≥ {threshold}")

    return {
        "skills": skills_display,
        "skills_en_for_reco": list(set(skills_en_lowercase)),
        "notes": note,
    }

# ── optional CLI test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = Path("sample_cv.txt")
    if sample.exists():
        print(analyze_cv(sample.read_text()))
    else:
        print("Add a sample_cv.txt to test locally.")
