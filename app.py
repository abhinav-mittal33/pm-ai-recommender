# app.py
import os
import re
import tempfile
import traceback
from collections import Counter
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect

# Embedding model (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False

# PDF/DOCX readers
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# Translator (try deep_translator, then googletrans as fallback)
TRANSLATOR = None
HAS_TRANSLATOR = False
try:
    # deep-translator wrapper
    from deep_translator import GoogleTranslator as DeepGoogleTranslator

    def _deep_translate(text, target):
        if not text:
            return text
        try:
            return DeepGoogleTranslator(source="auto", target=target).translate(text)
        except Exception:
            return text

    TRANSLATOR = _deep_translate
    HAS_TRANSLATOR = True
except Exception:
    try:
        from googletrans import Translator as GTranslator

        _g = GTranslator()

        def _gtrans(text, target):
            if not text:
                return text
            try:
                out = _g.translate(text, dest=target)
                return out.text
            except Exception:
                return text

        TRANSLATOR = _gtrans
        HAS_TRANSLATOR = True
    except Exception:
        TRANSLATOR = None
        HAS_TRANSLATOR = False

app = Flask(__name__)
app.secret_key = "replace-with-secure-key"

# ---------------- CONFIG ----------------
DATA_FILE = os.path.join("data", "internships.csv")  # CSV path
MODELS_DIR = "models"
LOCAL_MODEL_DIR = os.path.join(MODELS_DIR, "sbert-local")  # optional local model path
EMBEDDING_CACHE = os.path.join(MODELS_DIR, "internships_emb.npy")
MODEL_NAME = os.environ.get("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = 5

# UI supported languages
SUPPORTED_LANGS = ["en", "hi", "ta", "gu", "bn", "mr", "te", "kn", "ml", "ur", "pa"]

# ---------------- Load internships CSV ----------------
def read_internships():
    if not os.path.exists(DATA_FILE):
        cols = ["id","title","company","domain","location","mode","duration_months","stipend","skills_required","description","apply_link"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(DATA_FILE, dtype=str).fillna("")
    return df

internships = read_internships()

# auto-detect skills column
POSSIBLE_SKILL_COLS = ["skills", "skills_required", "skills_required_text", "skills_required;skills"]
skill_col = None
for c in POSSIBLE_SKILL_COLS:
    if c in internships.columns:
        skill_col = c
        break
if not skill_col:
    for c in internships.columns:
        if "skill" in c.lower():
            skill_col = c
            break
if not skill_col:
    internships["skills_required"] = ""
    skill_col = "skills_required"

app.logger.info("Skills column: %s", skill_col)
app.logger.info("Loaded internships: %d", len(internships))

# ---------------- Model loading & corpus embeddings ----------------
model = None
corpus_rows = []
corpus_embeddings = None

def try_load_model():
    global model
    if not EMBEDDINGS_AVAILABLE:
        app.logger.warning("sentence-transformers not available; embeddings disabled.")
        return None
    # Try to load local model first
    if os.path.exists(LOCAL_MODEL_DIR):
        try:
            app.logger.info("Loading local model from %s", LOCAL_MODEL_DIR)
            model = SentenceTransformer(LOCAL_MODEL_DIR)
            return model
        except Exception as e:
            app.logger.warning("Local model load failed: %s", e)
    # Fallback to HF model
    try:
        app.logger.info("Loading model: %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        app.logger.error("Failed to load SBERT model: %s", e)
        model = None
        return None

def build_corpus_embeddings(df):
    global corpus_rows, corpus_embeddings
    corpus_rows = []
    corpus = []
    for _, row in df.iterrows():
        skills_text = str(row.get(skill_col,""))
        doc = " ".join([str(row.get("title","")), str(row.get("domain","")), skills_text, str(row.get("description",""))])
        corpus.append(doc)
        corpus_rows.append(row)
    if not corpus:
        corpus_embeddings = None
        return
    # Try load cached embeddings
    if model is not None and os.path.exists(EMBEDDING_CACHE):
        try:
            emb = np.load(EMBEDDING_CACHE, allow_pickle=False)
            corpus_embeddings = emb
            app.logger.info("Loaded precomputed embeddings (%s)", EMBEDDING_CACHE)
            return
        except Exception:
            app.logger.warning("Failed to load cached embeddings, will compute.")
    # compute embeddings (may take time)
    if model is not None:
        app.logger.info("Encoding corpus (%d items)...", len(corpus))
        try:
            corpus_embeddings = model.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
            # store numpy cache
            os.makedirs(os.path.dirname(EMBEDDING_CACHE), exist_ok=True)
            np.save(EMBEDDING_CACHE, np.array(corpus_embeddings))
            app.logger.info("Corpus embeddings computed & cached.")
        except Exception as e:
            app.logger.error("Failed to compute corpus embeddings: %s", e)
            corpus_embeddings = None
    else:
        corpus_embeddings = None

# Try loading model + embeddings at startup
try_load_model()
build_corpus_embeddings(internships)

# ---------------- Skill extraction ----------------
BASE_SKILLS = [
    "python","java","c++","c","sql","pandas","numpy","tensorflow","pytorch",
    "scikit-learn","machine learning","deep learning","nlp","computer vision",
    "react","node.js","javascript","html","css","docker","kubernetes","aws","azure",
    "gcp","devops","flutter","android","ios","swift","matlab","autocad","solidworks",
    "research","marketing","finance","excel","tableau","powerbi","blockchain","law","design","figma","ux"
]
BASE_SKILLS = list(dict.fromkeys([s.lower() for s in BASE_SKILLS]))

def extract_skills_simple(text):
    text = (text or "").lower()
    found = []
    for kw in BASE_SKILLS:
        if kw in text:
            found.append(kw)
    # parse explicit "skills:" lines
    for m in re.finditer(r"(?:skills|skillset)[:\s-]*([^\n\r]+)", text):
        tail = m.group(1)
        for part in re.split(r"[;,|/]+", tail):
            p = part.strip().lower()
            if p and len(p) > 1 and p not in found:
                found.append(p)
    # deduplicate preserve order
    merged = []
    for s in found:
        if s not in merged:
            merged.append(s)
    return merged

# ---------------- Ranking (clamped, friendly explain, embedding fallback) ----------------
def rank_internships_by_query(query_text, top_k=TOP_K):
    query = (query_text or "").strip()
    if not query:
        return []

    extracted_skills = extract_skills_simple(query)
    cleaned = [s for s in extracted_skills if not re.match(r"^skill[_\s-]*\d+$", s)]
    cleaned = cleaned[:10]

    qdoc = query + " " + " ".join(cleaned)

    scores = None
    used_embeddings = False
    try:
        if model is not None and corpus_embeddings is not None:
            q_emb = model.encode(qdoc, convert_to_tensor=True, show_progress_bar=False)
            try:
                import torch
                if isinstance(corpus_embeddings, np.ndarray):
                    corpus_tensor = torch.tensor(corpus_embeddings)
                else:
                    corpus_tensor = corpus_embeddings
                sims = util.cos_sim(q_emb, corpus_tensor)[0].cpu().numpy()
            except Exception:
                sims = util.cos_sim(q_emb, corpus_embeddings)[0].cpu().numpy()
            scores = sims
            used_embeddings = True
    except Exception as e:
        app.logger.debug("Embedding similarity failed: %s", e)
        scores = None
        used_embeddings = False

    results = []
    for idx, row in enumerate(corpus_rows):
        raw_base = float(scores[idx]) if (scores is not None and idx < len(scores)) else 0.0

        if used_embeddings:
            if raw_base < -0.0001 or raw_base > 1.0001:
                base_score = max(0.0, min(1.0, (raw_base + 1) / 2.0))
            else:
                base_score = max(0.0, min(1.0, raw_base))
        else:
            base_score = 0.0

        job_skills_text = str(row.get(skill_col, "")).lower()
        keyword_matches = 0
        matched_skill_names = []
        for s in cleaned:
            if s and s.lower() in job_skills_text:
                keyword_matches += 1
                if s not in matched_skill_names:
                    matched_skill_names.append(s)

        domain = str(row.get("domain","")).strip()
        location = str(row.get("location","")).strip()
        domain_match = 1 if domain and domain.lower() in query.lower() else 0
        location_match = 1 if location and location.lower() in query.lower() else 0

        keyword_boost = min(0.4, 0.08 * keyword_matches)
        domain_loc_boost = 0.08 * (domain_match + location_match)
        combined = base_score + keyword_boost + domain_loc_boost

        if not used_embeddings:
            kw_score = min(0.6, 0.2 * keyword_matches)
            dl_score = 0.1 * (domain_match + location_match)
            combined = max(combined, kw_score + dl_score)

        combined = max(0.0, min(1.0, combined))
        percent = round(combined * 100, 2)

        why_parts = []
        if matched_skill_names:
            why_parts.append("Matched skills: " + ", ".join(matched_skill_names))
        if domain_match:
            why_parts.append(f"Matches domain: {domain}")
        if location_match:
            why_parts.append(f"Location match: {location}")
        why_parts.append(f"Confidence: {percent}%")
        why_text = " · ".join(why_parts)

        results.append({
            "row_index": idx,
            "title": row.get("title",""),
            "company": row.get("company",""),
            "domain": domain,
            "location": location,
            "duration": row.get("duration_months", row.get("duration","N/A")),
            "stipend": row.get("stipend","N/A"),
            "skills": row.get(skill_col,""),
            "description": row.get("description",""),
            "apply_link": row.get("apply_link",""),
            "embedding_raw": round(float(base_score), 4),
            "keyword_matches": keyword_matches,
            "match_percent": percent,
            "matched_skills": matched_skill_names,
            "explain": why_text
        })

    results_sorted = sorted(results, key=lambda x: x["match_percent"], reverse=True)
    return results_sorted[:min(top_k, len(results_sorted))]

# ---------------- Career guidance & leaderboard ----------------
CAREER_MAP = {
    "python": ["Data Science", "AI/ML", "Backend Development"],
    "nlp": ["AI/ML", "Research"],
    "sql": ["Data Analytics", "Data Engineering"],
    "react": ["Frontend", "Full Stack"],
    "docker": ["DevOps", "Cloud"],
    "aws": ["Cloud Engineer", "DevOps"],
}

def career_guidance_from_skills(skills_list):
    scores = Counter()
    for s in skills_list:
        key = s.lower()
        for k, domains in CAREER_MAP.items():
            if k in key:
                for d in domains:
                    scores[d] += 1
    if not scores:
        return []
    return [d for d,_ in scores.most_common(3)]

def skills_leaderboard(top_k=5):
    df = read_internships()
    col = skill_col if skill_col in df.columns else "skills_required"
    all_sk = []
    for s in df[col].astype(str).tolist():
        parts = re.split(r"[;,/|]+", s)
        for p in parts:
            p = p.strip().lower()
            if not p:
                continue
            if re.match(r"^skill[_\s-]*\d+$", p):
                continue
            if re.match(r"^\d+$", p):
                continue
            if len(p) <= 1:
                continue
            all_sk.append(p)
    cnt = Counter(all_sk)
    out = [{"skill": k.title(), "count": v} for k, v in cnt.most_common(top_k)]
    if not out:
        raw = []
        for s in df[col].astype(str).tolist():
            for p in re.split(r"[;,/|]+", s):
                p = p.strip()
                if p and p.lower() not in ("skill1","skill2","skill3"):
                    raw.append(p.lower())
        cnt2 = Counter(raw)
        out = [{"skill": k.title(), "count": v} for k, v in cnt2.most_common(top_k)]
    return out

# ---------------- Translator helper ----------------
def translate_text(text, dest_lang="en"):
    if not text:
        return text
    if not HAS_TRANSLATOR or not TRANSLATOR:
        return text
    try:
        # if destination is English, keep original to avoid double-translation
        if dest_lang == "en":
            return text
        return TRANSLATOR(text, dest_lang)
    except Exception as e:
        app.logger.debug("Translation failed (%s -> %s): %s", getattr(text, '__repr__', lambda: '')(), dest_lang, e)
        return text

# ---------------- Routes & UI translations ----------------
TRANSLATIONS = {
    "en": {"title": "PM-internAI — Internship Recommender", "upload_cta": "Upload your CV (PDF/DOCX/TXT)", "or_fill": "Or fill manually", "find_btn": "Find Internships", "assistant": "AI Assistant", "skills_in_demand": "Skills in demand", "career_suggestions": "Career suggestions", "match_score": "Match Score", "apply": "Apply Now", "loading": "🔍 AI analyzing your resume..."},
    "hi": {"title": "PM-internAI — इंटर्नशिप सिफारिशकर्ता", "upload_cta": "अपना CV अपलोड करें (PDF/DOCX/TXT)", "or_fill": "या मैन्युअली भरें", "find_btn": "इंटर्नशिप खोजें", "assistant": "एआई सहायक", "skills_in_demand": "जरूरी स्किल्स", "career_suggestions": "करिअर सुझाव", "match_score": "मेल स्कोर", "apply": "अभी आवेदन करें", "loading": "🔍 एआई आपके रिज़्यूमे को विश्लेषित कर रहा है..."},
    "ta": {"title": "PM-internAI — இன்டர்ன்ஷிப் பரிந்துரை", "upload_cta": "உங்கள் சி.வி பதிவேற்றவும் (PDF/DOCX/TXT)", "or_fill": "அல்லது கைமுறையாக பூர்த்தி செய்யவும்", "find_btn": "இன்டர்ன்ஷிப் தேடு", "assistant": "ஏ.ஐ உதவியாளர்", "skills_in_demand": "தேவைப்பட்ட நெருக்கடி", "career_suggestions": "தொழில் பரிந்துரைகள்", "match_score": "பொருத்த மதிப்பெண்", "apply": "இப்போது விண்ணப்பிக்க", "loading": "🔍 உங்கள் ரெஸ்யூமையை ஏ.ஐ ஆய்வு செய்கிறது..."}
}
def t_get(lang, key):
    if not lang or lang not in TRANSLATIONS:
        lang = "en"
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))

@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.values.get("lang", "en")
    if lang not in SUPPORTED_LANGS:
        lang = "en"

    assistant_answer = ""
    guidance = []
    leaderboard = skills_leaderboard()
    recommendations = []
    debug_text = ""

    if request.method == "POST":
        # ---- DEBUG: introspect incoming POST (helps verify uploads reached server) ----
        try:
            app.logger.info("---- POST incoming ----")
            app.logger.info("form keys: %s", list(request.form.keys()))
            app.logger.info("files keys: %s", list(request.files.keys()))
            if "resume" in request.files:
                ftmp = request.files["resume"]
                try:
                    data = safe_read_file_bytes(ftmp)
                    app.logger.info("Uploaded file name=%s content_type=%s bytes=%d",
                                    getattr(ftmp, "filename", None),
                                    getattr(ftmp, "content_type", None),
                                    len(data) if data is not None else 0)
                    try:
                        ftmp.stream.seek(0)
                    except Exception:
                        pass
                except Exception as e:
                    app.logger.exception("Failed reading uploaded file: %s", e)
            else:
                app.logger.info("No resume file in request.files")
        except Exception as e:
            app.logger.exception("POST debug failed: %s", e)
        # ---- end debug ----

        # AI Assistant question box (live recommender)
        question = request.form.get("assistant_question","").strip()
        if question:
            recs = rank_internships_by_query(question, top_k=3)
            recs_translated = []
            for r in recs:
                recs_translated.append({
                    **r,
                    "title_t": translate_text(r["title"], dest_lang=lang),
                    "company_t": translate_text(r["company"], dest_lang=lang),
                    "domain_t": translate_text(r["domain"], dest_lang=lang),
                    "location_t": translate_text(r["location"], dest_lang=lang),
                    "description_t": translate_text(r["description"], dest_lang=lang),
                    "matched_skills_t": [translate_text(s, dest_lang=lang) for s in r["matched_skills"]],
                    "explain_t": translate_text(r["explain"], dest_lang=lang),
                })
            assistant_answer = recs_translated
        else:
            # Resume upload?
            if "resume" in request.files and request.files["resume"].filename != "":
                file = request.files["resume"]
                text = extract_text_from_file(file)
                debug_text = f"DEBUG: Resume length={len(text)}"
                recs = rank_internships_by_query(text, top_k=TOP_K)
                guidance = career_guidance_from_skills(extract_skills_simple(text))
            else:
                # Manual form
                name = request.form.get("name","").strip()
                email = request.form.get("email","").strip()
                skills_input = request.form.get("skills","").strip()
                domain = request.form.get("domain","").strip()
                location_pref = request.form.get("location","").strip()
                manual_text = " ".join([name, skills_input, domain, location_pref])
                recs = rank_internships_by_query(manual_text, top_k=TOP_K)
                guidance = career_guidance_from_skills(extract_skills_simple(manual_text))

            # translate recs for UI
            recs_translated = []
            for r in recs:
                recs_translated.append({
                    **r,
                    "title_t": translate_text(r["title"], dest_lang=lang),
                    "company_t": translate_text(r["company"], dest_lang=lang),
                    "domain_t": translate_text(r["domain"], dest_lang=lang),
                    "location_t": translate_text(r["location"], dest_lang=lang),
                    "description_t": translate_text(r["description"], dest_lang=lang),
                    "matched_skills_t": [translate_text(s, dest_lang=lang) for s in r["matched_skills"]],
                    "explain_t": translate_text(r["explain"], dest_lang=lang),
                })
            recommendations = recs_translated

    return render_template("index.html",
                           lang=lang,
                           t_get=t_get,
                           assistant_answer=assistant_answer,
                           guidance=guidance,
                           leaderboard=leaderboard,
                           recommendations=recommendations,
                           debug_text=debug_text)

@app.route("/apply_redirect", methods=["POST"])
def apply_redirect():
    base_link = request.form.get("apply_link","")
    profile = {
        "name": request.form.get("name",""),
        "email": request.form.get("email",""),
        "skills": [s.strip() for s in request.form.get("skills","").split(",") if s.strip()]
    }
    params = {}
    if profile.get("name"):
        params["name"] = profile["name"]
    if profile.get("email"):
        params["email"] = profile["email"]
    if profile.get("skills"):
        params["skills"] = ",".join(profile["skills"])
    if not params:
        return redirect(base_link or "/")
    query = urlencode(params)
    if "?" in base_link:
        return redirect(base_link + "&" + query)
    else:
        return redirect(base_link + "?" + query)

@app.route("/admin", methods=["GET","POST"])
def admin():
    msg = ""
    if request.method == "POST":
        if "csvfile" in request.files and request.files["csvfile"].filename != "":
            f = request.files["csvfile"]
            try:
                data = pd.read_csv(f)
                os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
                backup = DATA_FILE + ".bak"
                if os.path.exists(DATA_FILE):
                    os.replace(DATA_FILE, backup)
                data.to_csv(DATA_FILE, index=False)
                # reload data and embeddings
                global internships, corpus_rows, corpus_embeddings
                internships = read_internships()
                build_corpus_embeddings(internships)
                msg = "CSV uploaded and data reloaded."
            except Exception as e:
                msg = f"Upload failed: {e}"
    return render_template("admin.html", msg=msg)

# ---------------- File extraction ----------------
def safe_read_file_bytes(file_storage):
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass
    data = file_storage.read()
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass
    return data

def extract_text_from_file(file_storage):
    fname = file_storage.filename or ""
    fname_lower = fname.lower()
    try:
        data = safe_read_file_bytes(file_storage)
        if fname_lower.endswith(".pdf") and fitz is not None:
            try:
                pdf = fitz.open(stream=data, filetype="pdf")
                text = ""
                for p in pdf:
                    text += p.get_text()
                pdf.close()
                return text
            except Exception as e:
                app.logger.debug("PDF read error: %s", e)
                return ""
        elif fname_lower.endswith(".docx") and docx2txt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    tmp_path = tmp.name
                text = docx2txt.process(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return text or ""
            except Exception as e:
                app.logger.debug("DOCX read error: %s", e)
                return ""
        elif fname_lower.endswith(".txt"):
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        else:
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return ""
    except Exception as e:
        app.logger.debug("extract_text_from_file error: %s", e)
        return ""

if __name__ == "__main__":
    internships = read_internships()
    build_corpus_embeddings(internships)
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

