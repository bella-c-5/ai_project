from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os, io, shutil, re, torch

from ai_models.probabilistic_model import predict_field
from ai_models.skill_dqn_agent import recommend_skills_dqn
from ai_models.pg_score_model import policy_gradient_score
from ai_models.resume_mdp_model import ResumeMDP
from ai_models.resume_attention_module import ResumeAttention

from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage


# initializing FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def clean_tokenize(text):
    # lowercase + remove punctuation
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = cleaned.split()
    return tokens

# Extracts raw text from a PDF file using pdfminer.
def pdf_reader(path):
    """Extract raw text from a PDF using pdfminer."""
    resource_manager = PDFResourceManager()
    handle = io.StringIO()
    converter = TextConverter(resource_manager, handle, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(path, "rb") as fh:
        for page in PDFPage.get_pages(fh):
            interpreter.process_page(page)

    text = handle.getvalue()

    converter.close()
    handle.close()
    return text


# Uses regex and simple heuristics to extract: email address and names
# Names are first capitalized multi-word line

def extract_basic_info(text):
    """Extract simple name + email using regex + heuristics."""

    # email
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group(0) if email_match else None

    # Simple name extraction: first capitalized line with multiple words
    name = None
    for line in text.split("\n"):
        clean = line.strip()
        if len(clean.split()) >= 2 and clean[0].isupper():
            name = clean
            break

    return {"name": name, "email": email}


# serves HTML/CSS/JS files for the frontend UI
@app.get("/", response_class=HTMLResponse)
def load_index():
    return open("frontend/index.html").read()


@app.get("/script.js")
def serve_js():
    return FileResponse("frontend/script.js")


@app.get("/styles.css")
def serve_css():
    return FileResponse("frontend/styles.css")


# MAIN API

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Main resume analysis endpoint."""

    # save uploaded resume
    upload_dir = "data/resumes_uploaded"
    os.makedirs(upload_dir, exist_ok=True)

    save_path = os.path.join(upload_dir, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # extracts text
    resume_text = pdf_reader(save_path)
    text_lower = resume_text.lower()

    # extracts name & email
    resume_info = extract_basic_info(resume_text)

    response = {
        "name": resume_info.get("name"),
        "email": resume_info.get("email"),
    }

    # Field prediction - trained on Resume.csv
    field, conf = predict_field(resume_text)
    response["field"] = field
    response["confidence"] = conf

    # Missing skill recommendation - trained on Resume.csv
    response["dqn_skill"] = recommend_skills_dqn(resume_text, field)

    # Computes resume score with coverage of sections + statistical length score
    features = [
        int("objective" in text_lower),
        int("projects" in text_lower),
        int("achievements" in text_lower),
        int("skills" in text_lower),
        int("education" in text_lower)
    ]

    wc = len(text_lower.split())
    score = policy_gradient_score(features, wc)
    response["score"] = round(score * 100, 2)

     # MDP-based improvement suggestions: add/improve each major resume section
    mdp = ResumeMDP()
    actions = []
    for sec in mdp.states:
        action = "add" if sec not in text_lower else "improve"
        actions.append({
            "section": sec,
            "action": action,
            "reward": mdp.reward(action)
        })
    response["mdp_actions"] = actions

    # attention module
    tokens = clean_tokenize(resume_text)

    # Convert tokens → numeric IDs - simple hashing for embedding lookup
    token_ids = [abs(hash(t)) % 5000 for t in tokens]
    token_tensor = torch.tensor(token_ids).unsqueeze(1)

    attn_model = ResumeAttention()

    with torch.no_grad():
        attn_scores = attn_model(token_tensor).tolist()

    # Pair each token with its attention weight
    token_importance = list(zip(tokens, attn_scores))

    # Sorts descending by importance
    token_importance.sort(key=lambda x: x[1], reverse=True)

    # Selects top 10 most important words
    top_words = [t for t, score in token_importance[:10]]

    # Cleaning words

    # Removes common stopwords (generic useless words)
    STOPWORDS = {
    "and","or","the","for","with","a","an","to","of","in","on","is","are",
    "was","were","this","that","these","those","as","by","at","from","it",
    "your","you","i","my","me","our","their","they","he","she","we","s"
    }
    cleaned = [w for w in top_words if w and len(w) > 2 and w.lower() not in STOPWORDS]

    # Removes numeric-only tokens
    cleaned = [w for w in cleaned if not w.isdigit()]

    # Removes months - date noise
    MONTHS = {"january","february","march","april","may","june",
          "july","august","september","october","november","december"}
    cleaned = [w for w in cleaned if w.lower() not in MONTHS]

    # Removes super-common resume filler words
    BAD_WORDS = {
        "experience","responsible","duties","tasks","worked","work","ability",
        "independent","effectively","objective","summary","resume","position"
    }
    cleaned = [w for w in cleaned if w.lower() not in BAD_WORDS]

    # Removes U.S. states, abbreviations
    STATES = {"nj","ny","mi","ca","tx","fl","pa","oh","wi","il"}
    cleaned = [w for w in cleaned if w.lower() not in STATES]

    # Brings in skills from your skill_map - the most powerful filter
    from ai_models.skill_dqn_agent import _skill_map
    ALL_SKILLS = {s.lower() for v in _skill_map.values() for s in v}

    # Skill-based filtering
    skill_filtered = [w for w in cleaned if w.lower() in ALL_SKILLS]

    # Adds curated resume keywords (backup when resume lacks technical keywords)
    RESUME_KEYWORDS = {
        "python","java","sql","c++","azure","aws","excel","tableau","git",
        "tensorflow","pytorch","react","node","leadership","communication",
        "research","analysis","data","clinical","patient","care","finance",
        "supervision","presentation","teaching","counseling","administration"
    }
    skill_filtered += [w for w in cleaned if w.lower() in RESUME_KEYWORDS]

    # Removes duplicates while preserving order
    unique = []
    for w in skill_filtered:
        if w not in unique:
            unique.append(w)

    # Requires professional-looking words (capitalized or long)
    final_keywords = [
        w for w in unique
        if w[0].isupper() or len(w) > 4
    ]

    # Store the final cleaned keywords
    response["important_words"] = unique[:5]

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)
