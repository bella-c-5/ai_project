from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os, io, shutil, re

from ai_models.probabilistic_model import predict_field
from ai_models.skill_dqn_agent import recommend_skills_dqn
from ai_models.pg_score_model import policy_gradient_score
from ai_models.resume_mdp_model import ResumeMDP

from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage


# ---------------- APP SETUP ----------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# -------------- PDF READER -----------------

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


# -------------- BASIC INFO EXTRACTOR -----------------

def extract_basic_info(text):
    """Extract simple name + email using regex + heuristics."""

    # Email
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group(0) if email_match else None

    # Naive name guess: first capitalized line with >= 2 words
    name = None
    for line in text.split("\n"):
        clean = line.strip()
        if len(clean.split()) >= 2 and clean[0].isupper():
            name = clean
            break

    return {"name": name, "email": email}


# -------------- FRONTEND ROUTES -----------------

@app.get("/", response_class=HTMLResponse)
def load_index():
    return open("frontend/index.html").read()


@app.get("/script.js")
def serve_js():
    return FileResponse("frontend/script.js")


@app.get("/styles.css")
def serve_css():
    return FileResponse("frontend/styles.css")


# -------------- MAIN API -----------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Main resume analysis endpoint."""

    # Save uploaded resume
    upload_dir = "data/resumes_uploaded"
    os.makedirs(upload_dir, exist_ok=True)

    save_path = os.path.join(upload_dir, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text
    resume_text = pdf_reader(save_path)
    text_lower = resume_text.lower()

    # Extract name & email (replaces broken pyresparser)
    resume_info = extract_basic_info(resume_text)

    response = {
        "name": resume_info.get("name"),
        "email": resume_info.get("email"),
    }

    # FIELD PREDICTION (trained on Resume.csv)
    field, conf = predict_field(resume_text)
    response["field"] = field
    response["confidence"] = conf

    # SKILL RECOMMENDATION (trained on Resume.csv)
    response["dqn_skill"] = recommend_skills_dqn(resume_text, field)

    # SCORE (section coverage + length statistics)
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

    # SECTION ACTIONS (MDP)
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

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)
