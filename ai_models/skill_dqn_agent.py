import os
import joblib

# Recommends missing skills using a field-specific skill map that generated during training
# Concept of DQN / state = resume text, actions = missing skill suggestions

THIS_DIR = os.path.dirname(__file__)
SKILL_MAP_PATH = os.path.join(THIS_DIR, "skill_map.pkl")

_skill_map = None

# Lazy load skill map
def _load_skill_map():
    global _skill_map
    if _skill_map is None:
        _skill_map = joblib.load(SKILL_MAP_PATH)

# Recommends top-k missing skills for the predicted career field
def recommend_skills_dqn(resume_text: str, career_field: str, top_k=5):
    """Recommend missing skills based on Resume.csv data."""
    _load_skill_map()

    resume_lower = resume_text.lower()

    # If field is unknown, fallback to global skills
    if not career_field or career_field not in _skill_map:
        skills = sorted(list({s for field in _skill_map for s in _skill_map[field]}))
        return skills[:top_k]

    field_skills = _skill_map[career_field]

    # Returns skills that aren't mentioned in the resume text
    missing = [s for s in field_skills if s not in resume_lower]

    # If resume contains all skills then return top skills
    if not missing:
        missing = field_skills

    # skills cleaning

    # Words that are NOT skills and should never appear in recommendations
    NON_SKILLS = {
        "state","city","name","patient","medical","care","information","team",
        "environment","service","member","work","experience","record","support",
        "clinic","hospital","office","responsibilities","role","position"
    }

    # Curated list of real skills & resume-relevant keywords
    REAL_SKILLS = {
        "python","java","sql","c++","c","javascript","typescript","react","node",
        "azure","aws","linux","excel","tableau","git","pandas","numpy","ml","ai",
        "leadership","communication","analysis","research","data","sales",
        "project","management","cpr","bls","hipaa","emr","vitals","triage",
        "documentation","phlebotomy","billing","coding","administration"
    }

    # Removes obvious non-skills
    cleaned = [s for s in missing if s.lower() not in NON_SKILLS]

    # Kees only real skills from curated list
    cleaned = [s for s in cleaned if s.lower() in REAL_SKILLS]

    # Fallback if nothing survived filtering
    if not cleaned:
        cleaned = [s for s in field_skills if s.lower() in REAL_SKILLS]

    # Final slice AFTER cleaning
    return cleaned[:top_k]