import os
import joblib

THIS_DIR = os.path.dirname(__file__)
SKILL_MAP_PATH = os.path.join(THIS_DIR, "skill_map.pkl")

_skill_map = None

def _load_skill_map():
    global _skill_map
    if _skill_map is None:
        _skill_map = joblib.load(SKILL_MAP_PATH)

def recommend_skills_dqn(resume_text: str, career_field: str, top_k=5):
    """Recommend missing skills based on Resume.csv data."""
    _load_skill_map()

    resume_lower = resume_text.lower()

    if not career_field or career_field not in _skill_map:
        # fallback: global top skills
        skills = sorted(list({s for field in _skill_map for s in _skill_map[field]}))
        return skills[:top_k]

    field_skills = _skill_map[career_field]

    missing = [s for s in field_skills if s not in resume_lower][:top_k]

    if not missing:
        missing = field_skills[:top_k]

    return missing
