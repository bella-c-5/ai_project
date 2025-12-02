import os
import joblib

THIS_DIR = os.path.dirname(__file__)
SCORE_STATS_PATH = os.path.join(THIS_DIR, "score_stats.pkl")

_stats = None

def _load():
    global _stats
    if _stats is None:
        _stats = joblib.load(SCORE_STATS_PATH)

def policy_gradient_score(section_flags, word_count):
    """
    section_flags: [objective, projects, achievements, skills, education]
    """
    _load()

    # Score part 1 — section coverage
    coverage_score = sum(section_flags) / 5.0

    # Score part 2 — realistic length relative to dataset
    mean_wc = _stats["mean_word_count"]
    std_wc = _stats["std_word_count"] or 1.0

    z = (word_count - mean_wc) / std_wc
    length_score = 1 / (1 + pow(2.71828, -z))

    # Weighted combination
    final = 0.65 * coverage_score + 0.35 * length_score

    return max(0.0, min(1.0, final))
