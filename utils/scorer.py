# Import the SBERT-based functions from similarity.py (located in the root directory)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from similarity import compute_semantic_score, apply_penalty
from utils.keyword_extractor import _normalize_keyword

def calculate_ats_score(
    matched_keywords: list,
    jd_keywords: list,
    similarity_score: float,
    or_groups: list | None = None,
    parsed_resume: dict = None,
    jd_text: str = ""
) -> tuple[float, float, dict]:
    """
    Calculates the final ATS score based on:
      1. Keyword match rate   — 40% weight
      2. Semantic similarity  — 50% weight
      3. Format score         — 10% weight (placeholder: 70.0)

    Also applies a soft penalty when similarity is very low (< 10%),
    which catches keyword-stuffed resumes that aren't actually relevant.

    Returns:
       final_score        (float, 0–100)
       keyword_match_pct  (float, 0–100)
       breakdown          (dict)  — component scores for display/suggestions
    """
    or_groups = or_groups or []
    
    # --- 1. Keyword match percentage ---
    if not jd_keywords:
        keyword_match_pct = 0.0
    else:
        # Use the actual lists so the percentage matches the displayed count
        # Normalize keywords: lowercase, strip punctuation, lemmatize
        matched_set = set(_normalize_keyword(kw) for kw in matched_keywords)
        jd_set = set(_normalize_keyword(kw) for kw in jd_keywords)
        keyword_match_pct = (len(matched_set) / len(jd_set)) * 100.0 if len(jd_set) > 0 else 0.0

    # --- 2. Semantic similarity calculation using SBERT ---
    # If parsed_resume and jd_text are provided, compute SBERT-based similarity
    if parsed_resume is not None and jd_text:
        # Build resume_sections from parsed_resume data
        resume_sections = {
            "summary": parsed_resume.get("summary", ""),
            "skills": " ".join(parsed_resume.get("skills", [])),
            "experience": " ".join(parsed_resume.get("experience", [])),
            "projects": " ".join(parsed_resume.get("projects", []))
        }
        
        # Compute semantic score using SBERT
        raw_similarity = compute_semantic_score(resume_sections, jd_text)
        
        # Apply penalty function
        similarity_score, penalty_msg = apply_penalty(raw_similarity)
    # Else, use the passed-in similarity_score (backward compatibility)
    
    # --- 3. Weighted combination ---
    weight_keywords   = 0.40
    weight_similarity = 0.50
    weight_format     = 0.10

    # Format score placeholder (would be calculated from resume structure/formatting)
    format_score = 70.0

    # Calculate final ATS score
    raw_score = (keyword_match_pct * 0.40) + (similarity_score * 0.50) + (format_score * 0.10)

    # --- 4. Soft penalty for very low semantic similarity ---
    # If cosine similarity < 10%, the resume is likely unrelated or poorly written.
    # Penalise up to 5 points to reflect that keyword hits alone aren't enough.
    penalty = 0.0
    if similarity_score < 10.0:
        # Linear penalty: 0 pts at 10%, up to 5 pts at 0%
        penalty = (10.0 - similarity_score) / 10.0 * 5.0

    final_score = max(0.0, min(raw_score - penalty, 100.0))

    # --- 5. Breakdown dict for UI / suggestions layer ---
    breakdown = {
        "keyword_match_pct": round(keyword_match_pct, 2),
        "similarity_score":  round(similarity_score, 2),
        "format_score":      round(format_score, 2),
        "penalty_applied":   round(penalty, 2),
        "keyword_weight":    weight_keywords,
        "similarity_weight": weight_similarity,
        "format_weight":     weight_format,
    }

    return round(final_score, 2), round(keyword_match_pct, 2), breakdown