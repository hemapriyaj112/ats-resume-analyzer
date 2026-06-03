import os
import re
from dotenv import load_dotenv
load_dotenv()

# Noise words that should never appear as "missing skills" in suggestions
_SUGGESTION_NOISE = {
    "must", "following", "follow", "also", "well", "least", "within",
    "across", "per", "via", "based", "using", "used", "including",
    "etc", "such", "like", "known", "given", "provide", "provided",
    "ensure", "ensuring", "support", "supporting", "strong", "good",
    "excellent", "various", "multiple", "preferred", "required",
    "relevant", "related", "solid", "proven", "demonstrated",
    "ability", "experience", "knowledge", "understanding", "skill",
    "skills", "role", "position", "team", "company", "work", "job",
    "year", "years", "field", "basic", "general", "open", "complex",
    "high", "low", "large", "small", "new", "old",
}

def _filter_keywords(keywords: list) -> list:
    """Remove noise words from keyword lists before showing in suggestions."""
    return [
        kw for kw in keywords
        if kw.lower().strip() not in _SUGGESTION_NOISE
        and len(kw.strip()) > 2
        and not kw.strip().isdigit()
    ]


def generate_suggestions(missing_keywords, matched_keywords, breakdown, resume_text="", or_groups=None):
    api_key = os.getenv("GROQ_API_KEY")

    # Filter noise from keywords before anything else
    missing_keywords = _filter_keywords(missing_keywords)
    matched_keywords = _filter_keywords(matched_keywords)

    if not api_key:
        print("Groq API unavailable, using fallback suggestions")
        return _fallback_suggestions(missing_keywords, matched_keywords, breakdown)

    try:
        from groq import Groq
    except ImportError:
        print("Groq package not installed, using fallback suggestions")
        return _fallback_suggestions(missing_keywords, matched_keywords, breakdown)

    prompt = f"""You are an ATS resume expert reviewing a candidate's
resume scan results. Give 5-6 specific, actionable suggestions.

ATS Scan Results:
- Overall ATS Score: {breakdown.get('ats_score', 0):.1f}%
- Keyword Match: {breakdown.get('keyword_match', 0):.1f}%
- Semantic Similarity: {breakdown.get('semantic_similarity', 0):.1f}%
- Format Score: {breakdown.get('format_score', 0):.1f}%

Matched Skills: {', '.join(matched_keywords) if matched_keywords else 'None'}
Missing Skills: {', '.join(missing_keywords) if missing_keywords else 'None'}

Resume Summary (for context — do not invent details not present here):
\"\"\"
{resume_text[:800] if resume_text else 'Not provided'}
\"\"\"

Generate suggestions across ALL of these categories
(use each category at most once):

1. MISSING KEYWORDS — For each missing skill, suggest exactly
   where and how to add it in the resume (which section,
   what phrasing to use). Only mention real technical skills,
   tools, or frameworks — never generic words.

2. SEMANTIC ALIGNMENT — The semantic similarity score shows how
   closely the resume language matches the JD. If below 50%,
   suggest specific phrases from the JD the candidate should
   mirror in their summary or experience bullets.

3. QUANTIFICATION — Identify vague responsibility statements
   and suggest how to rewrite them with measurable outcomes
   (numbers, percentages, scale)

4. SECTION IMPROVEMENT — Suggest one specific section
   (summary, skills, projects, or experience) that needs
   the most rewriting and give a concrete example of how
   to improve it

5. PROOF OF WORK — Suggest adding GitHub links, live demo
   URLs, or certifications that prove the matched skills

Rules:
- Be specific, never generic
- Only suggest REAL technical skills, tools, or frameworks as missing keywords
- Never mention generic words like 'must', 'following', 'various', 'strong' as skills
- Each suggestion must reference actual skills or scores above
- Each suggestion starts with an emoji and bold title
- Max 3 sentences per suggestion
- Do not repeat the same category twice
- Do not suggest adding skills that are already matched
- Only reference actual details from the Resume Summary above.
  Never invent company names, years, or metrics not present
  in the resume.

Return only the numbered suggestions, no preamble or closing."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an ATS resume expert. Give specific, actionable suggestions to improve a resume. Never treat generic English words as skills."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500
        )
        result = response.choices[0].message.content
        return [result]
    except Exception as e:
        print(f"Groq API error: {type(e).__name__}: {e}")
        return _fallback_suggestions(missing_keywords, matched_keywords, breakdown)


def _fallback_suggestions(missing_keywords, matched_keywords, breakdown):
    """Generate fallback suggestions when Groq API is unavailable."""
    suggestions = []

    keyword_pct = breakdown.get("keyword_match", 0)
    similarity = breakdown.get("semantic_similarity", 0)

    if missing_keywords:
        missing_str = ", ".join(missing_keywords[:3])
        suggestions.append(
            f"🔑 **Add Missing Keywords**: Your resume is missing key skills like {missing_str}. "
            f"Add these explicitly to your skills section and weave them into your experience descriptions."
        )
    else:
        suggestions.append(
            "✅ **Strong Keyword Match**: Your resume covers the key skills from the job description."
        )

    if similarity < 35.0:
        suggestions.append(
            "📄 **Improve Text Alignment**: Mirror the job description's language in your summary and experience. "
            "Use the same terminology the JD uses to describe responsibilities and achievements."
        )
    else:
        suggestions.append(
            "📄 **Good Contextual Alignment**: Your resume language aligns well with the JD. "
            "Continue using similar phrasing to maintain this alignment."
        )

    if keyword_pct < 65.0:
        suggestions.append(
            "📝 **Expand Keyword Coverage**: Add more of the missing technical skills to your resume. "
            "Focus on a dedicated Skills section and integrate keywords naturally into your project descriptions."
        )
    else:
        suggestions.append(
            "✅ **Excellent Keyword Coverage**: Your resume covers most essential skills. "
            "Focus on formatting and presentation to maximize impact."
        )

    return suggestions