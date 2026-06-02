import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

from groq import Groq

import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


def generate_suggestions(missing_keywords, matched_keywords, breakdown, resume_text="", or_groups=None):
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Groq API unavailable, using fallback suggestions")
        return _fallback_suggestions(missing_keywords, matched_keywords, breakdown)
    
    try:
        from groq import Groq
    except ImportError:
        print("Groq API unavailable, using fallback suggestions")
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
   what phrasing to use)

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
        import ssl
        import httpx

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        http_client = httpx.Client(verify=False)

        client = Groq(
            api_key=api_key,
            http_client=http_client
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an ATS resume expert. Give specific, actionable suggestions to improve a resume."
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
    """Generate 3 generic fallback suggestions when Groq API is unavailable."""
    suggestions = []
    
    keyword_pct = breakdown.get("keyword_match", 0)
    similarity = breakdown.get("semantic_similarity", 0)
    
    # Suggestion 1: Based on missing keywords
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
    
    # Suggestion 2: Based on semantic similarity
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
    
    # Suggestion 3: Based on keyword match rate
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
