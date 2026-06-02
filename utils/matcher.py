import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.keyword_extractor import SKILL_SYNONYMS, SYNONYMS, _clean_phrase, lemmatizer

# ---------------------------------------------------------------------------
# Semantic similarity using TF-IDF cosine similarity only
# ---------------------------------------------------------------------------

# Expanded bidirectional synonym map
SYNONYM_MAP = {
    "ml": ["machine learning"],
    "machine learning": ["ml"],
    "ai": ["artificial intelligence"],
    "artificial intelligence": ["ai"],
    "nlp": ["natural language processing"],
    "natural language processing": ["nlp"],
    "aws": ["amazon web services"],
    "amazon web services": ["aws"],
    "gcp": ["google cloud", "google cloud platform"],
    "google cloud": ["gcp", "google cloud platform"],
    "google cloud platform": ["gcp", "google cloud"],
    "azure": ["microsoft azure"],
    "microsoft azure": ["azure"],
    "react": ["reactjs", "react.js"],
    "reactjs": ["react", "react.js"],
    "react.js": ["react", "reactjs"],
    "node": ["nodejs", "node.js"],
    "nodejs": ["node", "node.js"],
    "node.js": ["node", "nodejs"],
    "js": ["javascript"],
    "javascript": ["js"],
    "ts": ["typescript"],
    "typescript": ["ts"],
    "ui": ["user interface"],
    "ux": ["user experience"],
    "cv": ["computer vision"],
    "computer vision": ["cv"],
    "dl": ["deep learning"],
    "deep learning": ["dl"],
    "sql": ["structured query language"],
    "structured query language": ["sql"],
    "oop": ["object oriented", "object-oriented", "object-oriented programming"],
    "object oriented": ["oop", "object-oriented", "object-oriented programming"],
    "object-oriented": ["oop", "object oriented", "object-oriented programming"],
    "object-oriented programming": ["oop", "object oriented", "object-oriented"],
    "api": ["application programming interface"],
    "ci/cd": ["continuous integration", "continuous deployment"],
    "continuous integration": ["ci/cd"],
    "continuous deployment": ["ci/cd"],
    "devops": ["dev ops"],
    "k8s": ["kubernetes"],
    "kubernetes": ["k8s"],
    "llm": ["large language model"],
    "large language model": ["llm"],
    # CS fundamentals
    "git": ["version control"],
    "version control": ["git"],
    "dsa": ["data structures", "data structures and algorithms", "algorithm development"],
    "data structures": ["dsa", "data structures and algorithms"],
    "data structures and algorithms": ["dsa", "data structures"],
    # Algorithm: match resume's "DSA" or "algorithm" against JD's
    # "algorithm development" and vice versa
    "algorithm": ["algorithm development", "algorithms"],
    "algorithm development": ["algorithm", "algorithms", "dsa"],
    "algorithms": ["algorithm", "algorithm development", "dsa"],
    # Debugging
    "debugging": ["troubleshooting", "debug", "testing"],
    "troubleshooting": ["debugging", "debug", "testing"],
    "testing": ["debugging", "troubleshooting"],
    # SDLC
    "sdlc": ["software development lifecycle", "software development life cycle", "development lifecycle", "object-oriented design"],
    "software development lifecycle": ["sdlc", "software development life cycle", "object-oriented design"],
    "software development life cycle": ["sdlc", "software development lifecycle", "object-oriented design"],
    "development lifecycle": ["sdlc", "software development lifecycle", "object-oriented design"],
    "object-oriented design": ["sdlc", "software development lifecycle", "object-oriented"],
    "object-oriented": ["oop", "object oriented", "object-oriented design", "sdlc"],
    # Internship matching
    "internship": ["intern", "internship(s", "internships"],
    "intern": ["internship", "internship(s", "internships"],
    "internship(s": ["internship", "intern", "internships"],
    "internships": ["internship", "intern", "internship(s"],
    # Firebase/NoSQL
    "firebase": ["nosql", "no sql", "realtime database"],
    "nosql": ["firebase", "no sql", "mongodb", "cassandra"],
    "no sql": ["nosql", "firebase", "mongodb"],
    # Conceptual skills
    "project experience": ["project", "projects", "built", "developed", "created"],
    "projects": ["project experience", "project", "built", "developed"],
    "project": ["project experience", "projects", "built", "developed"],
    "built": ["project experience", "projects", "developed", "created"],
    "developed": ["project experience", "projects", "built", "created"],
    "problem-solving": ["problem solving", "analytical", "analytical skills"],
    "problem solving": ["problem-solving", "analytical", "analytical skills"],
    "analytical": ["analytical skills", "problem-solving", "problem solving"],
    "analytical skills": ["analytical", "problem-solving", "problem solving"],
    "communication": ["communication skills", "verbal", "written"],
    "communication skills": ["communication", "verbal", "written"],
    "technical internship": ["internship", "intern", "internship(s", "internships"],
    "technical skills": ["skills", "technical", "programming"],
    "soft skills": ["communication", "problem-solving", "leadership", "teamwork"],
    "computer science": ["cs", "bachelor", "bachelor's", "degree", "computer science degree"],
    "bachelor": ["bachelor's", "degree", "computer science", "computer science degree"],
    "bachelor's": ["bachelor", "degree", "computer science", "computer science degree"],
    "degree": ["bachelor", "bachelor's", "computer science", "computer science degree"],
    "computer science degree": ["computer science", "bachelor", "bachelor's", "degree"],
    "database systems": ["sql", "nosql", "database", "firebase", "mongodb", "postgresql"],
    "database": ["sql", "nosql", "database systems", "firebase", "mongodb"],
    "version control systems": ["git", "github", "gitlab", "version control"],
    "version control": ["git", "github", "gitlab", "version control systems"],
    "ai tools": ["machine learning", "ml", "ai", "scikit-learn", "nlp", "google colab"],
    "machine learning": ["ai tools", "ml", "ai", "scikit-learn", "nlp", "google colab"],
    "scikit-learn": ["machine learning", "ml", "ai tools", "ai"],
    "google colab": ["ai tools", "machine learning", "ml", "ai"],
    "nlp": ["ai tools", "machine learning", "ml", "ai", "natural language processing"],
}

# Hardcoded OR groups for skill matching
OR_GROUPS = {
    "Programming Language": ["Java", "Python", "C++", "C#", "Go", "Rust", "TypeScript"],
    "Degree Field": ["Computer Science", "Computer Engineering", "Data Science", 
                     "Information Systems", "STEM"],
    "Core CS Concept": ["Data Structures", "Algorithm Development", 
                        "Object-Oriented Design"],
    "Cloud Platform": ["AWS", "Azure", "GCP"],
    "Database": ["SQL", "NoSQL"],
}

# Hardcoded OR groups for skill matching
OR_GROUPS = {
    "Programming Language": ["Java", "Python", "C++", "C#", "Go", "Rust", "TypeScript"],
    "Degree Field": ["Computer Science", "Computer Engineering", "Data Science", 
                     "Information Systems", "STEM"],
    "Core CS Concept": ["Data Structures", "Algorithm Development", 
                         "Object-Oriented", "SDLC"],
    "Cloud Platform": ["AWS", "Azure", "GCP"],
    "Database": ["SQL", "NoSQL"],
}


# def _preprocess_for_similarity(text: str) -> str:
#     """Preprocess text for semantic similarity calculation."""
#     if not text:
#         return ""
#     # Lowercase
#     text = text.lower()
#     # Remove punctuation (keep spaces)
#     text = re.sub(r'[^\w\s]', ' ', text)
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text


# Single generic words that should only match as part of a full phrase
GENERIC_WORDS = {"problem", "degree", "bachelor", "projects", "internship", "version", "control", "technical", "data"}

# Ambiguous short skills that need stricter matching (not substrings)
AMBIGUOUS_SKILLS = {"go", "c", "r", "rust", "swift", "kotlin"}


def _clean_keyword(keyword: str) -> str:
    """Strip special characters from keyword for exact matching."""
    return re.sub(r"[()\[\].,]", "", keyword).strip()


def _normalize_text_for_match(text: str) -> str:
    """
    Normalize text for matching by:
    1. Converting to lowercase
    2. Stripping punctuation and extra whitespace
    3. Applying lemmatization to each word
    """
    cleaned = _clean_phrase(text)
    words = cleaned.split()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized_words)


def _normalize_keyword_for_match(keyword: str) -> str:
    """
    Normalize keyword for matching by:
    1. Converting to lowercase
    2. Stripping punctuation and extra whitespace
    3. Applying lemmatization
    """
    # Clean the keyword (lowercase, strip punctuation/whitespace)
    cleaned = _clean_phrase(keyword)
    # Apply lemmatization to each word
    words = cleaned.split()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized_words)


# Build normalized synonym map for matching
NORMALIZED_SYNONYMS = {}
for key, syns in SKILL_SYNONYMS.items():
    norm_key = _normalize_keyword_for_match(key)
    norm_syns = [_normalize_keyword_for_match(s) for s in syns]
    NORMALIZED_SYNONYMS[norm_key] = norm_syns

# Build normalized synonym map for the new SYNONYMS dict
NORMALIZED_SYNONYMS_2 = {}
for key, syns in SYNONYMS.items():
    norm_key = _normalize_keyword_for_match(key)
    norm_syns = [_normalize_keyword_for_match(s) for s in syns]
    NORMALIZED_SYNONYMS_2[norm_key] = norm_syns


def _exact_match(keyword: str, text: str) -> bool:
    """Case-insensitive exact string match with word boundary for ambiguous skills."""
    keyword_lower = keyword.lower()
    text_lower = text.lower()
    
    # For ambiguous short skills, use word boundary matching to avoid substring matches
    # e.g., "go" should not match in "Google" or "going"
    if keyword_lower in AMBIGUOUS_SKILLS:
        pattern = rf"\b{re.escape(keyword_lower)}\b"
        return bool(re.search(pattern, text_lower))
    
    return keyword_lower in text_lower


def _word_boundary_match(keyword: str, text: str) -> bool:
    escaped = re.escape(keyword)
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _remove_subsumed(keywords: list[str]) -> list[str]:
    kw_set = set(k.lower() for k in keywords)
    bigrams = {kw for kw in kw_set if len(kw.split()) > 1}
    unigrams_to_drop = set()
    for bigram in bigrams:
        for part in bigram.split():
            if part in kw_set:
                unigrams_to_drop.add(part)
    return [kw for kw in keywords if kw.lower() not in unigrams_to_drop]


def _deduplicate_keywords(keywords: list[str]) -> list[str]:
    """
    Deduplicate keywords by merging variants.
    - Bachelor and Bachelor's → Bachelor's Degree
    - Problem and Problem-Solving → Problem-Solving
    - Version Control and Version Control Systems → Version Control
    - Internship(s → Internships
    - Technical Internship and Internships → Internships
    """
    kw_set = set(k.lower().strip() for k in keywords)
    
    # Define merge rules
    merge_rules = {
        frozenset(['bachelor', "bachelor's", 'degree', 'computer science', 'computer science degree']): 'Bachelor\'s Degree',
        frozenset(['problem', 'problem-solving', 'problem solving', 'analytical', 'analytical skills']): 'Problem-Solving',
        frozenset(['version control', 'version control systems', 'git', 'github', 'gitlab']): 'Version Control',
        frozenset(['internship', 'internship(s', 'internships', 'intern', 'technical internship']): 'Internship',
    }
    
    merged = set()
    processed = set()
    
    for kw in kw_set:
        if kw in processed:
            continue
        
        # Check if this keyword is part of a merge group
        for group, merged_name in merge_rules.items():
            if kw in group:
                merged.add(merged_name)
                processed.update(group)
                break
        else:
            merged.add(kw)
    
    return list(merged)


def _detect_or_groups(jd_text: str) -> list[dict]:
    """
    Use hardcoded OR_GROUPS to detect skill groups in job description.
    Returns list of dicts with 'group_name', 'skills', and 'matched' fields.
    """
    or_groups = []
    
    # Use the hardcoded OR_GROUPS dictionary
    for group_name, skills in OR_GROUPS.items():
        or_groups.append({
            'group_name': group_name,
            'skills': skills,
            'matched': []
        })
    
    return or_groups


def match_keywords(resume_text: str, jd_keywords: list[str], jd_text: str = "") -> tuple[list, list, list, list]:
    """
    Finds which JD keywords are present in the resume text.
    Uses exact string matching after cleaning and normalizing keywords.
    Returns (matched_keywords, hard_missing, soft_missing, or_group_results).
    Hard Missing: skill not found and no synonym matched.
    Soft Missing: skill not in resume but related term found nearby.
    """
    if not resume_text:
        return [], list(jd_keywords), [], []

    matched_keywords = []
    hard_missing = []
    soft_missing = []
    
    # Detect OR groups
    or_groups = _detect_or_groups(jd_text) if jd_text else []
    
    # Get all OR group skills for exclusion from missing list
    or_group_skills = set()
    for group in or_groups:
        or_group_skills.update(s.lower() for s in group['skills'])
    
    # Normalize resume text for matching (lowercase, strip punctuation, lemmatize)
    resume_normalized = _normalize_text_for_match(resume_text)
    
    for kw in jd_keywords:
        # Clean keyword: strip ( ) [ ] . , and normalize
        cleaned_kw = _clean_keyword(kw)
        
        # Skip single generic words - they should only match as part of a full phrase
        if cleaned_kw in GENERIC_WORDS:
            continue
        
        matched = False
        synonym_matched = False

        # 1. Normalize keyword and match case-insensitively
        normalized_kw = _normalize_keyword_for_match(cleaned_kw)
        if _exact_match(normalized_kw, resume_normalized):
            matched = True
        
        # 2. Check synonyms for partial phrase matching (with normalization)
        if not matched:
            # Check with normalized key for synonym lookup
            normalized_key = _normalize_keyword_for_match(cleaned_kw)
            if normalized_key in NORMALIZED_SYNONYMS:
                for syn in NORMALIZED_SYNONYMS[normalized_key]:
                    if _exact_match(syn, resume_normalized):
                        matched = True
                        synonym_matched = True
                        break
            elif cleaned_kw in SKILL_SYNONYMS:
                for syn in SKILL_SYNONYMS[cleaned_kw]:
                    normalized_syn = _normalize_keyword_for_match(syn)
                    if _exact_match(normalized_syn, resume_normalized):
                        matched = True
                        synonym_matched = True
                        break

        # 3. Check new SYNONYMS dict for related term matching
        if not matched:
            normalized_key = _normalize_keyword_for_match(cleaned_kw)
            if normalized_key in NORMALIZED_SYNONYMS_2:
                for syn in NORMALIZED_SYNONYMS_2[normalized_key]:
                    if _exact_match(syn, resume_normalized):
                        matched = True
                        synonym_matched = True
                        break
            elif cleaned_kw in SYNONYMS:
                for syn in SYNONYMS[cleaned_kw]:
                    normalized_syn = _normalize_keyword_for_match(syn)
                    if _exact_match(normalized_syn, resume_normalized):
                        matched = True
                        synonym_matched = True
                        break

        if matched:
            matched_keywords.append(kw)
        else:
            # Don't add to missing if it's part of an OR group (will be handled separately)
            if cleaned_kw not in or_group_skills:
                if synonym_matched:
                    soft_missing.append(kw)
                else:
                    hard_missing.append(kw)

    matched_keywords = _remove_subsumed(matched_keywords)
    hard_missing = _remove_subsumed(hard_missing)
    soft_missing = _remove_subsumed(soft_missing)
    
    # Process OR groups - check against already-matched skills list
    or_group_results = []
    matched_set = set(_clean_keyword(kw).lower() for kw in matched_keywords)
    
    for group in or_groups:
        matched_in_group = []
        for skill in group['skills']:
            if _clean_keyword(skill).lower() in matched_set:
                matched_in_group.append(skill)
        or_group_results.append({
            'group_name': group['group_name'],
            'skills': group['skills'],
            'matched': matched_in_group
        })

    return matched_keywords, hard_missing, soft_missing, or_group_results


def calculate_semantic_similarity(resume_text, jd_text):
    def clean(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    resume_clean = clean(resume_text)
    jd_clean = clean(jd_text)
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        print(f"[DEBUG] Similarity method used: TF-IDF cosine, raw score: {score:.2f}")
        return round(score, 2)
    except Exception as e:
        print(f"[DEBUG] Similarity calculation failed: {e}")
        return 0