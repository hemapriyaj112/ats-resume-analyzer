import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()

# NLTK stopwords as replacement for spaCy STOP_WORDS
STOP_WORDS = set(stopwords.words('english'))

# ── Words that are NEVER skills ────────────────────────────────────────────
NOISE_WORDS = STOP_WORDS | {
    "strong", "old", "older", "new", "general", "open", "complex", "good",
    "excellent", "great", "high", "low", "large", "small", "various",
    "multiple", "preferred", "preferably", "required", "relevant", "related",
    "solid", "proven", "demonstrated", "hands",
    "year", "years", "field", "ability", "previous", "purpose", "source",
    "basic", "datum", "role", "position", "opportunity", "candidate",
    "team", "company", "business", "work", "job", "requirement",
    "qualification", "responsibility", "education",
    "master", "age", "plus", "experience", "knowledge",
    "understanding", "skill", "skills", "statement", "policy", "benefit",
    "compensation", "culture", "value", "environment", "including", "etc",
    "use", "way", "tool", "tools", "system", "systems",
    "language", "languages", "technology", "technologies", "platform",
    "platforms", "lifecycle", "cycle", "principle", "principles",
    "productivity", "information", "object", "stem", "area", "areas",
    "solution", "solutions", "implementation", "management", "process",
    "processes", "problems", "software",
    "design", "designs", "cloud", "database", "databases",
    "application", "applications",
    "service", "services", "model", "models", "network", "networks",
    "testing", "test", "code", "coding", "program", "programs",
    "programming", "development", "analysis", "data",
    # Fine as part of bigrams but noisy solo
    "structure", "structures", "control", "version",
    # Extra HR noise that leaks through NER / noun chunks
    "engineering", "requirement", "requirements", "qualification",
    "qualifications", "enrol", "enrollment",
}

# NLTK NE chunk labels that correspond to useful entity types
VALID_NE_TYPES = {"ORGANIZATION", "GPE", "FACILITY", "GSP"}

TECH_WHITELIST = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "rust", "go",
    "sql", "nosql", "html", "css", "bash", "scala", "kotlin", "swift",
    # Frameworks & libraries
    "react", "node", "nodejs", "django", "flask", "fastapi", "spring",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    # Cloud & infra
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform",
    "git", "github", "gitlab", "linux", "unix",
    # Data & ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "data science", "data engineering", "data analysis",
    "spark", "kafka", "airflow", "dbt", "hadoop",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "firebase",
    # Protocols & patterns
    "rest", "graphql", "grpc", "api", "microservices",
    # Practices & concepts
    "ci/cd", "devops", "agile", "scrum",
    "object-oriented", "oop", "sdlc", "tdd", "version control",
    # CS fundamentals
    "data structures", "algorithm", "algorithm development", "debugging",
    # Conceptual skills
    "internship", "internship(s", "internships", "intern",
    "problem-solving", "problem solving", "analytical skills", "analytical",
    "communication", "communication skills",
    "project experience", "projects", "project",
    "technical internship", "technical skills", "soft skills",
    "computer science", "bachelor", "bachelor's", "degree", "computer science degree",
    "database systems", "version control systems",
    "ai tools", "google colab",
}

GENERIC_PHRASES = {
    "software development", "software lifecycle",
    "open source", "cloud platforms", "database systems",
    "problem solving", "analytical skills", "communication skills",
    "written communication", "verbal communication",
}

# Ambiguous short skills that need stricter matching (not substrings)
AMBIGUOUS_SKILLS = {"go", "c", "r", "rust", "swift", "kotlin"}

# Synonym map for skill matching - if resume contains any synonym, count as matched
SKILL_SYNONYMS = {
    "version control": ["git", "github", "gitlab", "svn"],
    "version control systems": ["git", "github", "gitlab"],
    "ai tools": ["machine learning", "ml", "artificial intelligence", "nlp", "scikit-learn", "google colab"],
    "debugging": ["troubleshooting", "bug fixing", "error resolution", "debug", "debugging"],
    "object oriented": ["oop", "object-oriented", "classes", "inheritance"],
    "nosql": ["firebase", "mongodb", "dynamodb"],
    "cloud platforms": ["aws", "azure", "gcp", "google cloud"],
    "project experience": ["projects", "built", "developed"],
    "technical internship": ["internship", "intern"],
    "object-oriented": ["oop", "object oriented", "classes", "inheritance"],
    "problem-solving": ["problem solving", "analytical", "analytical skills"],
    "bachelor's": ["bachelor", "degree", "computer science"],
    "bachelor": ["bachelor's", "degree", "computer science"],
    "degree": ["bachelor", "bachelor's", "computer science"],
    "algorithm development": ["algorithm", "algorithms", "dsa"],
    "data structures": ["dsa", "algorithms"],
    "sdlc": ["software development lifecycle", "development lifecycle"],
}

PROTECTED_TERMS = [
    "c++", "c#", "typescript", "node.js",
    "next.js", "express.js", "scikit-learn"
]

SYNONYMS = {
    "version control": ["git", "github", "gitlab"],
    "debugging": ["debug", "debugged", "debugs", "troubleshoot", "troubleshooting", "troubleshot", "fixed bugs", "bug fixing", "resolved issues", "diagnosed"],
    "troubleshooting": ["debugging", "debug", "debugged", "debugs", "troubleshoot", "troubleshot", "fixed bugs", "bug fixing", "resolved issues", "diagnosed"],
    "object oriented": ["oop", "object-oriented", "classes"],
    "machine learning": ["ml", "scikit-learn", "sklearn"],
    "database": ["sql", "nosql", "firebase", "mongodb"],
    "cloud platform": ["aws", "azure", "gcp", "cloud"],
    "algorithm development": ["dsa", "data structures", "algorithms"],
    "sdlc": ["software development lifecycle", "agile", "scrum"]
}


def _validate_ambiguous_skill(skill: str, text: str, skills_section: str = "") -> bool:
    """
    Validate that ambiguous short skills are not false positives.
    Returns True only if the skill appears as a standalone word (not substring).
    For skills in AMBIGUOUS_SKILLS, requires word boundary match.
    """
    if skill not in AMBIGUOUS_SKILLS:
        return True

    pattern = rf"\b{re.escape(skill)}\b"
    return bool(re.search(pattern, text, re.IGNORECASE))

DISPLAY_OVERRIDES = {
    "sql": "SQL", "nosql": "NoSQL", "api": "API", "html": "HTML",
    "css": "CSS", "oop": "OOP", "sdlc": "SDLC", "tdd": "TDD",
    "aws": "AWS", "gcp": "GCP", "nlp": "NLP", "rest": "REST",
    "grpc": "gRPC", "llm": "LLM", "ml": "ML", "ai": "AI",
    "ui": "UI", "ux": "UX", "cv": "CV", "dl": "DL",
    "k8s": "k8s", "ci/cd": "CI/CD",
    "c++": "C++", "c#": "C#", "typescript": "TypeScript",
    "javascript": "JavaScript", "python": "Python", "java": "Java",
    "rust": "Rust", "go": "Go", "scala": "Scala", "kotlin": "Kotlin",
    "swift": "Swift", "bash": "Bash",
    "nodejs": "Node.js", "node": "Node.js", "reactjs": "React",
    "react": "React", "django": "Django", "flask": "Flask",
    "fastapi": "FastAPI", "spring": "Spring", "tensorflow": "TensorFlow",
    "pytorch": "PyTorch", "scikit-learn": "Scikit-learn",
    "pandas": "Pandas", "numpy": "NumPy", "docker": "Docker",
    "kubernetes": "Kubernetes", "terraform": "Terraform",
    "git": "Git", "github": "GitHub", "gitlab": "GitLab",
    "linux": "Linux", "unix": "Unix", "kafka": "Kafka",
    "airflow": "Airflow", "spark": "Spark", "hadoop": "Hadoop",
    "redis": "Redis", "mongodb": "MongoDB", "mysql": "MySQL",
    "postgresql": "PostgreSQL", "elasticsearch": "Elasticsearch",
    "graphql": "GraphQL", "dbt": "dbt", "azure": "Azure",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "computer vision": "Computer Vision", "data science": "Data Science",
    "data engineering": "Data Engineering", "data analysis": "Data Analysis",
    "version control": "Version Control", "object-oriented": "Object-Oriented",
    "ci cd": "CI/CD", "devops": "DevOps", "agile": "Agile",
    "scrum": "Scrum", "microservices": "Microservices",
    "data structures": "Data Structures",
    "algorithm": "Algorithm", "debugging": "Debugging",
    "open source": "Open Source", "cloud platforms": "Cloud Platforms",
    "database systems": "Database Systems", "problem solving": "Problem Solving",
    "software development": "Software Development",
    "analytical skills": "Analytical Skills",
    "algorithm development": "Algorithm Development",
    "firebase": "Firebase",
    "internship": "Internship", "internship(s": "Internship(s", "internships": "Internships",
    "intern": "Intern",
    "problem-solving": "Problem-Solving",
    "analytical": "Analytical",
    "communication": "Communication", "communication skills": "Communication Skills",
    "project experience": "Project Experience", "projects": "Projects", "project": "Project",
    "technical internship": "Technical Internship", "technical skills": "Technical Skills",
    "soft skills": "Soft Skills",
    "computer science": "Computer Science", "bachelor": "Bachelor", "bachelor's": "Bachelor's",
    "degree": "Degree", "computer science degree": "Computer Science Degree",
    "version control systems": "Version Control Systems",
    "ai tools": "AI Tools", "google colab": "Google Colab",
}

# ── Pure section-header lines — dropped entirely
_HEADER_LINE = re.compile(
    r"^\s*(basic qualifications?|minimum qualifications?|preferred qualifications?|"
    r"education requirements?|equal opportunity|eeo statement|compensation|"
    r"benefits?|about us|who we are|our values?|what we offer|why join|perks?|"
    r"responsibilities|what you.?ll do|what you will do|"
    r"legal|privacy policy|disclaimer)\s*$",
    re.IGNORECASE,
)

# ── Sections where ALL content is skipped
_SKIP_SECTION = re.compile(
    r"^\s*(equal opportunity|eeo statement|compensation|benefits?|"
    r"about us|who we are|our values?|what we offer|why join|perks?|"
    r"legal|privacy policy|disclaimer)\s*$",
    re.IGNORECASE,
)

# ── Patterns that disqualify a candidate phrase entirely
_BAD_FRAGMENT = re.compile(
    r"\bqualifications?\b"
    r"|\brequirements?\b"
    r"|\binternships?\b"
    r"|\(s\b"
    r"|\bengineer(ing)?\b"
    r"|\benroll(ment|ed)?\b"
    r"|\bmust\b|\bage\b"
    r"|\bolder\b|\byears?\b",
    re.IGNORECASE,
)

_EDUCATION_FIELD = re.compile(
    r"computer\s+(science|engineering)|"
    r"data\s+science\s+information|"
    r"information\s+systems?|"
    r"stem\s+field",
    re.IGNORECASE,
)


def _strip_boilerplate(text: str) -> str:
    """
    - Drop pure section-header lines so their words don't pollute extraction.
    - Skip entire irrelevant sections until the next blank line.
    - Keep content inside qualification/responsibility sections.
    """
    lines = text.split("\n")
    cleaned = []
    skip = False

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            skip = False
            cleaned.append(line)
            continue

        if _SKIP_SECTION.match(stripped):
            skip = True
            continue

        if _HEADER_LINE.match(stripped):
            continue

        if not skip:
            cleaned.append(line)

    return "\n".join(cleaned)


def _clean_phrase(text: str) -> str:
    """Clean phrase by removing punctuation artifacts and normalizing."""
    text = text.lower().strip()
    if text in PROTECTED_TERMS:
        return text
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_noise(phrase: str) -> bool:
    """Returns True if phrase should be discarded."""
    cleaned = re.sub(r"[^\w\s\-]", "", phrase).strip()

    if len(cleaned) < 3:
        return True

    words = cleaned.split()

    if all(w in NOISE_WORDS for w in words):
        return True
    if len(words) == 1 and words[0] in NOISE_WORDS:
        return True
    if _BAD_FRAGMENT.search(phrase):
        return True
    if _EDUCATION_FIELD.search(phrase):
        return True
    if len(words) > 3:
        return True

    return False


def _format_keyword(kw: str) -> str:
    kw_lower = kw.lower().strip()
    if kw_lower in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[kw_lower]
    words = kw_lower.split()
    if len(words) > 1:
        return " ".join(DISPLAY_OVERRIDES.get(w, w.capitalize()) for w in words)
    return kw_lower.capitalize()


def _normalize_keyword(kw: str) -> str:
    """Normalize keyword: lowercase, strip punctuation, lemmatize."""
    cleaned = _clean_phrase(kw)
    words = cleaned.split()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized_words)


def _deduplicate(keywords: set) -> set:
    bigrams = {kw for kw in keywords if len(kw.split()) > 1}
    unigrams_to_drop = set()
    for bigram in bigrams:
        for part in bigram.split():
            if part in keywords:
                unigrams_to_drop.add(part)
    return keywords - unigrams_to_drop


def _extract_ne_chunks(text: str) -> set:
    """
    Replaces spaCy NER (Step 2).
    Uses NLTK's ne_chunk to extract named entities of useful types.
    """
    keywords = set()
    try:
        sentences = sent_tokenize(text)
        for sent in sentences:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            chunked = ne_chunk(tagged, binary=False)
            for subtree in chunked:
                if isinstance(subtree, Tree) and subtree.label() in VALID_NE_TYPES:
                    entity = " ".join(word.lower() for word, tag in subtree.leaves())
                    kw = _clean_phrase(entity)
                    if kw and not _is_noise(kw) and len(kw) > 2:
                        keywords.add(kw)
    except Exception:
        pass
    return keywords


def _extract_noun_bigrams(text: str) -> set:
    """
    Replaces spaCy noun chunks (Step 3).
    Extracts 2-word noun phrases using NLTK POS tags (NN, NNS, NNP, NNPS).
    """
    keywords = set()
    noun_tags = {"NN", "NNS", "NNP", "NNPS"}
    try:
        sentences = sent_tokenize(text)
        for sent in sentences:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            for i in range(len(tagged) - 1):
                w1, t1 = tagged[i]
                w2, t2 = tagged[i + 1]
                if t1 in noun_tags and t2 in noun_tags:
                    l1 = lemmatizer.lemmatize(w1.lower())
                    l2 = lemmatizer.lemmatize(w2.lower())
                    if l1 not in NOISE_WORDS and l2 not in NOISE_WORDS:
                        phrase = _clean_phrase(f"{l1} {l2}")
                        if not _is_noise(phrase):
                            keywords.add(phrase)
    except Exception:
        pass
    return keywords


def _extract_nouns(text: str) -> set:
    """
    Replaces spaCy single NOUN/PROPN tokens (Step 4).
    Extracts individual noun tokens using NLTK POS tags.
    """
    keywords = set()
    noun_tags = {"NN", "NNS", "NNP", "NNPS"}
    try:
        sentences = sent_tokenize(text)
        for sent in sentences:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            for word, tag in tagged:
                if tag in noun_tags:
                    lemma = lemmatizer.lemmatize(word.lower())
                    if lemma not in NOISE_WORDS and len(lemma) > 2:
                        keywords.add(lemma)
    except Exception:
        pass
    return keywords


def extract_keywords(text: str) -> list[str]:
    """
    Extracts skill/technology keywords from a job description or resume.

    Steps:
      1. Strip boilerplate (header lines + irrelevant sections)
      2. Whitelist scan — catches known multi-word tech terms reliably
      3. NLTK NER — organization/location entities (replaces spaCy NER)
      4. Noun bigrams — 2-word noun pairs, both words must pass noise filter
      5. Single noun tokens — noise-filtered
      6. Deduplicate — remove unigrams subsumed by bigrams
      7. Remove generic HR phrases
      8. Normalize keywords (lowercase, strip punctuation, lemmatize)
    """
    if not text:
        return []

    stripped = _strip_boilerplate(text)
    keywords = set()
    text_lower = stripped.lower()

    # Step 1 — whitelist scan (most reliable for multi-word tech terms)
    for term in TECH_WHITELIST:
        escaped = re.escape(term)
        if term in AMBIGUOUS_SKILLS:
            if re.search(rf"\b{escaped}\b", text_lower):
                keywords.add(term)
        else:
            if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text_lower):
                keywords.add(term)

    # Step 2 — NLTK NER (replaces spaCy NER)
    keywords |= _extract_ne_chunks(stripped)

    # Step 3 — Noun bigrams (replaces spaCy noun chunks)
    keywords |= _extract_noun_bigrams(stripped)

    # Step 4 — Single noun tokens (replaces spaCy NOUN/PROPN tokens)
    keywords |= _extract_nouns(stripped)

    # Step 5 — deduplicate and strip generic phrases
    keywords = _deduplicate(keywords)
    keywords = keywords - GENERIC_PHRASES

    # Step 6 — normalize keywords (lowercase, strip punctuation, lemmatize)
    keywords = {_normalize_keyword(kw) for kw in keywords}

    return list(keywords)