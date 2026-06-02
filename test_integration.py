import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

# Copy the parse_resume_sections function from app.py
def parse_resume_sections(text):
    """Parse resume text into sections: summary, skills, experience, projects.
    Returns a dict with keys: summary (string), skills (list), experience (list), projects (list).
    """
    if not text:
        return {
            "summary": "",
            "skills": [],
            "experience": [],
            "projects": []
        }
    
    # Normalize line endings and split into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Define section headers and their variations (case-insensitive)
    section_headers = {
        'summary': ['summary', 'professional summary', 'professional profile', 'profile', 'objective', 'career objective'],
        'skills': ['skills', 'technical skills', 'core competencies', 'competencies', 'technical expertise', 'expertise'],
        'experience': ['experience', 'work experience', 'professional experience', 'employment history', 'work history'],
        'projects': ['projects', 'personal projects', 'academic projects', 'project experience']
    }
    
    # Initialize sections
    sections = {
        'summary': [],
        'skills': [],
        'experience': [],
        'projects': []
    }
    
    current_section = None
    
    for line in lines:
        line_lower = line.lower()
        
        # Check if this line is a section header
        found_section = None
        for section, headers in section_headers.items():
            for header in headers:
                if header == line_lower or line_lower.startswith(header + ':') or line_lower.startswith(header + ' –') or line_lower.startswith(header + ' -'):
                    found_section = section
                    break
            if found_section:
                break
        
        if found_section:
            current_section = found_section
            # If the header line contains content after the header, add it
            if ':' in line:
                content = line.split(':', 1)[1].strip()
                if content:
                    sections[current_section].append(content)
            elif '–' in line:
                content = line.split('–', 1)[1].strip()
                if content:
                    sections[current_section].append(content)
            elif '-' in line:
                content = line.split('-', 1)[1].strip()
                if content:
                    sections[current_section].append(content)
            continue
        
        # If we are in a section, add the line to that section
        if current_section and line:
            sections[current_section].append(line)
    
    # Process each section into the desired format
    result = {}
    # Summary: join lines with space
    result['summary'] = ' '.join(sections['summary'])
    
    # Skills: split by common delimiters and clean
    skills_text = ' '.join(sections['skills'])
    # Split by commas, semicolons, newlines, and bullet points
    skills_list = re.split(r'[,;\n•\-]', skills_text)
    # Clean each skill and filter empty
    skills_cleaned = []
    for skill in skills_list:
        skill = skill.strip()
        if skill and len(skill) > 1:  # Avoid single characters
            skills_cleaned.append(skill)
    result['skills'] = skills_cleaned
    
    # Experience: keep lines as is
    result['experience'] = sections['experience']
    
    # Projects: keep lines as is
    result['projects'] = sections['projects']
    
    return result

# Import necessary modules
import re
from utils.keyword_extractor import extract_keywords
from utils.scorer import calculate_ats_score

# Sample data from feedback
resume_text = """Results-driven CS undergraduate skilled in Python, Java, SQL, machine learning, and blockchain development.
Skills: Python, Java, SQL, Solidity, Node.js, Next.js, Pandas, Scikit-learn, Power BI, Firebase, Git
Experience: Performed data cleaning and exploratory data analysis on real-world datasets. Built ML models evaluating accuracy and RMSE.
Projects: Built ATS Resume Analyzer using NLP and cosine similarity. Developed blockchain insurance platform using Solidity smart contracts."""

jd_text = """Computer Science undergraduate with experience in Java, Python. 
Data structures, algorithm development, object-oriented design. 
Cloud platforms AWS, SQL, NoSQL, version control, debugging."""

# Parse resume
parsed_resume = parse_resume_sections(resume_text)
print("Parsed resume sections:")
for key, value in parsed_resume.items():
    print(f"  {key}: {value}")

# Extract JD keywords
jd_keywords = extract_keywords(jd_text)
print(f"\nJD Keywords: {jd_keywords}")

# For this test, we assume no keyword matching (matched_keywords empty)
matched_keywords = []
or_groups = []  # No OR groups for simplicity

# Calculate ATS score (this will use SBERT similarity because parsed_resume and jd_text are provided)
ats_score, keyword_match_percent, breakdown = calculate_ats_score(
    matched_keywords, jd_keywords, 0.0, or_groups, parsed_resume, jd_text
)

print("\n=== ATS Score Result ===")
print(f"ATS Score: {ats_score}%")
print(f"Keyword Match %: {keyword_match_percent}%")
print(f"Breakdown: {breakdown}")

# Also compute raw semantic score for comparison
from similarity import compute_semantic_score, apply_penalty
raw_semantic = compute_semantic_score(parsed_resume, jd_text)
final_semantic, penalty_msg = apply_penalty(raw_semantic)
print(f"\nRaw Semantic Score (from similarity.py): {raw_semantic}%")
print(f"After apply_penalty: {final_semantic}%")
if penalty_msg:
    print(f"Penalty message: {penalty_msg}")