import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
import re

# Copy the parse_resume_sections function from app.py (since we cannot import app.py due to Streamlit)
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
        'experience': ['experience', 'work experience', 'professional experience', 'employment history', 'work history', 'internships', 'internships and simulations', 'internship', 'employment'],
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

# Now import the necessary functions from the utils modules
from utils.parser import clean_text, normalize_text
from utils.keyword_extractor import extract_keywords
from utils.matcher import match_keywords
from utils.scorer import calculate_ats_score

# Resume text from the prompt
resume_text = """
Hemapriya J
Chennai, India | 9150517594 | hemapriyajagadeesh11@gmail.com

SUMMARY
Results-driven Computer Science undergraduate with hands-on experience 
in full-stack development, blockchain applications, and data-driven 
projects. Skilled in Python, Java, and SQL, with a strong foundation 
in DSA, machine learning, and smart contract development to build 
efficient and scalable solutions.

SKILLS
Languages: Python, Java, SQL, Solidity
Frameworks & Libraries: Node.js, Next.js, Express.js, Pandas, 
Scikit-learn, NLP
Tools & Platforms: Power BI, Firebase, Truffle, Ganache, Google Colab, 
Git, Excel

INTERNSHIPS
PROPGROWTHX | Remote, Data Analyst and ML Intern | SEP 2025 - OCT 2025
Performed data cleaning, merging, and exploratory data analysis on 
real-world datasets using Python, delivering structured analytical 
reports with actionable conclusions. Built basic ML models in Google 
Colab, evaluating performance metrics including accuracy and RMSE.

PROJECTS
ATS Resume Analyzer | Python, NLP, Machine Learning | MAR 2026
Developed an ATS Resume Analyzer that parses and evaluates resumes 
against job descriptions using NLP techniques, extracting keywords 
and scoring relevance to optimize candidate shortlisting efficiency.

Campus Event Hub | Next.js, Express, Firebase | JAN 2025
Developed a full-stack College Event Calendar supporting role-based 
access and event registration, with real-time database and 
authentication integration.

ChainInsure | Solidity, Truffle | JUL 2025
Built a blockchain-based insurance claims platform using Solidity 
smart contracts to automate claim verification and reduce fraud risk.

EDUCATION
Bachelor of Engineering in Computer Science and Engineering
with Specialization in Blockchain Technology - 8.5 CGPA
Sathyabama Institute of Science and Technology | Aug 2023 - Jun 2027

CERTIFICATIONS
Data Analysis using Excel - Coursera (2025)
Introduction to Machine Learning - NPTEL (2025)
Data Science Methodology - Cognitive Class AI (2024)
"""

# JD text from the prompt
jd_text = """
BASIC QUALIFICATIONS
Must be 18 years of age or older
Currently enrolled in Bachelor's degree or above in Computer Science, 
Computer Engineering, Data Science, Information Systems, or related 
STEM fields.
Demonstrated experience with at least one general-purpose programming 
language such as Java, Python, C++, C#, Go, Rust, or TypeScript
Demonstrated experience in one or more of the following:
Data structures implementation
Basic algorithm development
Object-oriented design principles

PREFERRED QUALIFICATIONS
Previous technical internship(s) or demonstrated project experience
Experience with one or more of the following:
AI tools for development productivity
Cloud platforms preferably AWS
Database systems SQL and NoSQL
Contributing to open-source projects
Version control systems
Debugging and troubleshooting complex systems
Strong problem-solving and analytical skills
Excellent written and verbal communication skills
Demonstrated ability to learn and adapt to new technologies quickly
Basic understanding of software development lifecycle SDLC
"""

# Step 1: Parse resume into sections
parsed_resume = parse_resume_sections(resume_text)
print("Parsed resume sections:")
for key, value in parsed_resume.items():
    print(f"  {key}: {value}")
print(f"\nExperience section length: {len(parsed_resume['experience'])} lines")

# Step 2: Get cleaned and normalized resume text for keyword matching and semantic similarity
resume_cleaned = clean_text(resume_text)
resume_normalized = normalize_text(resume_text)

# Step 3: Extract JD keywords
jd_keywords = extract_keywords(jd_text)
print(f"\nJD Keywords: {jd_keywords}")

# Step 4: Match keywords (using cleaned resume text)
matched, hard_missing, soft_missing, or_groups = match_keywords(resume_cleaned, jd_keywords, jd_text)
print(f"\nMatched keywords: {matched}")
print(f"Hard Missing keywords: {hard_missing}")
print(f"Soft Missing keywords: {soft_missing}")
print(f"OR groups: {or_groups}")

# Step 5: Calculate ATS score (this will use SBERT similarity because parsed_resume and jd_text are provided)
ats_score, keyword_match_percent, breakdown = calculate_ats_score(
    matched, jd_keywords, 0.0, or_groups, parsed_resume, jd_text
)

print("\n=== ATS Score Result ===")
print(f"ATS Score (overall %): {ats_score}%")
print(f"Keyword Match %: {keyword_match_percent}%")
print(f"Semantic Similarity %: {breakdown['similarity_score']}%")
penalty_applied = breakdown['penalty_applied']
if penalty_applied > 0:
    print(f"Penalty applied (yes/no + amount): yes, {penalty_applied} points")
else:
    print("Penalty applied (yes/no + amount): no")

print("\n=== Score Breakdown ===")
print(f"Keyword Match: {keyword_match_percent}% × {breakdown['keyword_weight']} = {keyword_match_percent * breakdown['keyword_weight']:.2f}")
print(f"Semantic Similarity: {breakdown['similarity_score']}% × {breakdown['similarity_weight']} = {breakdown['similarity_score'] * breakdown['similarity_weight']:.2f}")
print(f"Format Score: {breakdown['format_score']}% × {breakdown['format_weight']} = {breakdown['format_score'] * breakdown['format_weight']:.2f}")
print(f"Penalty: -{penalty_applied}")

# Format matched and missing skills for output
# We'll use the matched and missing from the matcher (which are keyword strings)
matched_skills = ", ".join(matched) if matched else "None"
hard_missing_skills = ", ".join(hard_missing) if hard_missing else "None"
soft_missing_skills = ", ".join(soft_missing) if soft_missing else "None"
print(f"Matched Skills: {matched_skills}")
print(f"Hard Missing Skills: {hard_missing_skills}")
print(f"Soft Missing Skills: {soft_missing_skills}")

# Generate suggestions
from utils.suggestions import generate_suggestions
missing = hard_missing + soft_missing
suggestions = generate_suggestions(missing, matched, breakdown, resume_text, or_groups)
print("\n=== Suggestions ===")
for i, suggestion in enumerate(suggestions, 1):
    print(f"{i}. {suggestion}")