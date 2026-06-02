import streamlit as st
import time
import re
import plotly.graph_objects as go

from utils.parser import extract_text, clean_text, normalize_text
from utils.keyword_extractor import extract_keywords, _format_keyword
from utils.matcher import match_keywords, _deduplicate_keywords
from utils.scorer import calculate_ats_score
from utils.suggestions import generate_suggestions
from similarity import compute_semantic_score, apply_penalty

def render_score_breakdown(ats_score, keyword_match, semantic_score):
    format_score = 70.0  # fixed for now
    
    # Weighted contributions
    keyword_contribution = round(keyword_match * 0.40, 2)
    semantic_contribution = round(semantic_score * 0.50, 2)
    format_contribution = round(format_score * 0.10, 2)
    
    categories = ["Keyword Match", "Semantic Similarity", "Format Score"]
    contributions = [keyword_contribution, semantic_contribution, format_contribution]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=contributions,
            marker_color=colors,
            text=[f"{v:.1f}pts" for v in contributions],
            textposition="outside",
            width=0.4
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Score Breakdown — How Your ATS Score Was Calculated",
            font=dict(size=14)
        ),
        yaxis=dict(
            title="Points Contributed to ATS Score",
            range=[0, 55]
        ),
        xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False
    )
    
    return fig

def render_ats_gauge(ats_score):
    if ats_score >= 75:
        color = "#4CAF50"
    elif ats_score >= 55:
        color = "#8BC34A"
    elif ats_score >= 35:
        color = "#FF9800"
    else:
        color = "#F44336"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ats_score,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 35], "color": "#FFEBEE"},
                {"range": [35, 55], "color": "#FFF3E0"},
                {"range": [55, 75], "color": "#F1F8E9"},
                {"range": [75, 100], "color": "#E8F5E9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": ats_score
            }
        },
        title={"text": "ATS Score", "font": {"size": 14}}
    ))
    
    fig.update_layout(
        height=260,
        margin=dict(t=30, b=30, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 24px;
        color: #e67e22;
        margin-top: 20px;
        border-bottom: 2px solid #e67e22;
        padding-bottom: 5px;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📄 ATS Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Optimize your resume for Applicant Tracking Systems (ATS) and land your dream job!</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">1. Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Must be a PDF or DOCX file.", type=["pdf", "docx"], label_visibility="collapsed")

with col2:
    st.markdown('<div class="section-header">2. Target Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area("Paste the job description here.", height=150, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
analyze_button = st.button("🔍 Analyze ATS Score", use_container_width=True)

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

if analyze_button:
    if uploaded_file is None:
        st.error("Please upload your resume first.")
    elif not jd_text.strip():
        st.error("Please paste the job description.")
    else:
        with st.spinner("Extracting hidden keywords and computing score..."):
            file_type = uploaded_file.name.split('.')[-1].lower()
            try:
                # extract_text now returns (cleaned_text, normalized_text)
                # cleaned  → keyword extraction + matching (lowercase, no punctuation)
                # normalized → semantic similarity (original casing + punctuation)
                resume_cleaned, resume_normalized = extract_text(uploaded_file, file_type)
            except Exception as e:
                st.error(f"Error extracting text from file: {e}")
                resume_cleaned, resume_normalized = "", ""
            
            if resume_cleaned:
                # JD comes in as a raw string from st.text_area
                # Apply the same two-path treatment
                jd_normalized = normalize_text(jd_text)          # for semantic similarity
                jd_keywords   = extract_keywords(jd_text)         # raw text → extractor handles its own cleaning
                
                if not jd_keywords:
                    st.warning("Could not extract technical skills or nouns from the job description.")
                else:
                    # 3. Match keywords (use cleaned resume text)
                    matched, hard_missing, soft_missing, or_groups = match_keywords(resume_cleaned, jd_keywords, jd_text)
                    missing = hard_missing + soft_missing
                    
                    # Apply keyword deduplication
                    matched = _deduplicate_keywords(matched)
                    
                    # 4. Semantic similarity (use normalized texts — preserves casing + punctuation
                    #    so the sentence transformer produces meaningful embeddings)
                    parsed_resume = parse_resume_sections(resume_normalized)
                    resume_sections = {
                        "summary": parsed_resume.get("summary", ""),
                        "skills": " ".join(parsed_resume.get("skills", [])) if isinstance(parsed_resume.get("skills"), list) else parsed_resume.get("skills", ""),
                        "experience": " ".join(parsed_resume.get("experience", [])) if isinstance(parsed_resume.get("experience"), list) else parsed_resume.get("experience", ""),
                        "projects": " ".join(parsed_resume.get("projects", [])) if isinstance(parsed_resume.get("projects"), list) else parsed_resume.get("projects", "")
                    }
                    raw_similarity = compute_semantic_score(resume_sections, jd_text)
                    similarity_score, penalty_msg = apply_penalty(raw_similarity)
                    
                    # 5. ATS score
                    ats_score, keyword_match_percent, breakdown = calculate_ats_score(
                        matched, jd_keywords, similarity_score, or_groups, parsed_resume, jd_text
                    )
                    
                    # 6. Suggestions
                    suggestions = generate_suggestions(missing, matched, breakdown, resume_normalized, or_groups)
                    
                    time.sleep(1.5)
                    st.success("Analysis Complete!")
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">ATS Analysis Results</div>', unsafe_allow_html=True)
                    
                    # Recruiter Verdict - recalibrated thresholds
                    if ats_score >= 75:
                        st.success(
                            "**Recruiter Verdict**: Strong match. This resume is well-aligned with the role."
                        )
                    elif ats_score >= 55:
                        st.info(
                            "**Recruiter Verdict**: Good match with some gaps. A few targeted improvements recommended."
                        )
                    elif ats_score >= 35:
                        st.warning(
                            "**Recruiter Verdict**: Partial match. Address the missing skills and reframe experience to align better."
                        )
                    else:
                        st.error(
                            "**Recruiter Verdict**: Low match. Significant rework needed to target this role."
                        )
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.plotly_chart(
                            render_ats_gauge(ats_score),
                            use_container_width=True
                        )
                    with metrics_col2:
                        st.metric(label="Keyword Match", value=f"{keyword_match_percent}%")
                    with metrics_col3:
                        st.metric(label="Semantic Similarity", value=f"{similarity_score}%")
                    
                    # Show penalty message if it exists
                    if penalty_msg:
                        st.warning(penalty_msg)
                    
                    st.plotly_chart(
                        render_score_breakdown(ats_score, keyword_match_percent, similarity_score),
                        use_container_width=True
                    )
                    
                    normalized_score = min(max(ats_score / 100.0, 0.0), 1.0)
                    st.progress(normalized_score)
                    if ats_score >= 80:
                        st.balloons()
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Calculate total skills and matched count
                    # Use the actual lists so the count matches the displayed skills
                    total_skills = len(set(jd_keywords))
                    matched_count = len(set(matched))
                    
                    # Build OR group skills set (needed for missing skills display)
                    or_group_skills = set()
                    for og in or_groups:
                        or_group_skills.update(s.lower() for s in og.get('skills', []))
                    
                    st.subheader(f"🎯 Skill Coverage: {matched_count} / {total_skills} Skills Matched")
                    coverage_fraction = matched_count / total_skills if total_skills > 0 else 0.0
                    st.progress(coverage_fraction)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    skills_col1, skills_col2 = st.columns(2)
                    with skills_col1:
                        st.markdown(f"**✅ Matched Skills ({matched_count})**")
                        if matched:
                            matched_tags = " • ".join(
                                _format_keyword(kw) for kw in sorted(set(matched))
                            )
                            st.info(matched_tags)
                        else:
                            st.warning("No critical skills matched.")
                    
                    with skills_col2:
                        # Calculate missing count including OR groups
                        or_group_missing_count = sum(1 for og in or_groups if not og.get('matched'))
                        individual_missing = [m for m in missing if m.lower() not in or_group_skills]
                        missing_count = len(set(individual_missing)) + or_group_missing_count
                        
                        st.markdown(f"**❌ Missing Skills ({missing_count})**")
                        if missing or or_group_missing_count > 0:
                            # Display OR groups that have matches (satisfied)
                            for or_group in or_groups:
                                if or_group.get('matched'):
                                    skills_str = ", ".join(_format_keyword(s) for s in or_group['skills'])
                                    matched_str = ", ".join(_format_keyword(s) for s in or_group['matched'])
                                    st.info(f"✅ {or_group['group_name']} — satisfied by: {matched_str}")
                            
                            # Display OR groups that have no matches
                            for or_group in or_groups:
                                if not or_group.get('matched'):
                                    skills_str = ", ".join(_format_keyword(s) for s in or_group['skills'])
                                    st.error(f"❌ {or_group['group_name']} — none of ({skills_str}) found")
                            
                            # Display individual missing skills (excluding OR group skills)
                            if individual_missing:
                                ranked_missing = sorted(set(individual_missing), key=lambda x: (-len(x.split()), x))
                                display_missing = " • ".join(
                                    _format_keyword(kw) for kw in ranked_missing
                                )
                                st.error(display_missing)
                        else:
                            st.success("Great job! No major skills missing.")
                    
                    st.markdown("---")
                    st.subheader("💡 Improvement Suggestions")
                    for suggestion in suggestions:
                        st.markdown(suggestion)