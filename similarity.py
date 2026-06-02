import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('./models/all-MiniLM-L6-v2')

def compute_semantic_score(resume_sections: dict, jd_text: str) -> float:
    weights = {
        "summary": 0.15,
        "skills": 0.25,
        "experience": 0.35,
        "projects": 0.25
    }
    total = 0.0
    for section, weight in weights.items():
        text = resume_sections.get(section, "")
        # Ensure text is a string; if it's a list, join with space
        if isinstance(text, list):
            text = " ".join(text)
        elif not isinstance(text, str):
            text = str(text)
        
        if text:
            section_embedding = model.encode(text, convert_to_tensor=True)
            jd_embedding = model.encode(jd_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(section_embedding, jd_embedding)
            total += weight * similarity.item()
        # If section is empty, contribution is 0 (implicitly)
    return round(total * 100, 2)

def apply_penalty(score: float) -> tuple[float, str]:
    if score < 15:
        adjusted = score - 11
        return (adjusted, "Penalty applied: -11 points for low score")
    else:
        return (score, "")

if __name__ == "__main__":
    resume_sections = {
        "summary": "Results-driven CS undergraduate skilled in Python, Java, SQL, machine learning, and blockchain development.",
        "skills": "Python, Java, SQL, Solidity, Node.js, Next.js, Pandas, Scikit-learn, Power BI, Firebase, Git",
        "experience": "Performed data cleaning and exploratory data analysis on real-world datasets. Built ML models evaluating accuracy and RMSE.",
        "projects": "Built ATS Resume Analyzer using NLP and cosine similarity. Developed blockchain insurance platform using Solidity smart contracts."
    }
    jd_text = """Computer Science undergraduate with experience in Java, Python. 
Data structures, algorithm development, object-oriented design. 
Cloud platforms AWS, SQL, NoSQL, version control, debugging."""

    score = compute_semantic_score(resume_sections, jd_text)
    final_score, penalty_msg = apply_penalty(score)
    print(f"Raw Semantic Score: {score}%")
    print(f"Final Score after penalty: {final_score}%")
    if penalty_msg:
        print(f"Penalty applied: {penalty_msg}")