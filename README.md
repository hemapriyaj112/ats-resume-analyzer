---

# 🎯 ATS Resume Analyzer

An intelligent ATS (Applicant Tracking System) Resume Analyzer that 
scores your resume against a job description using real semantic 
understanding — not just keyword matching.

## 🚀 Live Demo
> Coming soon after deployment

## 📸 Screenshot
> Add screenshot of results page here

## ✨ Features

- **SBERT Semantic Similarity** — Uses Sentence-BERT (all-MiniLM-L6-v2) 
  for meaning-based matching, not just word overlap
- **Section-Weighted Scoring** — Experience (35%), Skills (25%), 
  Projects (25%), Summary (15%)
- **OR-Group Skill Matching** — Understands "Java OR Python OR C++" 
  like a real recruiter
- **AI-Powered Suggestions** — Groq API (Llama 3.1) gives specific, 
  resume-grounded improvement tips
- **Score Breakdown Chart** — Visual breakdown of keyword, semantic, 
  and format contributions
- **ATS Gauge Chart** — Color-coded score gauge with recruiter verdict
- **Penalty System** — Applies score penalty when semantic similarity 
  is critically low

## 🧠 How Scoring Works

| Component | Weight |
|---|---|
| Keyword Match | 40% |
| Semantic Similarity (SBERT) | 50% |
| Format Score | 10% |

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Semantic Scoring | Sentence-Transformers (all-MiniLM-L6-v2) |
| Keyword Extraction | NLP + custom OR-group resolver |
| AI Suggestions | Groq API (llama-3.1-8b-instant) |
| PDF Parsing | pdfplumber |
| Visualization | Plotly |
| Environment | python-dotenv |

## 📁 Project Structure
ats_resume_analyzer/
├── app.py                  # Streamlit frontend
├── run_ats.py              # CLI runner
├── models/
│   └── all-MiniLM-L6-v2/  # Local SBERT model
├── utils/
│   ├── parser.py           # Resume PDF parser
│   ├── matcher.py          # Keyword + OR-group matcher
│   ├── scorer.py           # ATS scoring engine
│   ├── similarity.py       # SBERT semantic scoring
│   └── suggestions.py      # Groq AI suggestions
├── .env                    # API keys (not committed)
├── requirements.txt
└── README.md

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ats-resume-analyzer.git
cd ats-resume-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SBERT model:
```bash
python -c "
import ssl, os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', 
                  local_dir='./models/all-MiniLM-L6-v2')
"
```

4. Create a `.env` file:
GROQ_API_KEY=your_groq_api_key_here

Get your free Groq API key at: https://console.groq.com

5. Run the app:
```bash
streamlit run app.py
```

## 🔑 Environment Variables

| Variable | Description | Where to Get |
|---|---|---|
| GROQ_API_KEY | Groq API key for AI suggestions | console.groq.com |

## 📊 Sample Output

- **ATS Score:** 46.82%
- **Keyword Match:** 53.57%
- **Semantic Similarity:** 36.78%
- **Missing Skills:** debugging
- **Matched Skills:** Python, Java, SQL, Data Structures, 
  Algorithm Development, and 10 more

## 🙋‍♀️ Author

**Hemapriya J**  
Computer Science undergraduate | Blockchain Technology  
Sathyabama Institute of Science and Technology, Chennai  
[LinkedIn](https://linkedin.com/in/hemapriya-j6b227828b)

---
⭐ If this helped you, consider starring the repo!

---