import pdfplumber
import docx
import re


# ---------------------------------------------------------------------------
# Low-level cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Aggressively clean text for keyword extraction / matching:
      - Lowercase
      - Strip special characters (keep alphanumeric + spaces)
      - Collapse whitespace
    
    ⚠️  Do NOT use this for semantic similarity — it degrades transformer
        embeddings. Use normalize_text() instead.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_text(text: str) -> str:
    """
    Light normalization for semantic similarity:
      - Preserve original casing (sentence transformers are case-sensitive)
      - Preserve punctuation (helps the model parse sentence boundaries)
      - Only collapse excessive whitespace / blank lines
      - Remove non-printable / control characters

    This keeps the text readable by a sentence transformer without the noise
    of PDF layout artefacts (random newlines, hyphenation, etc.).
    """
    if not text:
        return ""
    # Remove non-printable control characters (but keep \n, space, punctuation)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    # Collapse 3+ consecutive newlines into two (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse runs of spaces/tabs into a single space within a line
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip leading/trailing whitespace per line
    text = '\n'.join(line.strip() for line in text.splitlines())
    # Final strip
    return text.strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path_or_bytes) -> tuple[str, str]:
    """
    Extract text from a PDF file.

    Returns:
        (cleaned_text, normalized_text)
        - cleaned_text    : lowercase, no punctuation — for keyword matching
        - normalized_text : original casing + punctuation — for semantic similarity
    """
    raw = ""
    try:
        with pdfplumber.open(file_path_or_bytes) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    raw += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")

    return clean_text(raw), normalize_text(raw)


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def extract_text_from_docx(file_path_or_bytes) -> tuple[str, str]:
    """
    Extract text from a DOCX file.

    Returns:
        (cleaned_text, normalized_text)
    """
    raw = ""
    try:
        doc = docx.Document(file_path_or_bytes)
        for paragraph in doc.paragraphs:
            if paragraph.text:
                raw += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")

    return clean_text(raw), normalize_text(raw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_text(file, file_type: str) -> tuple[str, str]:
    """
    Main entry point for text extraction.

    Args:
        file      : File path string or file-like object (e.g. Streamlit UploadedFile).
        file_type : 'pdf' or 'docx'

    Returns:
        (cleaned_text, normalized_text)

        cleaned_text    — aggressively cleaned, lowercase, no punctuation.
                          Use this for keyword extraction and matching.

        normalized_text — lightly cleaned, preserves casing and punctuation.
                          Use this for semantic similarity (sentence transformers).

    Raises:
        ValueError if file_type is not 'pdf' or 'docx'.
    """
    file_type = file_type.lower().strip()

    if file_type == 'pdf':
        return extract_text_from_pdf(file)
    elif file_type == 'docx':
        return extract_text_from_docx(file)
    else:
        raise ValueError(
            f"Unsupported file type: '{file_type}'. Supported types: 'pdf', 'docx'."
        )