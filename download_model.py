import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

from sentence_transformers import SentenceTransformer

print("Downloading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded, saving locally...")
model.save('./models/all-MiniLM-L6-v2')
print("Model saved successfully")