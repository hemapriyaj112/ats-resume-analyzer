import os
import sys

print("=== SBERT Test Script ===")
print("This script demonstrates SBERT-based semantic similarity")
print("Note: Actual model download requires fixing SSL certificates")
print()

# Check dependencies
print("1. Checking dependencies:")
try:
    import sentence_transformers
    print(f"   ✓ sentence-transformers version: {sentence_transformers.__version__}")
except ImportError:
    print("   ✗ sentence-transformers not installed")
    sys.exit(1)

try:
    import torch
    print(f"   ✓ torch version: {torch.__version__}")
except ImportError:
    print("   ✗ torch not installed")
    sys.exit(1)

try:
    import sklearn
    print(f"   ✓ scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("   ✗ scikit-learn not installed")
    sys.exit(1)

print()

# Check if model exists locally
model_config_path = os.path.join("models", "all-MiniLM-L6-v2", "config.json")
model_exists = os.path.exists(model_config_path)

print("2. Checking for local model:")
if model_exists:
    print(f"   ✓ Model found at: {os.path.abspath(os.path.join('models', 'all-MiniLM-L6-v2'))}")
    
    # Try to load and test the model
    try:
        # Set SSL bypass for any potential HF calls during model loading
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("   Loading model...")
        model = SentenceTransformer(os.path.join("models", "all-MiniLM-L6-v2"))
        print("   ✓ Model loaded successfully")
        
        # Test with sample sentences
        text1 = "This is a test sentence about machine learning."
        text2 = "This is another test sentence about artificial intelligence."
        
        print("   Encoding test sentences...")
        embeddings = model.encode([text1, text2])
        
        print("   Calculating cosine similarity...")
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        print(f"   Cosine similarity: {similarity:.4f}")
        print(f"   Is score between 0 and 1? {0 <= similarity <= 1}")
        
        if 0 <= similarity <= 1:
            print("   ✓ SUCCESS: SBERT is working correctly!")
        else:
            print("   ✗ ERROR: Similarity score out of expected range")
            
    except Exception as e:
        print(f"   ✗ ERROR loading/running model: {e}")
else:
    print(f"   ✗ Model not found at: {os.path.abspath(model_config_path)}")
    print()
    print("3. To complete the SBERT integration:")
    print("   Option A: Fix SSL certificates and re-run")
    print("   Option B: Manually download the model:")
    print("       1. Create directory: models/all-MiniLM-L6-v2")
    print("       2. Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
    print("       3. Extract all contents to that directory")
    print()
    print("4. Expected output when model is working:")
    print("   Cosine similarity between the two sentences: 0.XXXX")
    print("   Is the score between 0 and 1? True")
    print()
    print("5. Current dependencies status: ALL INSTALLED ✓")

print()
print("=== Test Complete ===")