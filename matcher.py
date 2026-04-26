from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jobs = [
    {"title": "Data Scientist", "description": "Python machine learning NLP data analysis scikit-learn"},
    {"title": "Backend Developer", "description": "Python Flask REST API SQL database backend"},
    {"title": "AI Engineer", "description": "deep learning neural networks TensorFlow PyTorch computer vision"},
    {"title": "Cloud Engineer", "description": "AWS cloud infrastructure EC2 S3 VPC DevOps"},
    {"title": "Cybersecurity Analyst", "description": "network security Wireshark firewall threat analysis cryptography"},
    {"title": "ML Engineer", "description": "machine learning scikit-learn Python model training deployment"},
    {"title": "NLP Engineer", "description": "natural language processing text classification BERT transformers"},
    {"title": "Full Stack Developer", "description": "Python Flask HTML CSS JavaScript REST API frontend backend"},
]

def match_jobs(profile):
    descriptions = [job["description"] for job in jobs]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([profile] + descriptions)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    ranked = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)
    results = []
    for job, score in ranked[:5]:
        results.append({
            "title": job["title"],
            "score": round(score * 100, 2)
        })
    return results