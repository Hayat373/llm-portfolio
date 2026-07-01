from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ResearcherAgent:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = [
            "Addis Ababa is the capital and largest city of Ethiopia.",
            "Ethiopia is known as the cradle of humanity.",
            "Injera is the national dish of Ethiopia made from teff flour.",
            "Lalibela is famous for its rock-hewn churches.",
            "Coffee was discovered in Ethiopia.",
            "AI and tech startups are growing rapidly in Addis Ababa.",
        ]
        self.embeddings = self.embedder.encode(self.knowledge_base)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def research(self, query):
        query_emb = self.embedder.encode([query])
        _, indices = self.index.search(query_emb, 3)
        results = [self.knowledge_base[i] for i in indices[0]]
        return "\n".join(results)