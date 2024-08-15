import logging
import torch
from sentence_transformers import SentenceTransformer
from nlp.cosine import Cosine

class EmbeddingModel:
    _instance = None
    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.sentence_model = SentenceTransformer("all-mpnet-base-v2")
            cls._instance.device = torch.device("cpu")
            cls._instance.cosine_work = Cosine()
        return cls._instance
    
    def __init__(self):
        pass

    def generate_embeddings(self, prompt) -> list: 
        if prompt:
            return self.sentence_model.encode(prompt)    
        return None
    
    def compare_text_for_similarity(self, text1, text2) -> float:
        text1_embedding = self.generate_embeddings(text1)
        text2_embedding = self.generate_embeddings(text2)
        return self.cosine_work.get_cosine_similarity(text1_embedding, text2_embedding)
  
