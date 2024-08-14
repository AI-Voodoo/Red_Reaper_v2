import logging
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    _instance = None
    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.sentence_model = SentenceTransformer("all-mpnet-base-v2")
            cls._instance.device = torch.device("cpu")
        return cls._instance
    
    def __init__(self):
        pass

    def generate_embeddings(self, prompt) -> list: 
        if prompt:
            return self.sentence_model.encode(prompt)    
        return None
  
