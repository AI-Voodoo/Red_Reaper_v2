import logging
from sklearn.metrics.pairwise import cosine_similarity

class CosineSimilarity:
    def __init__(self) -> None:
        pass
    
    def get_cosine_similarity(self, embedding1, embedding2) -> float:
        cosine = cosine_similarity([embedding1], [embedding2])[0][0]
        return cosine