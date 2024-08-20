import json
import logging
from sklearn.model_selection import train_test_split
import torch
from sentence_transformers import SentenceTransformer
from nlp.cosine import Cosine
from torch.utils.data import TensorDataset
from nlp.prompts import TargetStrings


class EmbeddingModel:
    _instance = None
    def __new__(cls) -> "EmbeddingModel":
        from data_processing.data_loaders import LoadEmailData, DataPreProcess
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.sentence_model = SentenceTransformer("all-mpnet-base-v2")
            cls._instance.device = torch.device("cpu")
            cls._instance.cosine_work = Cosine()
            cls._instance.loader_email = LoadEmailData()
            cls._instance.proc_data = DataPreProcess()
        return cls._instance
    
    def __init__(self):
        self.target_prompts = TargetStrings()
        self.law_target = self.target_prompts.collection_goals_legal()
        self.money_target = self.target_prompts.collection_goals_money()

    def generate_embeddings(self, prompt, normalize=False) -> list: 
        if prompt:
            if normalize:
                return self.sentence_model.encode(prompt, normalize_embeddings=True)
            else:
                 return self.sentence_model.encode(prompt)   
        return None
    
    def compare_text_for_similarity(self, text1, text2) -> float:
        text1_embedding = self.generate_embeddings(text1)
        text2_embedding = self.generate_embeddings(text2)
        return self.cosine_work.get_cosine_similarity(text1_embedding, text2_embedding)
    
    def train_test_set_embeddings(self, path) -> tuple:
        with open(path, 'r') as file:
            data = json.load(file)
        embeddings = []
        for item in data:
            content = item['content']
            embedding = self.generate_embeddings(content)
            embeddings.append(torch.tensor(embedding, dtype=torch.float32)) 
        embeddings_tensor = torch.stack(embeddings)
        train_embeddings, test_embeddings = train_test_split(embeddings_tensor, test_size=0.2, random_state=42)

        # Convert to TensorDataset if needed
        train_dataset = TensorDataset(torch.tensor(train_embeddings, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32))
        return train_dataset, val_dataset
    

    def test_ae_classificaton_load_set(self, csv_path, sample_amount) -> tuple:
        df = self.loader_email.load_csv_to_df(csv_path) 
        df = df.sample(n=sample_amount)
        contents = []
        embeddings = []
        law_score_list = []
        money_score_list = []

        for _, row in df.iterrows():
            law_score = 0.00
            money_score = 0.00

            email_content = row[1]
            content = self.proc_data.clean_email_content(email_content)
            if not content:
                continue
            law_score = self.compare_text_for_similarity(content, self.law_target)
            money_score = self.compare_text_for_similarity(content, self.money_target)
            embedding = self.generate_embeddings(content)
            law_score_list.append(law_score)
            money_score_list.append(money_score)
     
            contents.append(content)
            embeddings.append(torch.tensor(embedding, dtype=torch.float32))
        embeddings_tensor = torch.stack(embeddings)
        return contents, embeddings_tensor, law_score_list, money_score_list
    
