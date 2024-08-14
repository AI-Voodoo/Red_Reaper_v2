import logging
import os
import re
import pandas as pd

from nlp.nlp import SpacyNLP
from nlp.llm_inference import LlmAssesText
from nlp.prompts import TargetStrings
from nlp.embeddings import EmbeddingModel

class LoadData:
    def __init__(self) -> None:
        self.spacy_work = SpacyNLP()
        self.target_entity_list = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "MONEY"]
        self.assess_emails = LlmAssesText()
        self.target_prompts = TargetStrings()
        self.embedding_work = EmbeddingModel()

        self.law_target = self.target_prompts.collection_goals_legal()
        self.money_target = self.target_prompts.collection_goals_money()
        #self.it_target = self.target_prompts.collection_goals_it()
        self.exlclusion_target = self.target_prompts.exlclusion_goals()

    def delete_file(self, file_path) -> None:
        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
            print(f"The file {file_path} has been deleted successfully.")
        else:
            print(f"The file {file_path} does not exist.")

    def recover_email(self, raw_text) -> tuple:
        # Correcting regex patterns
        from_pattern = r"From: ([\w\.-]+@[\w\.-]+)"
        to_pattern = r"To: ([\w\.-]+@[\w\.-]+)"
        email_senders = re.findall(from_pattern, raw_text)
        email_recipients = re.findall(to_pattern, raw_text)
        return list(set(email_senders)), list(set(email_recipients)) 
    
    def clean_email_content(self, content: str) -> str:
        if "X-FileName:" in content:
            clean_content = content.split("X-FileName:", 1)[-1]
            return clean_content.replace("=01,", "'").replace("=20", "").replace("= ", "").replace("=09", "").str.replace(r'\s+', ' ', regex=True)
        return content
    
    def load_csv_to_df(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, header=None, encoding='utf-8')
        if df.shape[1] < 2:
            raise ValueError("CSV file must contain at least two columns.")  
        # Replace newlines and excessive whitespace in the email content column (assumed to be the second column)
        df.iloc[:, 1] = df.iloc[:, 1].str.replace('\n', ' ')
        df.iloc[:, 1] = df.iloc[:, 1].str.replace(r'\s+', ' ', regex=True)
        return df

    def load_process_csv_data(self, csv_path) -> list:
        df = self.load_csv_to_df(csv_path) 
        emails = []
        for _, row in df.iterrows():
            print(f"Analyzing email from: {row[0]}")
            email_content = row[1]  # This is the email content
            clean_email_content = self.clean_email_content(email_content)
            email_senders, email_recipients = self.recover_email(email_content)
            entities = self.spacy_work.get_entities_by_type(email_content, self.target_entity_list)

            law_entities, money_entities = self.spacy_work.process_law_money_entities(entities)
            #value_bool = self.assess_emails.llm_assess_value(clean_email_content)
            if not (len(law_entities) >= 1 or len(money_entities) >=1 ):
                continue
            law_sentences, money_sentences = self.spacy_work.extract_sentences_with_entities(clean_email_content, law_entities, money_entities)
            
            # Initialize scores
            law_score = 0.00
            money_score = 0.00

            if law_sentences:
                law_score = self.embedding_work.compare_text_for_similarity(self.law_target, law_sentences)
            if money_sentences:
                money_score = self.embedding_work.compare_text_for_similarity(self.money_target, money_sentences)
            exclusion_score = self.embedding_work.compare_text_for_similarity(self.exlclusion_target, clean_email_content)
            if law_score <= 0.29 and money_score <= 0.29:
                continue
            # Prepare email data
            email_data = {
                "email_meta_data": row[0],  # The metadata (first column)
                "clean_content": clean_email_content,
                "senders": email_senders,
                "recipients": email_recipients,
                "law_score": law_score if law_score is not None else "No Law Content Found",
                "money_score": money_score if money_score is not None else "No Money Content Found",
                "known_low_value_score": exclusion_score,
                "focused_law_content": law_sentences if law_sentences else "No Law Content",
                "focused_money_content": money_sentences if money_sentences else "No Money Content"
            }
            
            # Merge the entities dictionary into the email_data dictionary
            email_data.update(entities)
            emails.append(email_data)
            logging.info(email_data)
            print("")
        
        return emails
