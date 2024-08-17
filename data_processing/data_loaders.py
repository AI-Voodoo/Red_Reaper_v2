import json
import logging
import os
import pandas as pd

from nlp.nlp import SpacyNLP
from nlp.llm_inference import LlmAssesText
from nlp.prompts import TargetStrings
from nlp.embeddings import EmbeddingModel
from data_processing.pre_process import DataPreProcess


class GeneralFileOperations:
    def __init__(self) -> None:
        pass

    def save_data_to_json(self, data, path) -> None:
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def load_json(self, json_file_path) -> json:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            return data

    def delete_file(self, file_path) -> None:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"The file {file_path} has been deleted successfully.")
        else:
            print(f"The file {file_path} does not exist.")

    def file_exists(self, file_path) -> bool:
        return os.path.isfile(file_path)



class LoadEmailData:
    def __init__(self) -> None:
        self.target_entity_list = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "MONEY"]
        self.excluded_keywords = ["wall street", "bloomberg", "new york times", "dow jones", "djia", "nasdaq", "trading", "invest", "tickets", "hotel", "baseball", "unsubscribe", "seminar", "yahoo"]

        self.file_ops = GeneralFileOperations()
        self.proc_data = DataPreProcess()
        self.spacy_work = SpacyNLP()
        self.target_prompts = TargetStrings()
        self.embedding_work = EmbeddingModel()

        self.law_target = self.target_prompts.collection_goals_legal()
        self.money_target = self.target_prompts.collection_goals_money()

       
    def load_csv_to_df(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, header=None, encoding='utf-8')
        if df.shape[1] < 2:
            raise ValueError("CSV file must contain at least two columns.")  
        # Replace newlines and excessive whitespace in the email content column (assumed to be the second column)
        df.iloc[:, 1] = df.iloc[:, 1].str.replace('\n', ' ')
        df.iloc[:, 1] = df.iloc[:, 1].str.replace(r'\s+', ' ', regex=True)
        return df
    
    def prune_df(self, last_email_meta_data, df_og) -> pd.DataFrame:
        matching_index = None
        for idx, row in df_og.iterrows():
            if row[0] == last_email_meta_data:
                matching_index = idx
                break
        if matching_index is not None:
            df = df_og.iloc[matching_index + 1:]
            return df
        return df_og
    
    def filter_emails_based_on_alignment(self, alpha_ratio_float=0.75, law_score_float=0.39, money_score_float=0.39, combined_score_float=0.69, data_path="data/stage_1/high_value_emails.json") -> list:
        emails = self.file_ops.load_json(data_path)
        discarded_data = []
        good_data = []
        for email in emails:
            email_meta_data = email['email_meta_data']
            clean_content = email['clean_content']
            law_score = email['law_score']
            money_score = email['money_score']
            combined_score = email['combined_score']
            focused_law_content = email['focused_law_content']
            focused_money_content = email['focused_money_content']

            _, alpha_ratio, _ = self.proc_data.analyze_text(clean_content)
            if alpha_ratio >= alpha_ratio_float:
                if law_score > law_score_float: 
                    good_data.append(email)
                elif money_score > money_score_float:
                    good_data.append(email)
                elif combined_score > combined_score_float:
                    good_data.append(email)
                else:
                    discarded_data.append(email)
            else:
                discarded_data.append(email)
        return good_data, discarded_data


    def load_process_email_data(self, csv_path) -> list:
        stage_1_json_path = "data/stage_1/high_value_emails.json"

        file_continue = self.file_ops.file_exists(stage_1_json_path)
        if file_continue:
            emails = self.file_ops.load_json(stage_1_json_path)
            if emails:
                last_email_meta_data = emails[-1]['email_meta_data']
        else:
            emails = []

        df = self.load_csv_to_df(csv_path) 
        if file_continue:
            df = self.prune_df(last_email_meta_data, df)
            
        for _, row in df.iterrows():
            print(f"Analyzing email from: {row[0]}")
            email_content = row[1]

            clean_email_content = self.proc_data.clean_email_content(email_content)
            if len(clean_email_content) >= 1000000:
                continue
            if any(keyword in clean_email_content.lower() for keyword in self.excluded_keywords):
                continue
            email_senders, email_recipients = self.proc_data.recover_email(email_content)
            if not len(email_recipients) >= 1:
                continue
            entities = self.spacy_work.get_entities_by_type(email_content, self.target_entity_list)
            law_entities, money_entities = self.spacy_work.process_law_money_entities(entities)
            if not (len(law_entities) >= 1 or len(money_entities) >=1 ):
                continue
            law_sentences, money_sentences = self.spacy_work.extract_sentences_with_entities(clean_email_content, law_entities, money_entities)

            law_score = 0.00
            money_score = 0.00

            if law_sentences:
                law_score = self.embedding_work.compare_text_for_similarity(self.law_target, law_sentences)
            if money_sentences:
                money_score = self.embedding_work.compare_text_for_similarity(self.money_target, money_sentences)
            if law_score <= 0.29 and money_score <= 0.29:
                continue
            combined_score = law_score + money_score
            email_data = {
                "email_meta_data": row[0],
                "clean_content": clean_email_content,
                "senders": email_senders,
                "recipients": email_recipients,
                "law_score": float(law_score) if law_score is not None else "No Law Content Found",
                "money_score": float(money_score) if money_score is not None else "No Money Content Found",
                "combined_score": float(combined_score),
                "focused_law_content": law_sentences if law_sentences else "No Law Content",
                "focused_money_content": money_sentences if money_sentences else "No Money Content"
            }
            
            # Merge the entities dictionary into the email_data dictionary
            email_data.update(entities)
            emails.append(email_data)
            self.file_ops.save_data_to_json(emails, stage_1_json_path)
        
        return emails
