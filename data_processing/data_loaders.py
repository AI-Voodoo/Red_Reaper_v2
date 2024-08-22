import hashlib
import json
import os
import pandas as pd
from tqdm import tqdm
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
    
    def hash_file(self, text) -> hash:
        if text:
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        return None  
    
    def check_hash_dict(self, text_content, hash_dict=None) -> tuple:
        if hash_dict is None:
            hash_dict ={}
        hash = self.hash_file(text_content)
        if hash not in hash_dict:
            hash_dict[hash] = text_content
            return True, hash_dict
        return False, hash_dict

    def create_data_dict_format(self, score, content) -> dict:
        data = {
            "score": score,
            "content": content
            }
        return data
    
    def prepapre_training_set(self, min_score_threshold, data_path) -> list:
        emails = self.file_ops.load_json(data_path)
        discarded_data = []
        good_data = []
        for email in emails:
            clean_content = email['clean_content']
            content_score = email['content_score']
            if content_score >= min_score_threshold: 
                data1 = self.create_data_dict_format(content_score, clean_content)
                good_data.append(data1)
            else:
                discarded_data.append(email)
        return good_data, discarded_data


    def load_process_email_data(self, samples, csv_path, min_score_threshold, start_seed) -> list:
        stage_1_json_path = "data/stage_1/high_value_emails.json"
        file_skip = self.file_ops.file_exists(stage_1_json_path)
        if file_skip:
            print(f"\n\n* Processed email data already exists {stage_1_json_path}")
            return
        emails = []
        df = self.load_csv_to_df(csv_path) 
        df = df.sample(n=samples, random_state=start_seed)
        #if file_continue:
            #df = self.prune_df(last_email_meta_data, df)

        hash_dict = {}
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing emails"):
            email_content = row[1]

            clean_email_content = self.proc_data.clean_email_content(email_content)
            if len(clean_email_content) >= 1000000:
                continue
            if any(keyword in clean_email_content.lower() for keyword in self.excluded_keywords):
                continue
            email_senders, email_recipients = self.proc_data.recover_email(email_content)
            if not len(email_recipients) >= 1:
                continue
            hash_novel, hash_dict = self.check_hash_dict(clean_email_content, hash_dict)
            if not hash_novel:
                continue
            if clean_email_content:
                law_score = self.embedding_work.compare_text_for_similarity(self.law_target, clean_email_content)
                money_score = self.embedding_work.compare_text_for_similarity(self.money_target, clean_email_content)
                content_score = max(law_score, money_score)
            if content_score <= min_score_threshold:
                continue
            email_data = {
                "email_meta_data": row[0],
                "clean_content": clean_email_content,
                "senders": email_senders,
                "recipients": email_recipients,
                "content_score": float(content_score)
            }
            emails.append(email_data)
            self.file_ops.save_data_to_json(emails, stage_1_json_path)
        
        return emails
