import logging
import os
import re
import pandas as pd

from nlp.nlp import SpacyNLP
from nlp.llm_inference import LlmAssesText

class LoadData:
    def __init__(self) -> None:
        self.spacy_work = SpacyNLP()
        self.target_entity_list = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "MONEY"]
        self.assess_emails = LlmAssesText()

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
        return email_senders, email_recipients 
    
    def clean_email_content(self, content: str) -> str:
        if "X-FileName:" in content:
            return content.split("X-FileName:", 1)[-1]
        elif "cc: " in content:
            return content.split("cc: ", 1)[-1]
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
            value_bool = self.assess_emails.llm_assess_value(clean_email_content)
            if not (value_bool or len(law_entities) >= 1 or len(money_entities) >=1 ):
                print("Skipping...")
                continue
            
            print("BOOM! Found some.")
            email_data = {
                "email_meta_data": row[0],  # The metadata (first column)
                #"raw_content": email_content, 
                "clean_content": clean_email_content, 
                "senders": email_senders,
                "recipients": email_recipients
            }
            
            # Merge the entities dictionary into the email_data dictionary
            email_data.update(entities)
            emails.append(email_data)
            logging.info(email_data)
            print("")
        
        return emails
