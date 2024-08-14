import json
import re
import pandas as pd

from nlp.nlp import SpacyNLP

class LoadData:
    def __init__(self) -> None:
        self.spacy_work = SpacyNLP()
        self.target_entity_list = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "MONEY"]

    def recover_email(self, raw_text) -> tuple:
        # Correcting regex patterns
        from_pattern = r"From: ([\w\.-]+@[\w\.-]+)"
        to_pattern = r"To: ([\w\.-]+@[\w\.-]+)"
        email_senders = re.findall(from_pattern, raw_text)
        email_recipients = re.findall(to_pattern, raw_text)
        return email_senders, email_recipients 

    def load_process_csv(self, csv_path, clean=False) -> list:
        df = pd.read_csv(csv_path, header=None, encoding='utf-8')
        if df.shape[1] < 2:
            raise ValueError("CSV file must contain two columns.")
        df.iloc[:, 1] = df.iloc[:, 1].str.replace('\n', ' ')
        
        # Strip everything before the first occurrence of "X-FileName:"
        if clean:
            df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: x.split("X-FileName:", 1)[-1] if "X-FileName:" in x else x)  
        # Replace excessive whitespace
        data = df.iloc[:, 1].str.replace(r'\s+', ' ', regex=True)
        emails = []
        for index, row in enumerate(data):
            email_senders, email_recipients = self.recover_email(row)
            entities = self.spacy_work.get_entities_by_type(row, self.target_entity_list )

            email_data = {
                "email_meta_data": df.iloc[index, 0],
                "content": row,
                "senders": email_senders,
                "recipients": email_recipients
            }
            
            # Merge the entities dictionary into the email_data dictionary
            email_data.update(entities)
            emails.append(email_data)


        return emails
