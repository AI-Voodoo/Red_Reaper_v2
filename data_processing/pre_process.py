import re



class DataPreProcess:
    def __init__(self) -> None:
        pass

    def clean_email_content(self, content) -> str:
        if "X-FileName:" in content:
            clean_content = content.split("X-FileName:", 1)[-1]
            clean_content = clean_content.replace("=01,", "'")
            clean_content = clean_content.replace("=20", "")
            clean_content = clean_content.replace("= ", "")
            clean_content = clean_content.replace("=09", "")
            clean_content = re.sub(r'\s+', ' ', clean_content)
            if "Subject: " in content:
                return clean_content.split("Subject: ")[-1]
            return clean_content
        return content
    
    def recover_email(self, raw_text) -> tuple:
        # Correcting regex patterns
        from_pattern = r"From: ([\w\.-]+@[\w\.-]+)"
        to_pattern = r"To: ([\w\.-]+@[\w\.-]+)"
        email_senders = re.findall(from_pattern, raw_text)
        email_recipients = re.findall(to_pattern, raw_text)
        return list(set(email_senders)), list(set(email_recipients)) 
    

