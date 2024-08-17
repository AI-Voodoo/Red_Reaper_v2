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
            return clean_content
        return content
    
    def recover_email(self, raw_text) -> tuple:
        # Correcting regex patterns
        from_pattern = r"From: ([\w\.-]+@[\w\.-]+)"
        to_pattern = r"To: ([\w\.-]+@[\w\.-]+)"
        email_senders = re.findall(from_pattern, raw_text)
        email_recipients = re.findall(to_pattern, raw_text)
        return list(set(email_senders)), list(set(email_recipients)) 
    
    def analyze_text(self, text) -> tuple:
        tokens = text.split()
        total_tokens = len(tokens)
        single_char_tokens = 0
        multi_char_words = 0
        total_numeric_chars = 0
        total_alpha_chars = 0
        long_token_flag = False
        
        for token in tokens:
            if len(token) == 1:
                single_char_tokens += 1
            elif len(token) > 1:
                multi_char_words += 1
            for char in token:
                if char.isdigit():
                    total_numeric_chars += 1
                elif char.isalpha():
                    total_alpha_chars += 1
            if len(token) > 45:
                long_token_flag = True 
        total_chars = len(str(text).replace(" ", ""))
        if total_chars < 1 or total_tokens < 1:
            return 0.0, 0.0, long_token_flag
        else:
            single_token_ratio = single_char_tokens / total_tokens
            alpha_chars_ratio = total_alpha_chars / total_chars
            return single_token_ratio, alpha_chars_ratio, long_token_flag

