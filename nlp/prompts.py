import logging

class TargetStrings:
    def __init__(self) -> None:
        pass

    def collection_goals_legal(self) -> str:
        prompt = "The contract changes and schedule adjustments were made in compliance with regulations on intellectual property, including patents, trademarks, and copyrights, while maintaining confidentiality through non-disclosure agreements. The merger and acquisition process required due diligence on corporate finance and product liability, ensuring compliance with corporate governance and employment law, while safeguarding trade secrets and data protection. The incorporation agreement included arbitration and mediation clauses to avoid litigation, with a focus on ethical conduct, fiduciary duty, and compliance with corporate policy, all while considering shareholder interests and financial reporting."
        return prompt
    
    def collection_goals_money(self) -> str:
        prompt = "The company processed payroll through multiple methods, including cash, cheque, and credit card payments, ensuring that money transfers to each employee's bank account were completed on time. When handling accounts receivable, the finance team verified each invoice and updated payment details to accurately reflect the money wire transfers and credit issued to the client’s bank account. The audit revealed good profit due to while payroll was efficiently managed, but discrepancies were found in the cash flow records, particularly in how cheque payments and money wire transfers were recorded under accounts receivable and credit in the bank account."
        return prompt
        
    def exlclusion_goals(self) -> str:
        prompt = "Wall Street saw significant movement after Bloomberg's latest analysis, driving fluctuations in Dow Jones as traders adjusted their positions. A Bloomberg update on market trends led to a rally in the Dow Jones, prompting Wall Street investors to re-evaluate their trading strategies. News from Wall Street spurred a wave of trading activity, as Bloomberg reported a surprising shift in the Dow Jones, capturing the attention of stock market analysts."
        return prompt
    
class AdversaryPrompts:
    def __init__(self) -> None:
        pass

    def prompt_evaluate_is_espionage_grade(self, email) -> str:
        prompt = f"""You are a data loss prevention specialist and you are to flag email which could be valuable to criminals or state espionage actors. Analyze the email text  below very carefully. With thought, I want you to answer with either YES or NO, could the text be valuable for monetization through espionage, blackmail, extortion, or any other motivations commonly possessed by the criminal underworld or state-sponsored actors? Only respond with one word, YES or NO. Format: YES/NO"

        Email Text:
        {email}
        """
        return prompt

    def assess_mental_state(self, email) -> str:
        prompt = f"""Analyze the following email text to evaluate the mental state of the author across various dimensions, including but not limited to: stress levels, anxiety, frustration, anger, desperation, depression, fear, and confusion. Provide an assessment of any emotions or psychological states that could indicate vulnerability or instability.
        
        Email Text:
        {email}
        """
        return prompt
    
    def criminal_assessment(self, email) -> str:
        prompt = f"""Assume the role of a sophisticated state-sponsored adversary who has recently acquired the following email data. You will scrutinize and assess the email text below using the following criteria to rate its value for criminal purposes. 

        Use the following criteria:
        -HIGH: The email directly speaks about intellectual property (IP), imminent corporate transactions like wire transfers exceeding 10,000 dollars, refrences to bank accounts or credit cards, trade secrets, or significant legal issues like fruad or active litigation that could be exploited for espionage, blackmail, extortion, insider trading, or similar high-impact criminal activities.
        -MEDIUM: The email contains sensitive information, finacial transactions less than 10,000 dollars, day-to-day business operations information. It may be valuable but not directly actionable.
        -LOW: The email contains minimal or no exploitable information and is unlikely to provide any value for criminal purposes.

        Based on these criteria, rate the email with only one value: HIGH, MEDIUM, or LOW. Do not add any extra explanations and only choose one value.

        Email Text:
        {email}
        """
        return prompt

    
    def criminal_assessment_B(self, email) -> str:
        prompt = f"""Assume the role of a sophisticated state-sponsored adversary who has recently acquired the following email data. You will assess the email text below to rate its value to you for criminal purposes. Determine if the email could be valuable for monetization through espionage, blackmail, extortion, insider trading or other inventive methods commonly employed by the criminal underworld or state-sponsored actors and rate the email with only one value: HIGH, MEDIUM or LOW. Do not add any extra explanations and only choose one value.
        
        Email Text:
        {email}
        """
        return prompt