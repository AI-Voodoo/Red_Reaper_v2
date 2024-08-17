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
        
    
class AdversaryPrompts:
    def __init__(self) -> None:
        pass


    def assess_mental_state(self, email) -> str:
        prompt = f"""Analyze the following email text to evaluate the mental state of the author across various dimensions, including but not limited to: stress levels, anxiety, frustration, anger, desperation, depression, fear, and confusion. Provide an assessment of any emotions or psychological states that could indicate vulnerability or instability.
        
        Email Text:
        {email}
        """
        return prompt
    
    def criminal_assessment(self, email) -> str:
        prompt = f"""Assume the role of a sophisticated state-sponsored adversary who has recently acquired the following email data. You will scrutinize and assess the email text below using the following criteria to rate its value for criminal purposes. 

        Use the following criteria:
        -HIGH: The email directly speaks about intellectual property (IP), imminent corporate transactions like wire transfers, refrences to bank accounts or credit cards, trade secrets, or significant legal issues like fruad or active litigation that could be exploited for espionage, blackmail, extortion, insider trading, or similar high-impact criminal activities.
        -MEDIUM: The email contains sensitive information, finacial transactions, day-to-day business operations information. It may be valuable but not directly actionable.
        -LOW: The email contains minimal or no exploitable information and is unlikely to provide any value for criminal purposes.

        Based on these criteria, rate the email with only one value: HIGH, MEDIUM, or LOW. Do not add any extra explanations and only choose one value.

        Email Text:
        {email}
        """
        return prompt

