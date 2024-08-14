import logging

class TargetStrings:
    def __init__(self) -> None:
        pass

    def collection_goals_legal() -> str:
        prompt = "The contract changes and schedule adjustments were made in compliance with regulations on intellectual property, including patents, trademarks, and copyrights, while maintaining confidentiality through non-disclosure agreements. The merger and acquisition process required due diligence on corporate finance and product liability, ensuring compliance with corporate governance and employment law, while safeguarding trade secrets and data protection. The incorporation agreement included arbitration and mediation clauses to avoid litigation, with a focus on ethical conduct, fiduciary duty, and compliance with corporate policy, all while considering shareholder interests and financial reporting."
        return prompt
    
    def collection_goals_money() -> str:
        prompt = "The company processed payroll through multiple methods, including cash, cheque, and credit card payments, ensuring that money transfers to each employee's bank account were completed on time. When handling accounts receivable, the finance team verified each invoice and updated payment details to accurately reflect the money wire transfers and credit issued to the client’s bank account. The audit revealed that while payroll was efficiently managed, discrepancies were found in the cash flow records, particularly in how cheque payments and money wire transfers were recorded under accounts receivable and credit in the bank account."
        return prompt
    
    def collection_goals_it() -> str:
        prompt = "The security team enforced multi-factor authentication for login procedures, requiring complex passwords to access the cloud server and database, all while ensuring secure connections through the VPN and firewall. The helpdesk escalated several tickets related to password resets and login failures, which were causing delays in accessing the virtual machines, network drives, and enterprise database through the VPN. To bolster cybersecurity, we are deploying a new server cluster with enhanced encryption protocols, integrating the database with real-time monitoring systems, and updating the VPN configurations to support more secure remote login, with the helpdesk available 24/7 for any connectivity or password issues."
        return prompt
        
    def exlclusion_goals() -> str:
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

    def assess_mental_state(email) -> str:
        prompt = f"""Analyze the following email text to evaluate the mental state of the author across various dimensions, including but not limited to: stress levels, anxiety, frustration, anger, desperation, depression, fear, and confusion. Provide an assessment of any emotions or psychological states that could indicate vulnerability or instability.
        
        Email Text:
        {email}
        """
        return prompt
    
    def criminal_assessment(email) -> str:
        prompt = f"""Assume the role of a sophisticated state-sponsored adversary who has recently acquired the following email data. You are assessing indentified important sentences of the email, not the whole email. Determine which pieces of information could be valuable for monetization through espionage, blackmail, extortion, or other inventive methods commonly employed by the criminal underworld and rate the vaue accordingly, HIGH, MEDIUM or LOW value with explanation.
        
        Email Text:
        {email}
        """