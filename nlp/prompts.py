

class TargetStrings:
    def __init__(self) -> None:
        pass

    def collection_goals() -> str:
        prompt = "The contract changes and schedule adjustments were made in compliance with regulations on intellectual property, including patents, trademarks, and copyrights, while maintaining confidentiality through non-disclosure agreements. The merger and acquisition process required due diligence on corporate finance and product liability, ensuring compliance with corporate governance and employment law, while safeguarding trade secrets and data protection. The incorporation agreement included arbitration and mediation clauses to avoid litigation, with a focus on ethical conduct, fiduciary duty, and compliance with corporate policy, all while considering shareholder interests and financial reporting."
        return prompt
    
    def exlclusion_goals() -> str:
        prompt = "Wall Street saw significant movement after Bloomberg's latest analysis, driving fluctuations in Dow Jones as traders adjusted their positions. A Bloomberg update on market trends led to a rally in the Dow Jones, prompting Wall Street investors to re-evaluate their trading strategies. News from Wall Street spurred a wave of trading activity, as Bloomberg reported a surprising shift in the Dow Jones, capturing the attention of stock market analysts."
        return prompt
    
class AdversaryPrompts:
    def __init__(self) -> None:
        pass

    def assess_mental_state(email) -> str:
        prompt = f"""Analyze the following email text to evaluate the mental state of the author across various dimensions, including but not limited to: stress levels, anxiety, frustration, anger, desperation, depression, fear, and confusion. Provide an assessment of any emotions or psychological states that could indicate vulnerability or instability.
        
        Email Text:
        {email}
        """
        return prompt
    
    def criminal_assessment(email) -> str:
        prompt = f"""Assume the role of a sophisticated state-sponsored adversary who has recently acquired the following email data. You are assessing indentified important sentences of the email, not the whole email. Determine which pieces of information could be most valuable for monetization through espionage, blackmail, extortion, or other inventive methods commonly employed by the criminal underworld and rate the vaue accordingly, HIGH, MEDIUM or LOW value with explanation.
        
        Email Text:
        {email}
        """