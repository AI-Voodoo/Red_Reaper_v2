from llm.network import PostToLlm
from nlp.prompts import AdversaryPrompts

class LlmAssesText:
    def __init__(self) -> None:
        self.post_to_llm = PostToLlm()
        self.adversary_prompts = AdversaryPrompts()

    def llm_assess_value(self, email) -> int:
        prompt = self.adversary_prompts.criminal_assessment(email)
        value_bool_raw = self.post_to_llm.send_post_to_llm(prompt)
        if "high" in str(value_bool_raw).lower():
            return 3
        if "medium" in str(value_bool_raw).lower():
            return 2
        if "low" in str(value_bool_raw).lower():
            return 1
        return 0