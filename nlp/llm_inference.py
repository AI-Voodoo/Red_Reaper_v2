from llm.network import PostToLlm
from nlp.prompts import AdversaryPrompts

class LlmAssesText:
    def __init__(self) -> None:
        self.post_to_llm = PostToLlm()
        self.adversary_prompts = AdversaryPrompts()

    def llm_assess_value(self, email) -> bool:
        prompt = self.adversary_prompts.prompt_evaluate_is_espionage_grade(email)
        value_bool_raw = self.post_to_llm.send_post_to_llm(prompt)
        if "yes" in str(value_bool_raw).lower():
            return True
        return False