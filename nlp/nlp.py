import logging
import spacy

class SpacyNLP:
    _instance = None
    def __new__(cls) -> "SpacyNLP":
        if cls._instance is None:
            cls._instance = super(SpacyNLP, cls).__new__(cls)
            cls._instance.nlp = spacy.load("en_core_web_lg")
        return cls._instance
    
    def __init__(self):
        pass

    def get_entities_by_type(self, text, entity_types) -> dict:
        doc = self.nlp(text)
        entities = {entity_type: [] for entity_type in entity_types}
        for ent in doc.ents:
            if ent.label_ in entity_types:
                entities[ent.label_].append(ent.text)  
        return entities
    
    def process_law_money_entities(self, entity_dict) -> tuple:
        law_entities = entity_dict.get("LAW", [])
        money_entities = entity_dict.get("MONEY", [])
        return law_entities, money_entities
    
    def extract_sentences_with_entities(self, text, law_entities, money_entities) -> tuple:
        law_sentences = []
        money_sentences = []
        doc = self.nlp(text)
        law_sentences = [sentence.text for sentence in doc.sents if any(law_entity in sentence.text for law_entity in law_entities)]
        money_sentences = [sentence.text for sentence in doc.sents if any(money_entity in sentence.text for money_entity in money_entities)]
        return " ".join(list(set(law_sentences))), " ".join(list(set(money_sentences)))