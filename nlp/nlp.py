import spacy


class SpacyNLP:
    _instance = None
    def __new__(cls) -> "SpacyNLP":
        if cls._instance is None:
            cls._instance = super(SpacyNLP, cls).__new__(cls)
            cls._instance.nlp = spacy.load("en_core_web_trf")
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
