import spacy

def get_tokenizer(lang):
    if lang == "de":
        return spacy.load("de_core_news_sm")
    return spacy.load("en_core_web_sm")
