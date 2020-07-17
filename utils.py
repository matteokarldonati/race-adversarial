import random

import spacy
import nltk
from nltk.corpus import names

nlp = spacy.load("en_core_web_sm")

nltk.download('names')

MALE_NAMES = names.words('male.txt')
FEMALE_NAMES = names.words('female.txt')


def get_names(text):
    doc = nlp(text)
    names = []
    for x in doc.ents:
        if x.label_ == 'PERSON':
            names.append(x.text)

    names = list(set(names))

    return names


def replace_names(text):
    names = get_names(text)

    if not names:
        return text

    for name in names:
        adv_name = random.choice(MALE_NAMES)
        adv_text = text.replace(name, adv_name)

    return adv_text


