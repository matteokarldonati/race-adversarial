import random
from difflib import SequenceMatcher

import nltk
import spacy
from nltk.corpus import names

nlp = spacy.load("en_core_web_sm")

nltk.download('names')

MALE_NAMES = names.words('male.txt')
FEMALE_NAMES = names.words('female.txt')


def group_names(names):
    result = []
    for sentence in names:
        if len(result) == 0:
            result.append([sentence])
        else:
            for i in range(0, len(result)):
                score = SequenceMatcher(None, sentence, result[i][0]).ratio()
                if score < 0.5:
                    if i == (len(result) - 1):
                        result.append([sentence])
                else:
                    if score != 1:
                        result[i].append(sentence)
    return result


def get_names(text):
    doc = nlp(text)
    names = []
    for x in doc.ents:
        if x.label_ == 'PERSON':
            names.append(x.text)

    names = list(set(names))

    return names


def replace_names(text, female_names):
    names = group_names(get_names(text))

    if not names:
        return text

    for name in names:
        if female_names:
            adv_name = random.choice(FEMALE_NAMES)
        else:
            adv_name = random.choice(MALE_NAMES)

        for i in name:
            adv_text = text.replace(i, adv_name)
            text = adv_text

    return adv_text
