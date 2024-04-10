import re
from typing import List

import nltk

import dataset
from translator import translate



class Sentence:
    def __init__(self, sentence, positions) -> None:
        self.sentence = sentence
        self.positions = positions
        self.en_text = translate(sentence)

def remove_hyphens(in_str):
    return re.sub(r'(?<=\w)-\s(?=\w)', '', str(in_str))

def full_cz_from_regions(dp):
    return [remove_hyphens(' '.join([line.text for line in reg.text_lines if line.text])) for reg in dp.text_regions]

def create_split_map(Datapoint : dataset.DataPoint) -> List[Sentence]:

    sentences = []
    for text in full_cz_from_regions(Datapoint):
        sentences.extend(nltk.sent_tokenize(text, language='czech'))

    text_array = [(line.text.split(), line) for line in Datapoint.text_lines if line.text is not None]
    new_text_array = [(word, line) for words, line in text_array for word in words]

    map = []

    for sentence in sentences:
        rows = []
        words = sentence.split()
        for word in words:
            if word == new_text_array[0][0]:
                rows.append(new_text_array[0][1])
                new_text_array.remove(new_text_array[0])
            elif new_text_array[0][0][-1] == '-':
                rows.append(new_text_array[0][1])
                rows.append(new_text_array[1][1])
                new_text_array.remove(new_text_array[0])
                new_text_array.remove(new_text_array[0])
        rows = set(rows)
        map.append(Sentence(sentence, rows))

    return map
