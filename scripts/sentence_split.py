import nltk
import dataset
import re

ds = dataset.load_data_set('../res/cache/datasets/dataset_10_2024-03-27_09-58-19')


def full_en_from_regions(dp):
    return [remove_hyphens(' '.join([line.en_text for line in reg.text_lines if line.en_text])) for reg in dp.text_regions]


def full_cz_from_regions(dp):
    return [remove_hyphens(' '.join([line.text for line in reg.text_lines if line.text])) for reg in dp.text_regions]


def remove_hyphens(in_str):
    return re.sub(r'(?<=\w)-\s(?=\w)', '', str(in_str))


# print('\n\nregion:\n'.join(full_cz_from_regions(ds[4])))

# print('\n'.join(nltk.sent_tokenize(test_sentence, language='english')))

dp = ds[6]

print('\n\nregion(split sentences):\n'.join(['\n'.join(nltk.sent_tokenize(text, language='czech')) for text in full_cz_from_regions(dp)]))
print('\n\n\n')
print('\n\nregion(split sentences):\n'.join(['\n'.join(nltk.sent_tokenize(text, language='english')) for text in full_en_from_regions(dp)]))