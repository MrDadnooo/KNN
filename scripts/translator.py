

TEST_XML_FILE_NAME='uuid:aae72cf6-1faa-4d81-95d4-24d15fbc41a4.xml'


path_to_xmls = '../res/xmls/'
uuid = TEST_XML_FILE_NAME[:-4]

from tqdm import tqdm
# loading xml file
import zipfile
import os
import xml.etree.ElementTree as ET
from transformers import MarianMTModel, AutoTokenizer

translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en")
translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")


def load_e_tree(filename=TEST_XML_FILE_NAME):
    with open(path_to_xmls + filename, 'r') as xml_file:
        e_tree = ET.parse(xml_file)
        return e_tree
        

# translating xml file
def translate_e_tree(e_tree: ET.ElementTree):

    top_level_schema = e_tree.getroot().tag.split('}')[0] + '}'
    number_of_translatable_elements = len(list(e_tree.iter(top_level_schema + 'Unicode')))
    
    for elem in tqdm(e_tree.iter(top_level_schema + 'Unicode'), desc='Translating page', total=number_of_translatable_elements, unit='  Elements processed'):
        text = elem.text
        tokens = translator_tokenizer([text], return_tensors="pt", padding=True)
        translated_tokens = translator.generate(**tokens, num_beams=4, max_length=40, early_stopping=True)
        translated_text = translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        elem.text = translated_text[0]


    ### saving translated xml
    e_tree.write(path_to_xmls + uuid + '_translated.xml')




translate_e_tree(load_e_tree())