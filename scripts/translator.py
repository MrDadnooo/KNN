

TEST_XML_FILE_NAME = 'uuid:aae72cf6-1faa-4d81-95d4-24d15fbc41a4.xml'


path_to_xmls = '../res/cache/processing.2023-09-18/zips/page_xml/'
uuid = TEST_XML_FILE_NAME[:-4]

from tqdm import tqdm
# loading xml file
import torch
import xml.etree.ElementTree as ET
from transformers import MarianMTModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")

if torch.cuda.is_available():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en", torch_dtype=torch.float16).to("cuda")
else:
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en", torch_dtype=torch.float32)


def translate(text: str) -> str:
    if torch.cuda.is_available():
        tokens = tokenizer([text], return_tensors='pt', padding=True).to('cuda')
    else:
        if text is None:
            return ""
        tokens = tokenizer([text], return_tensors='pt', padding=True)
    translated_tokens = model.generate(**tokens, num_beams=4, max_length=40, early_stopping=True)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]


def load_e_tree(filename=TEST_XML_FILE_NAME):
    with open(path_to_xmls + filename, 'r') as xml_file:
        e_tree = ET.parse(xml_file)
        return e_tree


# translating xml file
def translate_e_tree(e_tree: ET.ElementTree):

    top_level_schema = e_tree.getroot().tag.split('}')[0] + '}'
    number_of_translatable_elements = len(list(e_tree.iter(top_level_schema + 'Unicode')))
    
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        for elem in tqdm(e_tree.iter(top_level_schema + 'Unicode'), desc='Translating page', total=number_of_translatable_elements, unit='  Elements processed'):
            text = elem.text
            tokens = tokenizer([text], return_tensors="pt", padding=True).to('cuda')
            translated_tokens = model.generate(**tokens, num_beams=4, max_length=40, early_stopping=True)
            translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            elem.text = translated_text[0]
    ### saving translated xml
    e_tree.write(path_to_xmls + uuid + '_translated.xml')
