from dataset import Dataset
from tqdm import tqdm
# loading xml file
import torch
import xml.etree.ElementTree as ET
from transformers import MarianMTModel, AutoTokenizer

TEST_XML_FILE_NAME = 'uuid:aae72cf6-1faa-4d81-95d4-24d15fbc41a4.xml'
path_to_xmls = '../res/cache/processing.2023-09-18/zips/page_xml/'
uuid = TEST_XML_FILE_NAME[:-4]


def translate_dataset(dataset : Dataset):
    model, tokenizer = load_cs_model()

    for dp in tqdm(dataset.data_points, desc="Translating czech datapoints"):
        if dp.language != 'de' and dp.page.uuid not in dataset.error_uuids:
            text = [sen.text for sen in dp.text_lines]
            tranlated_text = translate(text, model, tokenizer)
            for i, en_text in enumerate(dp.text_lines):
                en_text.text = tranlated_text[i]
        else:
            continue

    unload_model(model, tokenizer)

    model, tokenizer = load_de_model()

    for dp in tqdm(dataset.data_points, desc="Translating german datapoints"):
        if dp.language == 'de' and dp.page.uuid not in dataset.error_uuids:
            text = [sen.text for sen in dp.text_lines]
            tranlated_text = translate(text, model, tokenizer)
            for i, en_text in enumerate(dp.text_lines):
                en_text.text = tranlated_text[i]
        else:
            continue
    
    unload_model(model, tokenizer)

def load_cs_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")

    if torch.cuda.is_available():
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en", torch_dtype=torch.float16).to('cuda')
    else:
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en", torch_dtype=torch.float32)


    return model, tokenizer

def load_de_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    if torch.cuda.is_available():
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en", torch_dtype=torch.float16).to('cuda')
    else:
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en", torch_dtype=torch.float32)

    return model, tokenizer

def unload_model(model, tokenizer):
    del model
    del tokenizer
    torch.cuda.empty_cache()


def translate(text: str, model, tokenizer) -> str:

    # print(text)

    if text is None:
        return ""

    # iterate over array and if some element is None, replace it with empty string
    if isinstance(text, list):
        text = ["" if x is None else x for x in text]

    if torch.cuda.is_available():
        tokens = tokenizer(text, return_tensors='pt', padding=True).to('cuda')
    else:
        tokens = tokenizer(text, return_tensors='pt', padding=True)

    translated_tokens = model.generate(**tokens, num_beams=4, max_length=40, early_stopping=True)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text
