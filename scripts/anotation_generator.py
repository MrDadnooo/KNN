import ollama
from ollama import generate
from PIL import Image
import os
from io import BytesIO
import dataset as ds
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List


class AnnotationGenerator:
    def __init__(self, ds_path: str):
        self.dataset = ds.load_data_set(ds_path)


    def process_image(self, image: Image):
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        full_response = ''
        for response in generate(model='llava',
                                 prompt='Create a annotation for this image based on what is present in the image.'
                                        'The annotation should be max 1 sentence long.',
                                 images=[image_bytes],
                                 stream=True):

            full_response += (response['response'])
        return full_response

    def process_dataset(self):
        result = []
        for data_point in tqdm(self.dataset):
            for image_ann in data_point.img_annotations:
                result.append(self.process_image(image_ann.image_data))


class SentenceEncoder:
    def __init__(self, ds_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        self.annotator = AnnotationGenerator(ds_path)

    def encode_data_point(self, data_point: ds.DataPoint):
        text = [line.en_text for line in data_point.text_lines]
        result = {}
        # encoding the text and generated annotations
        text_features = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False).to(self.device)
        for image_ann in data_point.img_annotations:
            generated_text = self.annotator.process_image(image_ann.image_data)
            # print(generated_text)
            image_features = self.model.encode(generated_text, convert_to_tensor=True, show_progress_bar=False).to(self.device)
            cos_scores = util.pytorch_cos_sim(image_features, text_features)
            cos_scores = cos_scores.cpu()

            tmp = (sorted(list(zip(text, cos_scores[0])), key=lambda x: x[1], reverse=True))
            tmp = [(x[0], x[1].item()) for x in tmp]
            result[image_ann] = tmp
        return result

    def encode_dataset(self):
        result = {}
        for data_point in tqdm(self.annotator.dataset):
            new_data = self.encode_data_point(data_point)
            result.update(new_data)
        return result


# encoder = SentenceEncoder('../res/cache/datasets/dataset_867')
# encoder.encode_dataset()


# TODO: before running, you need to install ollama
# then in terminal run the following command:
# ollama run llava


