import open_clip
import dataset
from annotation import ImageAnnotation
from parse_xml import TextLine, TextRegion
from shapely import Point, Polygon
from typing import Any, TypeVar

import torch
import re
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k', device=device,
                                                             precision='fp16')
tokenizer = open_clip.get_tokenizer('ViT-g-14')


def remove_hyphens(in_str):
    return re.sub(r'--\s+', ' ', re.sub(r'(?<=\w)-\s(?=\w)', '', str(in_str)))


def en_text_from_regions(region: TextRegion):
    return remove_hyphens(' '.join([line.get_en_text() for line in region.text_lines if line.get_en_text()]))


def __compute_clip_distances(data_point: dataset.DataPoint, text: list[str], point_entities: list[Any]) \
        -> dict[ImageAnnotation, list[tuple[Any, float]]]:
    tokenized_text = tokenizer(text).to(device)
    result: dict[ImageAnnotation, list[tuple[Any, float]]] = {}

    for image_ann in data_point.img_annotations:
        preprocessed_image = preprocess(image_ann.image_data).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                image_features = model.encode_image(preprocessed_image).to(device)
                text_features = model.encode_text(tokenized_text).to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                result[image_ann] = sorted(list(zip(point_entities, text_probs.tolist()[0])), key=lambda x: x[1],
                                           reverse=True)
            except RuntimeError:
                print(data_point.page.uuid)
                break
    return result


def compute_clip_lines_dst(data_point: dataset.DataPoint) -> dict[ImageAnnotation, list[tuple[TextLine, float]]]:
    text: list[str] = [line.get_en_text() for line in data_point.text_lines]
    return __compute_clip_distances(data_point, text, data_point.text_lines)


def compute_clip_regions_dst(data_point: dataset.DataPoint) -> dict[ImageAnnotation, list[tuple[TextRegion, float]]]:
    text: list[str] = [en_text_from_regions(region) for region in data_point.text_regions]
    return __compute_clip_distances(data_point, text, data_point.text_regions)


def compute_clip_sentences_dst(data_point: dataset.DataPoint) \
        -> dict[ImageAnnotation, list[tuple[dataset.Sentence, float]]]:
    sentences = [sen.sentence for sen in data_point.sentences]
    return __compute_clip_distances(data_point, sentences, data_point.sentences)
