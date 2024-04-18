import torch
import open_clip
import dataset
from download import dataManager
from annotation import TextAnnotation, ImageAnnotation
from parse_xml import TextLine

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sen_split import create_split_map

import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k', device=device, precision='fp16')
tokenizer = open_clip.get_tokenizer('ViT-g-14')


def compute_clip_probs(data_point: dataset.DataPoint) -> dict[ImageAnnotation, list[tuple[TextLine, float]]]:
    en_text = [line.en_text for line in data_point.text_lines]
    cz_text = [line.text for line in data_point.text_lines]
    tokenized_text = tokenizer(en_text).to(device)
    result: dict[ImageAnnotation, list[tuple[TextLine, float]]] = {}

    for image_ann in data_point.img_annotations:
        preprocessed_image = preprocess(image_ann.image_data).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                image_features = model.encode_image(preprocessed_image).to(device)
                text_features = model.encode_text(tokenized_text).to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                result[image_ann] = sorted(list(zip(data_point.text_lines, text_probs.tolist()[0])), key=lambda x: x[1], reverse=True)
            except RuntimeError:
                print(data_point.page.uuid)
                break
    return result

def compute_clip_probs_sentence(data_point: dataset.DataPoint) -> dict[ImageAnnotation, list[tuple[TextLine, float]]]:
    sentences = [sen.en_text for sen in data_point.sentences]

    tokenized_text = tokenizer(sentences).to(device)
    result: dict[ImageAnnotation, list[tuple[TextLine, float]]] = {}

    for image_ann in data_point.img_annotations:
        preprocessed_image = preprocess(image_ann.image_data).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                image_features = model.encode_image(preprocessed_image).to(device)
                text_features = model.encode_text(tokenized_text).to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                result[image_ann] = sorted(list(zip(data_point.text_lines, text_probs.tolist()[0])), key=lambda x: x[1], reverse=True)
            except RuntimeError:
                print(data_point.page.uuid)
                break
    return result

def eval_clip_result(
        dp: dataset.DataPoint,
        result_dict: dict[ImageAnnotation, list[tuple[TextLine, float]]]
) -> dict[ImageAnnotation, float]:
    result: dict[ImageAnnotation, float] = {}
    for img_ann, results in result_dict.items():
        text_anns = [t_ann for t_ann in dp.text_annotations if t_ann.image == img_ann]
        text_ann_lines = set([text_line for t_ann in text_anns for text_line in t_ann.text_lines])
        text_line_count: int = len(text_ann_lines)
        score: float = sum([i if text_line in text_ann_lines and i >= text_line_count else 0 for i, (text_line, _) in enumerate(results)])
        norm: float = (text_line_count/2) * (len(results)*2 - text_line_count + 1)
        if norm == 0.0:
            print(f"Missing text annotations, possibly filtered due to overlapping with image ...")
            print("setting `score=0.0`")
            continue
        norm_score: float = score / norm
        result[img_ann] = 1.0 - norm_score
    return result
