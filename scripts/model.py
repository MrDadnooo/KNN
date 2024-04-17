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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k', device=device, precision='fp16')
tokenizer = open_clip.get_tokenizer('ViT-g-14')


def compute_clip_probs(data_point: dataset.DataPoint) -> dict[ImageAnnotation, list[tuple[TextLine, float]]]:
    en_text = [line.en_text for line in data_point.text_lines]
    cz_text = [line.text for line in data_point.text_lines]
    tokenized_text = tokenizer(en_text).to(device)
    result: dict[ImageAnnotation, list[tuple[TextLine, float]]] = {}

    for image_ann in data_point.img_annotations:
        image = dataManager.get_image_crops(data_point.page.uuid, image_ann.ocr_ref)
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(preprocessed_image).to(device)
            text_features = model.encode_text(tokenized_text).to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            result[image_ann] = sorted(list(zip(data_point.text_lines, text_probs.tolist()[0])), key=lambda x: x[1], reverse=True)
    return result


def collate_fn(batch):
    return batch


def compute_clip_probs_batch(dataset, device, model, tokenizer, preprocess, batch_size=16):
    result = []
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_images = {}

        for dp in batch:
            for image_ann in dp.img_annotations:
                image_key = (dp.page.uuid, image_ann.ocr_ref)
                if image_key not in batch_images:
                    image = dataManager.get_image_crops(dp.page.uuid, image_ann.ocr_ref)
                    preprocessed_image = preprocess(image).to(device).unsqueeze(0)
                    batch_images[image_key] = preprocessed_image

        en_texts = [[line.en_text for line in dp.text_lines] for dp in batch]
        tokenized_texts = [tokenizer(en_text) for en_text in en_texts]

        batch_result = []
        for dp, tokenized_text in zip(batch, tokenized_texts):
            try:
                result_dict = {}
                for image_ann in dp.img_annotations:
                    image_key = (dp.page.uuid, image_ann.ocr_ref)
                    preprocessed_image = batch_images[image_key]
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_features = model.encode_image(preprocessed_image)
                        text_features = model.encode_text(tokenized_text.to(device))
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                        result_dict[image_ann] = sorted(list(zip(dp.text_lines, text_probs.tolist()[0])), key=lambda x: x[1], reverse=True)
                batch_result.append(result_dict)
            except RuntimeError as e:
                batch_result.append({dp.page.uuid: e})

        result.extend(batch_result)
        torch.cuda.empty_cache()

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
        norm_score: float = score / norm
        result[img_ann] = 1.0 - norm_score
    return result
