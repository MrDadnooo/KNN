import torch
import open_clip
import dataset
from download import dataManager
from annotation import TextAnnotation, ImageAnnotation
from parse_xml import TextLine


model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-g-14')


def compute_clip_probs(data_point: dataset.DataPoint) -> dict[ImageAnnotation, list[tuple[TextLine, float]]]:
    en_text = [line.en_text for line in data_point.text_lines]
    cz_text = [line.text for line in data_point.text_lines]
    tokenized_text = tokenizer(en_text)

    result: dict[ImageAnnotation, list[tuple[TextLine, float]]] = {}

    for image_ann in data_point.img_annotations:
        image = dataManager.get_image_crops(data_point.page.uuid, image_ann.ocr_ref)
        preprocessed_image = preprocess(image).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(preprocessed_image)
            text_features = model.encode_text(tokenized_text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            result[image_ann] = sorted(list(zip(data_point.text_lines, text_probs.tolist()[0])), key=lambda x: x[1], reverse=True)
    return result


def eval_clip_result(
        dp: dataset.DataPoint,
        result_dict: dict[ImageAnnotation, list[tuple[TextLine, float]]]
) -> dict[ImageAnnotation, float]:
    result: dict[ImageAnnotation, float] = {}
    for img_ann, results in result_dict.items():
        text_anns = [t_ann for t_ann in dp.text_annotations if t_ann.image == img_ann]
        text_ann_lines = set([text_line for t_ann in text_anns for text_line in t_ann.text_lines])
        score: float = 0.0
        text_line_count: int = len(text_ann_lines)
        for expected_idx in range(text_line_count):
            curr_skip: int = 0
            for line_idx, (text_line, _) in enumerate(results[expected_idx:]):
                if text_line in text_ann_lines:
                    if curr_skip >= expected_idx:
                        score += float(line_idx - expected_idx)
                    else:
                        curr_skip += 1
        norm: float = float((len(results) - text_line_count) * text_line_count)
        norm_score: float = score / norm
        result[img_ann] = 1.0 - norm_score
    return result
