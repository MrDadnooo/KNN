import dataset
import numpy as np
from annotation import ImageAnnotation
from parse_xml import TextLine, TextRegion
from shapely import Polygon


def eval_sorted_score(
        dp: dataset.DataPoint,
        result_dict: dict[ImageAnnotation, list[tuple[TextLine, float]]]
) -> dict[ImageAnnotation, float]:
    result: dict[ImageAnnotation, float] = {}
    for img_ann, results in result_dict.items():
        text_anns = [t_ann for t_ann in dp.text_annotations if t_ann.image == img_ann]
        text_ann_lines = set([text_line for t_ann in text_anns for text_line in t_ann.text_lines])
        text_line_count: int = len(text_ann_lines)
        score: float = sum([i if text_line in text_ann_lines and i >= text_line_count else 0 for i, (text_line, _) in
                            enumerate(results)])
        norm: float = (text_line_count / 2) * (len(results) * 2 - text_line_count + 1)
        if norm == 0.0:
            print(f"Missing text annotations, possibly filtered due to overlapping with image ...")
            print("setting `score=0.0`")
            continue
        norm_score: float = score / norm
        result[img_ann] = 1.0 - norm_score
    return result


def adjust_region_score(
        score_dict: dict[ImageAnnotation, list[tuple[TextRegion, float]]],
        len_max: int = 10,
        len_penalty: float = 0.4,
        dist_far_threshold: float = 0.5,
        dist_far_penalty: float = 0.00001,
        dist_near_threshold: float = 0.02,
        dist_near_penalty: float = 10000
) -> dict[ImageAnnotation, list[tuple[TextRegion, float]]]:
    result: dict[ImageAnnotation, list[tuple[TextRegion, float]]] = {}
    for img_ann, results in score_dict.items():
        img_poly = Polygon(img_ann.coords)
        dists = [img_poly.distance(Polygon(reg.coords)) for reg, _ in results]
        max_dist = max(dists)
        norm_dists = [dist / max_dist for dist in dists]

        # compute penalties
        new_reg_list = []
        for idx, ((reg, score), dist) in enumerate(zip(results, norm_dists)):
            new_score = score
            if len(reg.text_lines) > len_max:
                new_score *= len_penalty
            if dist > dist_far_threshold:
                new_score *= dist_far_penalty
            if dist < dist_near_threshold:
                new_score *= dist_near_penalty
            new_reg_list.append((reg, new_score))

        # re-normalize
        sum_score = sum([score for _, score in new_reg_list])
        for i, (reg, score) in enumerate(new_reg_list):
            new_reg_list[i] = (reg, score / sum_score)
        result[img_ann] = sorted(new_reg_list, key=lambda x: x[1], reverse=True)
    return result


def eval_threshold(
        dp: dataset.DataPoint,
        result_dict: dict[ImageAnnotation, list[tuple[object, float]]],
        threshold: float = 0.8) -> np.array:
    for img_ann, results in result_dict.items():
        res_sum = 0
        # calculate index of the positive samples
        threshold_idx = None
        for threshold_idx, res in enumerate(results):
            if res_sum > threshold:
                break
            res_sum += res[1]
        text_anns = [t_ann for t_ann in dp.text_annotations if t_ann.image == img_ann]
        text_ann_lines = set([text_line for t_ann in text_anns for text_line in t_ann.text_lines])

        result = np.array([0, 0, 0, 0])
        # positives
        for res in results[:threshold_idx]:
            if res[0] in text_ann_lines:
                result[0] += 1
            else:
                result[1] += 1

        # negatives
        for res in results[threshold_idx:]:
            if res[0] in text_ann_lines:
                result[3] += 1
            else:
                result[2] += 1
        return result


def eval_regions_threshold(dp: dataset.DataPoint,
                           result_dict: dict[ImageAnnotation, list[tuple[TextRegion, float]]],
                           threshold: float = 0.8):
    for img_ann, results in result_dict.items():
        res_sum = 0
        # calculate index of the positive samples
        threshold_idx = None
        for threshold_idx, res in enumerate(results):
            if res_sum > threshold:
                break
            res_sum += res[1]
        text_anns = [t_ann for t_ann in dp.text_annotations if t_ann.image == img_ann]
        text_ann_lines = set([text_line for t_ann in text_anns for text_line in t_ann.text_lines])

        # mark what regions are annotations
        for result, _ in results:
            result.is_annotation = all(line in text_ann_lines for line in result.text_lines)

        result = np.array([0, 0, 0, 0])
        # positives
        for res in results[:threshold_idx]:
            if res[0].is_annotation:
                result[0] += 1
            else:
                result[1] += 1

        # negatives
        for res in results[threshold_idx:]:
            if res[0].is_annotation:
                result[3] += 1
            else:
                result[2] += 1
        return result


def eval_region_data_set(
        data_set: dataset.Dataset,
        model_results: list[dict[ImageAnnotation, list[tuple[object, float]]]],
        threshold: float = 0.8) -> np.array:
    result = np.array([0, 0, 0, 0])
    invalid = 0
    for dp, result_dict in zip(data_set, model_results):
        temp = eval_regions_threshold(dp, result_dict, threshold)
        if temp[3] > 5:
            invalid += 1
            continue
        result += temp

    return result

def eval_sentences_data_set(
        data_set: dataset.Dataset,
        model_results: list[dict[ImageAnnotation, list[tuple[object, float]]]],
        threshold: float = 0.8) -> np.array:
    result = np.array([0, 0, 0, 0])
    for dp, result_dict in zip(data_set, model_results):
        temp = eval_sentences_threshold(dp, result_dict, threshold)
        result += temp

    return result


def eval_sentences_threshold(
        data_point: dataset.DataPoint,
        result_dict: dict[ImageAnnotation, list[tuple[dataset.Sentence, float]]],
        threshold: float = 0.8) -> np.array:
     for img_ann, results in result_dict.items():
        res_sum = 0
        # calculate index of the positive samples
        threshold_idx = None
        for threshold_idx, res in enumerate(results):
            if res_sum > threshold:
                break
            res_sum += res[1]
        text_anns = [t_ann for t_ann in data_point.text_annotations if t_ann.image == img_ann]
        text_ann_lines = set([text_line for t_ann in text_anns for text_line in t_ann.text_lines])

        # mark what regions are annotations
        for sentence, _ in results:
            sentence.is_annotation = any(line in text_ann_lines for line in sentence.positions)

        result = np.array([0, 0, 0, 0])
        # positives
        for res in results[:threshold_idx]:
            if res[0].is_annotation:
                result[0] += 1
            else:
                result[1] += 1

        # negatives
        for res in results[threshold_idx:]:
            if res[0].is_annotation:
                result[3] += 1
            else:
                result[2] += 1
        return result


def eval_threshold_data_set(
        data_set: dataset.Dataset,
        model_results: list[dict[ImageAnnotation, list[tuple[object, float]]]],
        threshold: float = 0.8) -> np.array:
    result = np.array([0, 0, 0, 0])
    for dp, result_dict in zip(data_set, model_results):
        result += eval_threshold(dp, result_dict, threshold)
    return result


def calculate_metrics(tp, fp, _, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score
