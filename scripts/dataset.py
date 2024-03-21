from shapely.geometry import Polygon
import parse_xml as ocr
import annotation
import numpy as np

from download import dataManager
from parse_xml import parse_points

class Dataset:
    def __init__(self):
        ...


def pair_text_data_and_annotations(ocr_data: ocr.Page, ann_el: annotation.Document) -> None:
    for image, text_anns in ann_el.annotations.items():
        for text_ann in text_anns:
            curr_best = (0.0, None)
            for text_region in ocr_data:
                for text_line in text_region:
                    ocr_poly = Polygon(text_line.coords)
                    ann_poly = Polygon(text_ann.coords)
                    intersect = ocr_poly.intersection(ann_poly)
                    if intersect.area > curr_best[0]:
                        curr_best = (intersect.area, text_line)
            text_ann.text_line = curr_best[1]


class ImageLabelData:
    def __init__(self, label: str, idx: int, coords: list[int]):
        self.label = label
        self.idx = idx
        x1, y1, x2, y2 = coords
        self.coords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def pair_image_annotations_with_labels(ann_el: annotation.Document):
    image_labels_file = dataManager.get_image_labels(ann_el.image_uuid)
    label_idx_mapping = {}
    image_labels = []
    for line in image_labels_file:
        label, *coords = line.decode('utf-8').strip().split(" ")
        if label in label_idx_mapping:
            label_idx_mapping += 1
        else:
            label_idx_mapping[label] = 0
        image_labels.append(ImageLabelData(label, label_idx_mapping[label], list(map(int, coords))))
    for image_ann, text_ann in ann_el.annotations.items():
        curr_best = (0.0, None)
        for image_label in image_labels:
            ann_poly = Polygon(image_ann.coords)
            img_label_poly = Polygon(image_label.coords)
            intersect = ann_poly.intersection(img_label_poly)
            if intersect.area > curr_best[0]:
                curr_best = (intersect.area, image_label)
        image_ann.image_ocr = curr_best[1]