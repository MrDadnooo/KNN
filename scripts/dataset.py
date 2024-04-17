from os import path
from os import mkdir
import pickle
from shapely.geometry import Polygon
import parse_xml as ocr
from annotation import ImageAnnotation, TextAnnotation, AnnotationRecord, ImageLabelData
import annotation
from download import dataManager
import json
from datetime import datetime


class DataPoint:
    def __init__(self, text_annotations: list[TextAnnotation],
                 img_annotations: list[ImageAnnotation],
                 text_lines: list[ocr.TextLine],
                 text_regions: list[ocr.TextRegion],
                 page: ocr.Page
                 ):
        self.text_annotations = text_annotations
        self.img_annotations = img_annotations
        self.text_lines = text_lines
        self.text_regions = text_regions
        self.page = page


def create_from_raw_data(image_uuid: str, ann_rec: AnnotationRecord) -> None | DataPoint:
    xml_file = dataManager.get_xml_file(image_uuid)
    if xml_file:
        pair_image_annotations_with_labels(ann_rec)

        # while parsing xml document, filter out
        ocr_page = ocr.parse_xml_document(xml_file, ann_rec)
        pair_text_data_and_annotations(ocr_page, ann_rec)

        text_anns = []
        for im, text_ann_list in ann_rec.annotations.items():
            for text in text_ann_list:
                text_anns.append(text)
                text.image = im
            im.texts = text_ann_list

        return DataPoint(
            text_anns,
            list(ann_rec.annotations.keys()),
            [line for region in ocr_page.text_regions for line in region.text_lines],
            [region for region in ocr_page.text_regions],
            ocr_page
        )
    return None


class Dataset:
    def __init__(self, data_points: list[DataPoint]):
        self.ann_paths: set[str] = set()
        self.uuids: set[str] = set()
        self.error_uuids: set[str] = set()
        self.data_points = data_points

    def save(self):
        cache_path = dataManager.cache_path

        if not path.isdir(f'{cache_path}/datasets'):
            mkdir(f'{cache_path}/datasets')

        ds_path = path.join(cache_path, 'datasets', f"dataset_{len(self.data_points)}")
        with open(ds_path, 'wb') as ds_file:
            pickle.dump(self, ds_file)

    def __iter__(self):
        return iter(self.data_points)

    def __len__(self):
        return len(self.data_points)

    def update(self, limit: int = 10, json_path: str = None):
        added: int = 0
        if json_path is not None:
            self.ann_paths.add(json_path)

        for json_path in self.ann_paths:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                annotation_records = annotation.parse_input_json(json_data)

                for image_uuid, ann_rec in annotation_records.items():
                    if added >= limit:
                        return
                    if image_uuid in self.uuids:
                        continue
                    try:
                        data_point = create_from_raw_data(image_uuid, ann_rec)
                    except Exception:
                        self.error_uuids.add(image_uuid)
                    if data_point:
                        self.data_points.append(data_point)
                        added += 1
                        print(f"successfully added data_point {image_uuid}")
                        self.uuids.add(image_uuid)
                    else:
                        print(f"Could not fetch a xml ocr data file for uuid: {image_uuid}")


def load_data_set(ds_path: str) -> None | Dataset:
    if path.isfile(ds_path):
        with open(ds_path, 'rb') as ds_file:
            dataset = pickle.load(ds_file)
            return dataset
    else:
        return None


def create_data_set(json_path: str, limit: int = None) -> None | Dataset:
    print(f"Creating a data set from annotation file at: {json_path}")
    created_uuids: set[str] = set()
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        annotation_records = annotation.parse_input_json(json_data)

        print(f"Annotation file successfully parsed. Found {len(annotation_records)} annotation records.")
        print("Starting a data point creation process")
        data_points = []
        for idx, (image_uuid, ann_rec) in enumerate(annotation_records.items()):
            if limit and idx >= limit:
                break
            data_point = create_from_raw_data(image_uuid, ann_rec)
            if data_point:
                print(f'[{idx}] successfully created a data point for uuid: {image_uuid}')
                created_uuids.add(image_uuid)
                data_points.append(data_point)
            else:
                print(f"Could not fetch a xml ocr data file for uuid: {image_uuid}")
        ds = Dataset(data_points)
        ds.ann_paths.add(json_path)
        ds.uuids = created_uuids
        return ds


def pair_text_data_and_annotations(ocr_data: ocr.Page, ann_el: AnnotationRecord) -> None:
    """
    Match text annotation with corresponding text lines there can be an N:1 relation
    between text_lines and text_annotation so the intersection will be used as:
    if intersection of text_line with text_annotation covers at least 50% of the area of the
    text_line's polygon.
    :param ocr_data:
    :param ann_el:
    :return:
    """
    for image, text_anns in ann_el.annotations.items():
        for text_ann in text_anns:
            intersect_text_lines = []
            for text_region in ocr_data:
                for text_line in text_region:
                    text_line_poly = Polygon(text_line.coords)
                    req_area = 0.5 * text_line_poly.area
                    ann_poly = Polygon(text_ann.coords)
                    intersect = text_line_poly.intersection(ann_poly)
                    if intersect.area > req_area:
                        intersect_text_lines.append(text_line)
            text_ann.text_lines = intersect_text_lines


def pair_image_annotations_with_labels(ann_el: AnnotationRecord):
    image_labels_file = dataManager.get_image_labels(ann_el.image_uuid)
    label_idx_mapping = {}
    image_labels = []
    for line in image_labels_file:
        label, *coords = line.decode('utf-8').strip().split(" ")
        if label in label_idx_mapping:
            label_idx_mapping[label] += 1
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
        image_ann.ocr_ref = curr_best[1]
        image_ann.image_data = dataManager.get_image_crops(ann_el.image_uuid, image_ann.ocr_ref)


def update_data_set(json_path: str, ds_path: str, limit: int = None) -> None | Dataset:
    dataset = load_data_set(ds_path)

    print(f"Updating a data set {ds_path} with {len(dataset)} records from annotation file at: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        annotation_records = annotation.parse_input_json(json_data)

        data_points = []
        for idx, (image_uuid, ann_rec) in enumerate(annotation_records.items()):
            if image_uuid in [dp.page.uuid for dp in dataset]:
                limit += 1
                continue
            if limit and idx >= limit:
                break
            data_point = create_from_raw_data(image_uuid, ann_rec)
            if data_point:
                print(f'[{idx}] successfully created a data point for uuid: {image_uuid}')
                data_points.append(data_point)
            else:
                print(f"Could not fetch a xml ocr data file for uuid: {image_uuid}")
        for dp in dataset:
            data_points.append(dp)
        return Dataset(data_points)

