import numpy as np
from urllib.parse import unquote


def box(value: dict[str, object], org_w: int, org_h: int) -> tuple[tuple[float, float, float, float], np.array]:
    x = float(value['x']) / 100 * org_w
    y = float(value['y']) / 100 * org_h
    w = float(value['width']) / 100 * org_w
    h = float(value['height']) / 100 * org_h
    return (x, y, w, h), np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


class Annotation:
    def __init__(self, ann_id: int, coords: dict[str, object], annotation_type: str, org_w: str, org_h: str):
        self.id = ann_id
        (self.x, self.y, self.w, self.h), self.coords = box(coords, int(org_w), int(org_h))
        self.is_text: bool = annotation_type != "obrázek"
        self.text_line = None


class Document:
    def __init__(self, image_uuid: str, annotations: dict[Annotation, list[Annotation]]):
        self.annotations = annotations
        self.image_uuid = image_uuid


def parse_input_json(in_dict) -> dict[str, Document]:
    documents = {}
    for el in in_dict:
        annotations = {}
        relations = {}
        # first pass, map texts and images
        for annotations_el in el['annotations']:
            for ann_data in annotations_el['result']:

                if ann_data['type'] == 'relation':
                    if relations.get(ann_data['to_id']):
                        relations[ann_data['to_id']].append(ann_data['from_id'])
                    else:
                        relations[ann_data['to_id']] = [ann_data['from_id']]
                else:
                    annotations[ann_data['id']] = \
                        Annotation(ann_data['id'], ann_data['value'], ann_data['value']['rectanglelabels'][0], \
                                   ann_data['original_width'], ann_data['original_height'])

        image_uuid = unquote((el['data']['image']).split('/')[-1])[5:-4]

        # second pass, map relations into a document
        # probably only one image has been specified
        if not relations:
            im_ann = None
            text_anns = []
            for _, ann in annotations.items():
                if not ann.is_text:
                    im_ann = ann
                else:
                    text_anns.append(ann)
            if im_ann:
                documents[image_uuid] = Document(image_uuid, {im_ann: text_anns})
        else:
            for ann_id, ann in annotations.items():
                if text_ann_ids := relations.get(ann_id):
                    documents[image_uuid] = \
                        Document(image_uuid, {ann: [annotations[text_ann_id] for text_ann_id in text_ann_ids]})
                else:
                    continue
    return documents

