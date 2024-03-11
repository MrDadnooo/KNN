import numpy as np
from urllib.parse import unquote

def box(value: dict[str, object]) -> np.array:
    x = float(value['x'])
    y = float(value['y'])
    w = float(value['width'])
    h = float(value['height'])
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


class Annotation:
    def __init__(self, ann_id: int, coords: dict[str, object], annotation_type: str):
        self.id = ann_id
        self.coords = box(coords)
        self.is_text: bool = annotation_type != "obrázek"


class Document:
    def __init__(self, image_uuid: str, annotations: dict[Annotation, list[Annotation]]):
        self.annotations = annotations
        self.image_uuid = image_uuid


def parse_input_json(in_dict) -> dict[str, Document]:
    result = {}
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
                        Annotation(ann_data['id'], ann_data['value'], ann_data['value']['rectanglelabels'][0])
        # second pass, map relations into a document
        image_uuid = unquote((el['data']['image']).split('/')[-1])[5:-4]

        # probably only one image has been specified
        documents = []
        if not relations:
            im_ann = None
            text_anns = []
            for _, ann in annotations.items():
                if not ann.is_text:
                    im_ann = ann
                else:
                    text_anns.append(ann)
            if im_ann:
                documents.append(Document(image_uuid, {im_ann: text_anns}))
        else:
            for ann_id, ann in annotations.items():
                if relations.get(ann_id):
                    ...
                else:
                    continue

