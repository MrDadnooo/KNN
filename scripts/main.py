import json
from typing import Iterator, Dict, Any, IO
import io

import argparse
from urllib.parse import unquote

from PIL import Image

import annotation
import dataset
from parse_xml import parse_xml_document
from visualise import plot_text_regions

from download import dataManager

HOSTNAME = 'merlin.fit.vutbr.cz'
PORT = 22


LOCAL_PATH = '../res/downloads/'
REMOTE_PATH = '/mnt/matylda1/ikiss/pero/experiments/digiknihovny'
IDK_JSON_FILE = '../res/project-9-at-2024-03-05-17-19-577ee11f.json'
CACHE_PATH = '../res/cache'
IMAGE_ZIP_MAPPING = 'image_zip_mapping'
JSON_PATH = '../res/project-9-at-2024-03-05-17-19-577ee11f.json'


def main():
    args = argparse.ArgumentParser(
        description='Download data from remote server')
    args.add_argument('--update', action='store_true')
    args = args.parse_args()

    image_uuids: Iterator[str] = get_image_uuids_from_json()
    image_uuid: str = next(image_uuids)
    xml_file = dataManager.get_xml_file(image_uuid)

    if xml_file:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            annotation_docs = annotation.parse_input_json(in_dict=json_data)
            ocr_data = parse_xml_document(xml_file)

            ann_el = annotation_docs[image_uuid]
            dataset.pair_text_data_and_annotations(ocr_data, ann_el)
            dataset.pair_image_annotations_with_labels(ann_el)

            plot_text_regions(ocr_data, annotation_docs, image_uuid)


def get_image_uuids_from_json(json_path: str = '../res/project-9-at-2024-03-05-17-19-577ee11f.json') -> Iterator[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)
        for el in data:
            uuid = unquote((el['data']['image']).split('/')[-1])[5:-4]
            yield uuid


if __name__ == "__main__":
    main()
