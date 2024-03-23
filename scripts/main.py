import json
from typing import Iterator

import argparse
from urllib.parse import unquote

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

    data_set = dataset.create_data_set(dataManager.annotation_path, 10)
    data_set.save()

if __name__ == "__main__":
    main()
