import argparse

import os
import dataset
import visualise
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

    dataset_path = '../res/cache/datasets/dataset_10_2024-03-23_22-00-35'
    if os.path.isfile(dataset_path):
        print('Loading a dataset ...')
        data_set = dataset.load_data_set(dataset_path)
    else:
        data_set = dataset.create_data_set(dataManager.annotation_path, limit=10)
        data_set.save()
    for dp in data_set:
        visualise.plot_text_regions(dp)


if __name__ == "__main__":
    main()
