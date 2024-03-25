import json
from typing import Iterator

import argparse
from urllib.parse import unquote

import os
import dataset
import visualise
from download import dataManager


def main():
    args = argparse.ArgumentParser(
        description='Download data from remote server')
    args.add_argument('--update', action='store_true')
    args = args.parse_args()

    dataset_path = '../res/cache/datasets/dataset_10_2024-03-25_09-38-36'
    if os.path.isfile(dataset_path):
        print('Loading a dataset ...')
        data_set = dataset.load_data_set(dataset_path)
    else:
        data_set = dataset.create_data_set(dataManager.annotation_path, limit=1)
        data_set.save()
    for dp in data_set:
        visualise.plot_text_regions(dp)


if __name__ == "__main__":
    main()
