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

    files = os.listdir('../res/cache/datasets/')

    dataset_files = [file for file in files if file.startswith('dataset')]

    if dataset_files:
        indices = [int(file.split('_')[-1]) for file in dataset_files]

        dataset_path = os.path.join('../res/cache/datasets/', f'dataset_{max(indices)}')

        if args.update:
            data_set = dataset.update_data_set(dataManager.annotation_path, dataset_path, limit=10)
            data_set.save()

        else:
            print('Loading a dataset ...')
            data_set = dataset.load_data_set(dataset_path)
    else:
        data_set = dataset.create_data_set(dataManager.annotation_path, limit=1)
        data_set.save()

    for dp in data_set:
        visualise.plot_text_regions(dp)


if __name__ == "__main__":
    main()
