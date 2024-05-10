import argparse

import os
import dataset
from download import dataManager


def main():
    args = argparse.ArgumentParser(
        description='Download data from remote server')
    args.add_argument('--update', action='store_true')
    args = args.parse_args()

    files = os.listdir('../res/cache/datasets/')

    # dataset_files = [file for file in files if file.startswith('dataset')]

    # if dataset_files:
    #     indices = [int(file.split('_')[-1]) for file in dataset_files]

    #     # dataset_path = os.path.join('../res/cache/datasets/', f'dataset_{max(indices)}')
    #     dataset_path = os.path.join('../res/cache/datasets/dataset_522')

    #     if args.update:
    #         print('Loading a dataset ...')
    #         update_dataset = dataset.load_data_set(dataset_path)
    #         print('Updating a dataset ...')     
    #         update_dataset.update(500, dataManager.annotation_path)
    #         update_dataset.save()

    #     else:
    #         print('Loading a dataset ...')
    #         data_set = dataset.load_data_set(dataset_path)
    # else:
    data_set = dataset.create_data_set(dataManager.annotation_path, limit=60)
    data_set.save()

    # for dp in data_set:
    #     visualise.plot_text_regions(dp)


if __name__ == "__main__":
    main()
