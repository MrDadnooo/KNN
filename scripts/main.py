import json
from typing import Iterator, Dict, Any

from multiprocessing import Process, Manager, Lock
import argparse
import paramiko
import zipfile
import credentials as creds
import io
import pickle
import os
from tqdm import tqdm
from os import path
from urllib.parse import unquote

HOSTNAME = 'merlin.fit.vutbr.cz'
PORT = 22

LOCAL_PATH = '../res/downloads/'
REMOTE_PATH = '/mnt/matylda1/ikiss/pero/experiments/digiknihovny'
IDK_JSON_FILE = '../res/project-9-at-2024-03-05-17-19-577ee11f.json'
CACHE_PATH = '../res/cache'
IMAGE_ZIP_MAPPING = 'image_zip_mapping'


def processing_folder_generator(sftp: paramiko.SFTPClient, remote_path: str) -> Iterator[str]:
    for file_attrs in sftp.listdir_iter(remote_path):
        if file_attrs.filename.startswith("processing"):
            yield file_attrs.filename


def create_uuid_zip_map(todo_list: list[str], folder_name: str, exists: bool) -> None:
    """Processing folders with info about images and their zip files"""
    print(f"Processing folder {folder_name}")

    ssh_client: paramiko.SSHClient = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(HOSTNAME, PORT, creds.USERNAME, creds.PASSWORD)

    if len(todo_list) == 0:
        ssh_client.close()
        return

    uuid_zip_dict: dict[str, str] = {}

    for image_uuid in todo_list[:]:
        grep_command_1: str = f"grep {image_uuid} {REMOTE_PATH}/{folder_name}/splits/*"

        stdin, stdout, stderr = ssh_client.exec_command(grep_command_1)
        output = stdout.read().decode()
        if output:
            grep_command_2: str = f"grep -n {REMOTE_PATH}/{folder_name}/splits/{output.split(':')[0][-9:]} {REMOTE_PATH}/{folder_name}/part_files.txt"

            stdin, stdout, stderr = ssh_client.exec_command(grep_command_2)
            output: str = stdout.read().decode()
            if output:
                print(f'uuid: {image_uuid} found in {folder_name}')
                uuid_zip_dict[image_uuid] = int(output.split(':')[0])

    ssh_client.close()

    if exists:
        with open(path.join(CACHE_PATH, folder_name, IMAGE_ZIP_MAPPING), 'rb') as f:
            old_uuid_zip_dict: dict[str, str] = pickle.load(f)
            old_uuid_zip_dict.update(uuid_zip_dict)
            with open(path.join(CACHE_PATH, folder_name, IMAGE_ZIP_MAPPING), 'wb') as f:
                pickle.dump(old_uuid_zip_dict, f)
    else:
        with open(path.join(CACHE_PATH, folder_name, IMAGE_ZIP_MAPPING), 'wb') as f:
            pickle.dump(uuid_zip_dict, f)


def create_uuid_zip_map_worker(folder_names: list[str]):
    """Create a process for each folder to get the relevant data"""

    todo_list: list[str] = [uuid for uuid in get_image_uuid_from_json()]
    exists: bool = True

    for folder_name in folder_names:
        if not path.isdir(path.join(CACHE_PATH, folder_name)):
            os.mkdir(path.join(CACHE_PATH, folder_name))
            exists = False

            # if pkl file exists, check ids which are in file and remove them from todo list
        if path.isfile(path.join(CACHE_PATH, folder_name, IMAGE_ZIP_MAPPING)):
            exists = True
            with open(path.join(CACHE_PATH, folder_name, IMAGE_ZIP_MAPPING), 'rb') as f:
                uuid_zip_dict: dict[str, str] = pickle.load(f)
                for uuid in uuid_zip_dict:
                    if uuid in todo_list:
                        todo_list.remove(uuid)

    with Manager() as manager:

        print(f'Number of images to process: {len(todo_list)}')
        processes = []
        for folder_name in folder_names:
            # check if folder name exists in cache
            p = Process(target=create_uuid_zip_map, args=(
                todo_list, folder_name, exists))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    print('All processes finished')


def main():
    args = argparse.ArgumentParser(
        description='Download data from remote server')
    args.add_argument('--update', action='store_true')
    args = args.parse_args()

    transport: paramiko.Transport = paramiko.Transport((HOSTNAME, PORT))
    transport.connect(None, creds.USERNAME, creds.PASSWORD)

    sftp: paramiko.SFTPClient = paramiko.SFTPClient.from_transport(transport)

    # Download file from remote server
    # sftp.get(REMOTE_ZIPS_PATH + LABELS + '1.zip', LOCAL_FILE_PATH + LABELS + '1.zip')
    if not os.path.exists(os.path.join(LOCAL_PATH, 'splits')):
        os.mkdir(os.path.join(LOCAL_PATH, 'splits'))

    # only used when trying to download new data
    p_dirs = [processing_name for processing_name in processing_folder_generator(
        sftp, REMOTE_PATH)]

    if args.update:
        create_uuid_zip_map_worker(p_dirs)

    # uuid_mappings = create_image_to_zip_mapping_local(p_dirs)

    # download_relevant_zips(sftp, p_dirs, uuid_mappings)

    if sftp:
        sftp.close()
    if transport:
        transport.close()


def download_relevant_zips(
        sftp: paramiko.SFTPClient,
        p_dirs: list[str],
        image_zip_mappings: dict[str, dict[str, str]],
        json_path: str = '../res/project-9-at-2024-03-05-17-19-577ee11f.json'):

    for p_dir in p_dirs:
        zip_path = path.join(CACHE_PATH, p_dir, 'zips')
        os.makedirs(path.join(zip_path, 'page_xml'), exist_ok=True)
        os.makedirs(path.join(zip_path, 'labels'), exist_ok=True)
        os.makedirs(path.join(zip_path, 'crops'), exist_ok=True)

    for image_uuid in get_image_uuid_from_json(json_path):
        for p_dir in p_dirs:
            if zip_idx := image_zip_mappings.get(p_dir).get(image_uuid):
                zip_path = f"{REMOTE_PATH}/{p_dir}/zips/page_xml/{zip_idx}.zip"

                if not path.exists(path.join(CACHE_PATH, p_dir, 'zips', 'page_xml', f'{zip_idx}.zip')):
                    try:
                        sftp.stat(zip_path)
                    except FileNotFoundError:
                        print(
                            f'{zip_path} does not exist on remote host for uuid {image_uuid}')
                        continue
                    print('downloading uuid', image_uuid, 'path', zip_path)
                    sftp.get(
                        zip_path, f"{CACHE_PATH}/{p_dir}/zips/page_xml/{zip_idx}.zip")
                else:
                    print(
                        f'zip {zip_idx} is already downloaded for uuid: {image_uuid}')
            else:
                continue


def create_image_to_zip_mapping_local(p_dirs: list[str]) -> dict[str, dict[str, int | Any] | Any]:
    processing_dict = {}
    for p_dir in p_dirs:
        print('start', p_dir)
        # create processing dir, if it does not exist
        if not path.isdir(path.join(CACHE_PATH, p_dir)):
            os.mkdir(path.join(CACHE_PATH, p_dir))

        # download and create part files index, if it does not exist
        if not path.isfile(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING)):
            print(f'Image index for {p_dir} isn\'t cached. Creating index ...')
            # create a temporary index from the part_files.txt
            with open(path.join(CACHE_PATH, p_dir, 'part_files.txt'), 'r') as part_file_idx:
                part_file_dict = {}
                for l_number, line in enumerate(part_file_idx.readlines()):
                    part_file_dict[line[:-1].split("/")
                                   [-1].split('_')[-1]] = l_number + 1

            del part_file_idx
            uuid_zip_dict = {}
            for part_file_name in os.listdir(path.join(CACHE_PATH, p_dir, 'splits')):
                print(f'processing: {p_dir}/{part_file_name} ...')
                with open(path.join(CACHE_PATH, p_dir, 'splits', part_file_name)) as part_file:
                    for line in part_file.readlines():
                        image_uuid = line[:-1].split(":")[-1][:-4]
                        uuid_zip_dict[image_uuid] = part_file_dict[part_file_name[5:]]
            with open(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING), 'wb') as f:
                pickle.dump(uuid_zip_dict, f)
                print('if', p_dir)
                processing_dict[p_dir] = uuid_zip_dict
        else:
            with open(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING), 'rb') as f:
                print('else', p_dir)
                processing_dict[p_dir] = pickle.load(f)
    return processing_dict


def create_image_to_zip_mapping(sftp: paramiko.SFTPClient, p_dirs: list[str]) -> dict[str, str]:
    for image_uuid in get_image_uuid_from_json():
        for p_dir in p_dirs:
            # create processing dir, if it does not exist
            if not path.isdir(path.join(CACHE_PATH, p_dir)):
                os.mkdir(path.join(CACHE_PATH, p_dir))

            # download and create part files index, if it does not exist
            if not path.isfile(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING)):
                print(
                    f'Image index for {p_dir} isn\'t cached. Creating index ...')
                # create a temporary index from the part_files.txt
                part_file_idx: paramiko.SFTPFile = sftp.open(
                    f"{REMOTE_PATH}/{p_dir}/part_files.txt", 'r')
                if not part_file_idx:
                    raise IOError
                part_file_dict = {}
                for l_number, line in enumerate(part_file_idx.readlines()):
                    part_file_dict[line[:-1].split("/")
                                   [-1].split('_')[-1]] = l_number + 1

                del part_file_idx
                uuid_zip_dict = {}
                for part_file_attrs in tqdm(sftp.listdir_iter(f"{REMOTE_PATH}/{p_dir}/splits")):
                    part_file_name: str = part_file_attrs.filename
                    part_file: paramiko.SFTPFile = sftp.open(
                        f"{REMOTE_PATH}/{p_dir}/splits/{part_file_name}")
                    print(part_file_name)
                    for line in part_file.readlines():
                        image_uuid = line[:-1].split(":")[-1][:-4]
                        uuid_zip_dict[image_uuid] = part_file_dict[part_file_name[5:]]
                with open(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING), 'wb') as f:
                    pickle.dump(uuid_zip_dict, f)
            else:
                with open(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING), 'rb') as f:
                    uuid_zip_dict = pickle.load(f)
    return uuid_zip_dict


def get_image_uuid_from_json(json_path: str = '../res/project-9-at-2024-03-05-17-19-577ee11f.json') -> list[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)
        for el in data:
            uuid = unquote((el['data']['image']).split('/')[-1])[5:-4]
            yield uuid


if __name__ == "__main__":
    main()
