from typing import Iterator, IO, Any

import paramiko
import credentials as creds
import os
from os import path
from multiprocessing import Process, Manager, Lock
import pickle
import zipfile

HOSTNAME = 'merlin.fit.vutbr.cz'
PORT = 22

LOCAL_PATH = '../res/downloads/'
REMOTE_PATH = '/mnt/matylda1/ikiss/pero/experiments/digiknihovny'
CACHE_PATH = '../res/cache'
IMAGE_ZIP_MAPPING = 'image_zip_mapping'
JSON_PATH = '../res/project-9-at-2024-03-05-17-19-577ee11f.json'


def processing_folder_generator(sftp: paramiko.SFTPClient, remote_path: str) -> Iterator[str]:
    for file_attrs in sftp.listdir_iter(remote_path):
        if file_attrs.filename.startswith("processing"):
            yield file_attrs.filename


class DataManager:
    def __init__(self,
                 remote_path: str,
                 local_path: str,
                 cache_path: str,
                 annotation_path: str,
                 host_name: str,
                 port: int):
        print('Initializing the data manager')
        transport: paramiko.Transport = paramiko.Transport((host_name, port))
        transport.connect(None, creds.USERNAME, creds.PASSWORD)

        self.sftp: paramiko.SFTPClient = paramiko.SFTPClient.from_transport(transport)
        self.remote_path = remote_path
        self.local_path = local_path
        self.cache_path = cache_path
        self.annotation_path = annotation_path
        self.p_dirs: tuple[str] = tuple(processing_folder_generator(self.sftp, self.remote_path))
        self.loaded_uuid_mappings: dict[str, dict[str, str]] = {}
        self.cached_zip_files: dict[tuple[str, str, str], zipfile.ZipFile] = {}
        self.lru_zip_keys: list[tuple[str, str, str]] = []
        self.zip_cache_size = 100
        print('Data manager initialization finished')

    def __find_zip_file_idx(self, image_uuid: str) -> (str, str):
        for p_dir in self.p_dirs:
            if p_dir_mapping := self.loaded_uuid_mappings.get(p_dir):
                if zip_file_idx := p_dir_mapping.get(image_uuid):
                    return p_dir, zip_file_idx
            # try to load uuid mapping into memory
            else:
                with open(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING), 'rb') as f:
                    print(f'loading {p_dir} mapping from disk')
                    uuid_zip_dict: dict[str, str] = pickle.load(f)
                    self.loaded_uuid_mappings[p_dir] = uuid_zip_dict
                    if image_uuid in uuid_zip_dict:
                        return p_dir, uuid_zip_dict[image_uuid]
        return self.__download_uuid_mapping(image_uuid)

    def __download_uuid_mapping(self, image_uuid: str) -> (str, str):
        # TODO jakub
        """
        implement a method to download the particular uuid mapping
        after download update the IN-MEMORY cache: self.loaded_uuid_mappings
        then set some flag that caches has been changed
        and add a mechanism to the DataManager to save all modified caches after halting
        :param image_uuid: uuid of the input image annotation
        :return: tuple of the processing dir name and a found zip file idx
        """
        return None, None

    def __fetch_zip_file(self, p_dir: str, zip_file_idx: str, image_uuid: str, file_type: str) -> zipfile.ZipFile | None:
        os.makedirs(path.join(self.cache_path, p_dir, 'zips', file_type), exist_ok=True)
        # zip file found in cache
        file_key = (p_dir, file_type, zip_file_idx)
        local_zip_path: str = path.join(self.cache_path, p_dir, 'zips', file_type, f'{zip_file_idx}.zip')
        if zip_file := self.cached_zip_files.get(file_key):
            # update the caching index
            self.lru_zip_keys.remove(file_key)
            self.lru_zip_keys.append(file_key)
            return zip_file
        # need to download
        elif not path.isfile(local_zip_path):
            remote_zip_path: str = f"{self.remote_path}/{p_dir}/zips/{file_type}/{zip_file_idx}.zip"

            try:
                self.sftp.stat(remote_zip_path)
            except FileNotFoundError:
                print(f"{remote_zip_path} does not exist on the remote host for uuid {image_uuid}")
                return None
            print(f"downloading '{file_type}' zip from {remote_zip_path}")
            self.sftp.get(remote_zip_path, f"{self.cache_path}/{p_dir}/zips/{file_type}/{zip_file_idx}.zip")

        # load the zip file from disk
        zip_file = zipfile.ZipFile(local_zip_path)
        if zip_file:
            # update the caching index
            if len(self.lru_zip_keys) >= self.zip_cache_size:
                cached_key_to_remove: tuple[str, str, str] = self.lru_zip_keys.pop(0)
                cached_zip_file: zipfile.ZipFile = self.cached_zip_files[cached_key_to_remove]
                cached_zip_file.close()
                self.cached_zip_files.pop(cached_key_to_remove)
            self.cached_zip_files[file_key] = zip_file
            if file_key in self.lru_zip_keys:
                self.lru_zip_keys.remove(file_key)
            self.lru_zip_keys.append(file_key)
            return zip_file
        else:
            return None

    def get_xml_file(self, image_uuid: str) -> IO[bytes] | None:
        (p_dir, zip_file_idx) = self.__find_zip_file_idx(image_uuid)
        zip_file: zipfile.ZipFile = self.__fetch_zip_file(p_dir, zip_file_idx, image_uuid, 'page_xml')
        if zip_file:
            return zip_file.open(f"uuid:{image_uuid}.xml", 'r')
        return None

    def get_image_crops(self, image_uuid: str, label_suffix: str) -> IO[bytes] | None:
        (p_dir, zip_file_idx) = self.__find_zip_file_idx(image_uuid)
        zip_file: zipfile.ZipFile = self.__fetch_zip_file(p_dir, zip_file_idx, image_uuid, 'crops')
        if zip_file:
            return zip_file.open(f"uuid:{image_uuid}.png", 'r')
        return None

    def get_image_labels(self, image_uuid: str) -> IO[bytes] | None:
        (p_dir, zip_file_idx) = self.__find_zip_file_idx(image_uuid)
        zip_file: zipfile.ZipFile = self.__fetch_zip_file(p_dir, zip_file_idx, image_uuid, 'labels')
        if zip_file:
            return zip_file.open(f"uuid:{image_uuid}.txt", 'r')
        return None


# create a global data manager
dataManager = DataManager(
    REMOTE_PATH,
    LOCAL_PATH,
    CACHE_PATH,
    JSON_PATH,
    HOSTNAME,
    PORT
)

def create_uuid_zip_map(todo_list: list[str], folder_name: str, exists: bool) -> None:
    """Processing folders with info about images and their zip files"""
    print(f"Processing folder {folder_name}")

    ssh_client: paramiko.SSHClient = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(HOSTNAME, PORT, creds.USERNAME, creds.PASSWORD)

    if len(todo_list) == 0:
        ssh_client.close()
        return

    uuid_zip_dict: dict[str, int] = {}

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

    todo_list: list[str] = []  # [uuid for uuid in get_image_uuids_from_json()]
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


def create_image_to_zip_mapping_local(p_dirs: list[str]) -> dict[str, dict[str, int | Any] | Any]:
    processing_dict = {}
    for p_dir in p_dirs:
        print('started loading: ', p_dir)
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
                    part_file_dict[line[:-1].split("/")[-1].split('_')[-1]] = l_number + 1

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
                processing_dict[p_dir] = uuid_zip_dict
        else:
            with open(path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING), 'rb') as f:
                processing_dict[p_dir] = pickle.load(f)
    return processing_dict
