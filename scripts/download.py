from typing import Iterator, IO, Any

import paramiko
import credentials as creds
import os
from os import path
import threading
import pickle
import zipfile
from PIL import Image
from annotation import ImageLabelData

HOSTNAME = 'merlin.fit.vutbr.cz'
PORT = 22

REMOTE_PATH = '/mnt/matylda1/ikiss/pero/experiments/digiknihovny'
CACHE_PATH = '../res/cache'
IMAGE_ZIP_MAPPING = 'image_zip_mapping'
# JSON_PATH = '../res/project-9-at-2024-03-05-17-19-577ee11f.json'
JSON_PATH = '../res/project-9-at-2024-04-23-08-54-b684460e.json'


def processing_folder_generator(sftp: paramiko.SFTPClient, remote_path: str) -> Iterator[str]:
    for file_attrs in sftp.listdir_iter(remote_path):
        if file_attrs.filename.startswith("processing"):
            yield file_attrs.filename


class DataManager:
    def __init__(self,
                 remote_path: str,
                 cache_path: str,
                 annotation_path: str,
                 host_name: str,
                 port: int):
        print('Initializing the data manager')
        ssh_client: paramiko.SSHClient = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(HOSTNAME, PORT, creds.USERNAME, creds.PASSWORD)
        transport: paramiko.Transport = paramiko.Transport((host_name, port))
        transport.connect(None, creds.USERNAME, creds.PASSWORD)

        self.sftp: paramiko.SFTPClient = paramiko.SFTPClient.from_transport(transport)
        self.ssh_client: paramiko.SSHClient = ssh_client
        self.remote_path = remote_path
        self.cache_path = cache_path
        self.annotation_path = annotation_path
        self.p_dirs: tuple[str] = tuple(processing_folder_generator(self.sftp, self.remote_path))
        self.loaded_uuid_mappings: dict[str, dict[str, str]] = {}
        self.cached_zip_files: dict[tuple[str, str, str], zipfile.ZipFile] = {}
        self.lru_zip_keys: list[tuple[str, str, str]] = []
        self.zip_cache_size = 100
        self.stop_event = threading.Event()
        print('Data manager initialization finished')

    def __find_zip_file_idx(self, image_uuid: str) -> (str, str):
        for p_dir in self.p_dirs:
            mapping_path = path.join(CACHE_PATH, p_dir, IMAGE_ZIP_MAPPING)
            if p_dir_mapping := self.loaded_uuid_mappings.get(p_dir):
                if zip_file_idx := p_dir_mapping.get(image_uuid):
                    return p_dir, zip_file_idx
            # try to load uuid mapping into memory
            elif path.isfile(mapping_path):
                with open(mapping_path, 'rb') as f:
                    print(f'loading {p_dir} mapping from disk')
                    uuid_zip_dict: dict[str, str] = pickle.load(f)
                    self.loaded_uuid_mappings[p_dir] = uuid_zip_dict
                    if image_uuid in uuid_zip_dict:
                        return p_dir, uuid_zip_dict[image_uuid]
        # mapping entry does not exist locally
        return self.__download_uuid_mapping(image_uuid)

    def __download_uuid_mapping(self, image_uuid: str) -> (str, str):

        for p_dir in self.p_dirs:
            grep_command: str = f"grep -r {image_uuid} {self.remote_path}/{p_dir}/splits/"
            stdin, stdout, stderr = self.ssh_client.exec_command(grep_command)
            output = stdout.read().decode()
            if output:
                grep_command_2 : str = f"grep -n {output.split(':')[0][-9:]} {self.remote_path}/{p_dir}/part_files.txt"
                stdin, stdout, stderr = self.ssh_client.exec_command(grep_command_2)
                output = stdout.read().decode()
                if output:
                    val = output.split(':')[0]
                    _dir = p_dir
                    break

        # check if file exists
        if not path.isfile(f'{self.cache_path}/{_dir}'):
            # create directory
            os.makedirs(f'{self.cache_path}/{_dir}', exist_ok=True)
            # create dictionary
            uuid_zip_dict = {image_uuid: int(val)}
            # save dictionary to file
            with open(f'{self.cache_path}/{_dir}/{IMAGE_ZIP_MAPPING}', 'wb') as f:
                pickle.dump(uuid_zip_dict, f)
        else:
            # update dictionary
            with open(f'{self.cache_path}/{_dir}/{IMAGE_ZIP_MAPPING}', 'rb') as f:
                uuid_zip_dict = pickle.load(f)
                uuid_zip_dict[image_uuid] = int(val)
            # save dictionary to file
            with open(f'{self.cache_path}/{_dir}/{IMAGE_ZIP_MAPPING}', 'wb') as f:
                pickle.dump(uuid_zip_dict, f)  
        
        return _dir, val

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

    def get_image_crops(self, image_uuid: str, image_label: ImageLabelData) -> Image:
        (p_dir, zip_file_idx) = self.__find_zip_file_idx(image_uuid)
        zip_file: zipfile.ZipFile = self.__fetch_zip_file(p_dir, zip_file_idx, image_uuid, 'crops')
        if zip_file:
            img_name: str = f"uuid:{image_uuid}__{image_label.label}_{image_label.idx}.jpg"
            return Image.open(zip_file.open(img_name, 'r'))
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
    CACHE_PATH,
    JSON_PATH,
    HOSTNAME,
    PORT
)


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
