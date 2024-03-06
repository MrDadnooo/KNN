import json
from fabric import Connection
import paramiko, os, zipfile, credentials as creds, io, pickle
from urllib.parse import unquote

HOSTNAME = 'merlin.fit.vutbr.cz'
PORT = 22

LOCAL_PATH = '../res/downloads/'
REMOTE_PATH = '/mnt/matylda1/ikiss/pero/experiments/digiknihovny'
PROCESSING_2023_09_18 = 'processing.2023-09-18'
IDK_JSON_FILE = '../res/project-9-at-2024-03-05-17-19-577ee11f.json'


def main():
    transport: paramiko.Transport = paramiko.Transport((HOSTNAME, PORT))
    transport.connect(None, creds.USERNAME, creds.PASSWORD)

    sftp: paramiko.SFTPClient = paramiko.SFTPClient.from_transport(transport)

    # Download file from remote server
    # sftp.get(REMOTE_ZIPS_PATH + LABELS + '1.zip', LOCAL_FILE_PATH + LABELS + '1.zip')
    if not os.path.exists(os.path.join(LOCAL_PATH, 'splits')):
        os.mkdir(os.path.join(LOCAL_PATH, 'splits'))

    remote_splits_path = f"{REMOTE_PATH}/{PROCESSING_2023_09_18}/splits/"
    remote_page_xml_path = f"{REMOTE_PATH}/{PROCESSING_2023_09_18}/zips/page_xml"

    # for entry in sftp.listdir_attr(remote_splits_path):
    #     print(entry.filename, end=' ')
    #
    #     sftp.get(f"{remote_splits_path}/{entry.filename}", f"{LOCAL_PATH}/splits/{entry.filename}")
    # def path(idx: int):
    #     return f"{LOCAL_PATH}/page_xml/{idx}.zip", f'{remote_page_xml_path}/{idx}.zip'
    #
    # zip_100_path = f"{LOCAL_PATH}/page_xml/100.zip"
    # zip_file_idx = get_zip_file_idx(sftp, )

    for uuid in get_annotation_image_uuid():
        zip_idx = get_zip_file_idx(sftp, uuid)
        if zip_idx:
            print(uuid, zip_idx)
        else:
            print(uuid, 'could\'nt find uuid in the processing2023-09-18')

    if sftp:
        sftp.close()
    if transport:
        transport.close()


def get_zip_file_idx(sftp: paramiko.SFTPClient, uuid: str, mapping_path: str = '../res/cache/uuid_mapping'):
    uuid_mapping = {}
    if os.path.isfile(mapping_path):
        with open(mapping_path, 'rb') as f:
            uuid_mapping = pickle.load(f)

    if not uuid_mapping.get(uuid):
        print(f'Could not find uuid: {uuid} in cached storage ...downloading')
        with io.BytesIO() as fl:
            sftp.getfo(f"{REMOTE_PATH}/{PROCESSING_2023_09_18}/crops.all", fl)
            fl.seek(0)
            while True:
                line = fl.readline()
                if not line:
                    break
                line_str = line.decode('utf-8').rstrip()
                # splitting magik
                l_side = line_str.split('|')[0]
                r_side = line_str.split('|')[1]
                zip_file_name = l_side.split('/')[-1]
                idx = int(zip_file_name[:-4])
                uuid_with_sub = r_side[5:-4]
                uuid_temp = uuid_with_sub.split('__')[0]
                uuid_mapping[uuid_temp] = idx

        with open(mapping_path, 'wb') as f:
            pickle.dump(uuid_mapping, f)

    return uuid_mapping.get(uuid)


def get_annotation_image_uuid(json_path: str = '../res/project-9-at-2024-03-05-17-19-577ee11f.json') -> list[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)
        for el in data:
            uuid = unquote((el['data']['image']).split('/')[-1])[5:-4]
            yield uuid


if __name__ == "__main__":
    main()
    # foo()
