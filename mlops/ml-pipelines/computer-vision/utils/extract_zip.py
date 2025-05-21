import os
import zipfile

from prefect import task

@task(name='info-dir', description='Find informations about dataset')
def info_dir(dataset_path):
    count = 0
    for dirpath, _, filenames in os.walk(dataset_path):
        folder_name = os.path.basename(dirpath)
        file_count = len([f for f in filenames if not f.startswith(".")])
        count += file_count
    print(f"Folder: {folder_name}, Banyaknya berkas: {file_count}")

@task(name='extract-zip', description='Extracting zip file dataset')
def extract_zip(zip_path, extracted_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extracted_path)

    print(f"Data was extracted into {extracted_path} path")
    info_dir(extracted_path)