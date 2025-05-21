import os
import zipfile

def info_dir(dataset_path):
    for dirpath, _, filenames in os.walk(dataset_path):
        folder_name = os.path.basename(dirpath)
        file_count = len([f for f in filenames if not f.startswith(".")]) 
        print(f"Folder: {folder_name}, Banyaknya berkas: {file_count}")

def extract_zip(zip_path, extracted_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extracted_path)

    print(f"Data was extracted into {extracted_path} path")

    info_dir(extracted_path)