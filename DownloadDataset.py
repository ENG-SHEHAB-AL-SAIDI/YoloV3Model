import requests
from tqdm import tqdm  # Optional: For displaying a progress bar
import zipfile
import os

# Define the URL of the file and the destination path in Google Drive
file_url = 'https://www.kaggle.com/api/v1/datasets/download/achrafkhazri/labeled-licence-plates-dataset?datasetVersionNumber=1'
destination_path = 'archive.zip'
extraction_dir = '.'

os.makedirs(extraction_dir, exist_ok=True)
# Make the request to download the file
response = requests.get(file_url, stream=True)
if response.status_code == 200:
    # Get the total file size (if available)
    total_size = int(response.headers.get('content-length', 0))

    # Open the destination file
    with open(destination_path, 'wb') as file:
        # Use tqdm to show a progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

    print("File downloaded successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")


# unzip downloaded archive

# Define the path to the zip file and the extraction directory

# Create extraction directory if it does not exist
os.makedirs(extraction_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(destination_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

print(f"Zip file extracted to {extraction_dir}.")

currentFolderName = 'dataset'
newFolderName = 'DataSet'

# Rename the folder
os.rename(currentFolderName, newFolderName)