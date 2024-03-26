import os
import shutil
folder_path = "RESULTS"

# Get a list of all subdirectories in the folder
subdirectories = [f.path for f in os.scandir(folder_path) if f.is_dir()]

# Iterate over each subdirectory
for subdirectory in subdirectories:
    # Check if the subdirectory contains any PNG or JPG files
    contains_image_files = any(
        file.endswith((".png", ".jpg")) for file in os.listdir(subdirectory)
    )

    # If the subdirectory does not contain any image files, delete it
    if not contains_image_files:
        shutil.rmtree(subdirectory)
