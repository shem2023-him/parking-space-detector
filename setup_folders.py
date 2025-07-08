# setup_folders.py

import os

# Define the base path
base_path = os.path.join(os.getcwd(), "data")

# Define subfolders
subfolders = [
    "train/empty",
    "train/occupied",
    "val/empty",
    "val/occupied",
    "test/empty",
    "test/occupied"
]

# Create folders
for subfolder in subfolders:
    folder_path = os.path.join(base_path, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"✅ Created: {folder_path}")

print("\n🎉 Folder structure created successfully!")
