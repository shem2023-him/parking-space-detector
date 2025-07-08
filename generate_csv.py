import json
import pandas as pd
import os

# === Step 1: CONFIG ===
json_path = r"C:\Users\shemd\Desktop\parking space detector system\pkl dataset\train\train_labels.json"
output_csv = r"C:\Users\shemd\Desktop\parking space detector system\pkl dataset\train\labels.csv"

# === Step 2: Load COCO JSON format ===
with open(json_path, 'r') as f:
    data = json.load(f)

# === Step 3: Map image_id to filename and labels ===
image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

image_id_to_label = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    if image_id not in image_id_to_label:
        image_id_to_label[image_id] = 1 if category_id == 2 else 0  # 1=occupied, 0=empty

# === Step 4: Build a CSV with filename and label ===
rows = []
for image_id, label in image_id_to_label.items():
    filename = image_id_to_filename.get(image_id)
    if filename:
        rows.append({"filename": filename, "label": label})

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print(f"✅ CSV created at: {output_csv}")
print(df.head())
