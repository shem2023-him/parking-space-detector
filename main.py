import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1. Prepare Data Directory ----------

def prepare_dataset_structure(source_dir='raw_images', target_dir='data', split_ratio=(0.7, 0.15, 0.15)):
    if not os.path.exists(source_dir):
        raise Exception(f"Source folder '{source_dir}' does not exist. Please add images there.")
    
    classes = os.listdir(source_dir)
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)

    for class_name in classes:
        images = os.listdir(os.path.join(source_dir, class_name))
        random.shuffle(images)
        total = len(images)
        train_end = int(split_ratio[0] * total)
        val_end = train_end + int(split_ratio[1] * total)
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        for split, split_files in splits.items():
            for file in split_files:
                src_path = os.path.join(source_dir, class_name, file)
                dst_path = os.path.join(target_dir, split, class_name, file)
                shutil.copy(src_path, dst_path)
    print("✅ Images split into train/val/test folders.")

# Only split if data/train is empty
if not os.path.exists("data/train/empty") or len(os.listdir("data/train/empty")) == 0:
    prepare_dataset_structure()

# ---------- 2. Image Generators ----------

img_size = (150, 150)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory("data/train", target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = val_gen.flow_from_directory("data/val", target_size=img_size, batch_size=batch_size, class_mode='binary')
test_data = test_gen.flow_from_directory("data/test", target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# ---------- 3. Define Model ----------

model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# ---------- 4. Train Model ----------

history = model.fit(train_data, validation_data=val_data, epochs=5)

# ---------- 5. Evaluate Model ----------

loss, acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# ---------- 6. Classification Report & Confusion Matrix ----------

y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ---------- 7. Save Model ----------

model.save("parking_space_model.h5")
print("✅ Model saved as parking_space_model.h5")

