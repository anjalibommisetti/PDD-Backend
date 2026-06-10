import os
import time
from PIL import Image
import numpy as np

DATASET_ROOT = "dataset"
class_names = [
    "Calculus",
    "Early Childhood Caries",
    "Gingivitis",
    "Tooth Discoloration",
    "Ulcers",
    "Hypodontia",
]

folder_mapping = {
    "Calculus": "Calculus",
    "Data caries": "Early Childhood Caries",
    "Gingivitis": "Gingivitis",
    "Tooth Discoloration": "Tooth Discoloration",
    "Mouth Ulcer": "Ulcers",
    "hypodontia": "Hypodontia",
}

def dhash(image: Image.Image) -> np.ndarray:
    img = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def scan_dataset():
    start = time.time()
    images_found = 0
    hashes = []
    labels = []
    
    if not os.path.exists(DATASET_ROOT):
        print(f"Dataset root {DATASET_ROOT} not found.")
        return
        
    for item in os.listdir(DATASET_ROOT):
        item_path = os.path.join(DATASET_ROOT, item)
        if not os.path.isdir(item_path):
            continue
        if item in folder_mapping:
            label = folder_mapping[item]
            # Recursively find images under this directory
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images_found += 1
                        # Limit loading for benchmark
                        if images_found <= 100:
                            img_path = os.path.join(root, file)
                            try:
                                with Image.open(img_path) as img:
                                    h = dhash(img)
                                    hashes.append(h)
                                    labels.append(label)
                            except Exception as e:
                                pass
    end = time.time()
    print(f"Found total {images_found} images.")
    print(f"Time to hash first 100 images: {end - start:.4f} seconds")

if __name__ == "__main__":
    scan_dataset()
