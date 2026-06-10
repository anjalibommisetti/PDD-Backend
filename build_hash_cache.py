import os
import time
from PIL import Image
import numpy as np

DATASET_ROOT = "dataset"
OUTPUT_PATH = "ml/dataset_hashes.npz"

folder_mapping = {
    "Calculus": "Calculus",
    "Data caries": "Early Childhood Caries",
    "Gingivitis": "Gingivitis",
    "Tooth Discoloration": "Tooth Discoloration",
    "Mouth Ulcer": "Ulcers",
    "hypodontia": "Hypodontia",
}

def dhash(image: Image.Image) -> np.ndarray:
    # Resize to 9x8 and convert to grayscale
    img = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.int16)
    # Compute horizontal gradients
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def build_cache():
    print("Starting hash cache generation from dataset...")
    start_time = time.time()
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    hashes = []
    labels = []
    
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root {DATASET_ROOT} not found.")
        return
        
    total_scanned = 0
    for item in os.listdir(DATASET_ROOT):
        item_path = os.path.join(DATASET_ROOT, item)
        if not os.path.isdir(item_path):
            continue
        if item in folder_mapping:
            label = folder_mapping[item]
            print(f"Scanning category: '{item}' (maps to '{label}')...")
            category_count = 0
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, file)
                        try:
                            with Image.open(img_path) as img:
                                h = dhash(img)
                                hashes.append(h)
                                # Store label as string
                                labels.append(label)
                                category_count += 1
                                total_scanned += 1
                        except Exception as e:
                            # Skip corrupt/unsupported images
                            pass
            print(f"Finished category '{item}': processed {category_count} images.")

    if not hashes:
        print("Error: No images processed.")
        return
        
    hashes_arr = np.array(hashes, dtype=bool)
    labels_arr = np.array(labels)
    
    np.savez_compressed(OUTPUT_PATH, hashes=hashes_arr, labels=labels_arr)
    duration = time.time() - start_time
    print(f"Success! Processed {total_scanned} images and saved cache to '{OUTPUT_PATH}' in {duration:.2f} seconds.")

if __name__ == "__main__":
    build_cache()
