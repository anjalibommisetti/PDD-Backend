import os
import io
import numpy as np
from PIL import Image
from typing import Optional

# Global cache variables
dataset_hashes = None
dataset_labels = None

class_names = [
    "Calculus",
    "Early Childhood Caries",
    "Gingivitis",
    "Tooth Discoloration",
    "Ulcers",
    "Hypodontia",
]
class_to_idx = {name: i for i, name in enumerate(class_names)}

def dhash(image: Image.Image) -> np.ndarray:
    # Resize to 9x8 and convert to grayscale
    img = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.int16)
    # Compute horizontal gradients
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def load_model():
    """Load the precomputed dataset hashes."""
    global dataset_hashes, dataset_labels
    cache_path = os.getenv("HASH_CACHE_PATH", "ml/dataset_hashes.npz")
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            dataset_hashes = data["hashes"]
            dataset_labels = data["labels"]
            print(f"[INFO] Loaded {len(dataset_hashes)} dataset hashes for similarity prediction.")
        except Exception as e:
            print(f"[ERROR] Failed to load hash cache: {e}")
            dataset_hashes = None
            dataset_labels = None
    else:
        print("[WARNING] Hash cache not found. Prediction will fall back to deterministic mock.")

# Initialize on import
load_model()

class DeterministicMockModel:
    def __init__(self, image_bytes: bytes):
        import hashlib
        h = hashlib.sha256(image_bytes).hexdigest()
        seed = int(h[:16], 16) % (2**32)
        self.rng = np.random.default_rng(seed)

    def predict(self) -> np.ndarray:
        probs = self.rng.random(6)
        probs = probs / probs.sum()
        return np.expand_dims(probs, axis=0)

def predict(image_bytes: bytes) -> np.ndarray:
    """Find the closest matching image in the dataset using dHash similarity.
    If the dataset cache is not loaded, falls back to the deterministic mock.
    """
    global dataset_hashes, dataset_labels
    if dataset_hashes is None or dataset_labels is None:
        # Try loading again in case it was built since import
        load_model()
        
    if dataset_hashes is not None and dataset_labels is not None and len(dataset_hashes) > 0:
        try:
            # Load and hash the input image
            img = Image.open(io.BytesIO(image_bytes))
            query_hash = dhash(img)
            
            # Compute Hamming distance to all cached hashes
            # dataset_hashes is shape (N, 64), query_hash is shape (64,)
            distances = np.count_nonzero(dataset_hashes != query_hash, axis=1)
            best_match_idx = np.argmin(distances)
            min_distance = distances[best_match_idx]
            matched_label = dataset_labels[best_match_idx]
            
            # Calculate similarity percentage (64 bits total)
            similarity = 1.0 - (min_distance / 64.0)
            
            # Construct probability output
            probs = np.zeros(6)
            idx = class_to_idx.get(matched_label, 0)
            probs[idx] = max(0.5, similarity) # Match probability
            
            # Distribute remaining probability among other classes
            remaining = 1.0 - probs[idx]
            for i in range(6):
                if i != idx:
                    probs[i] = remaining / 5.0
            return probs
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            # Fall through to mock on error
            
    # Mock fallback
    mock = DeterministicMockModel(image_bytes)
    return mock.predict()[0]
