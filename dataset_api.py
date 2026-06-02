from flask import Flask, jsonify
import os

app = Flask(__name__)

# Path to the dataset directory (relative to project root)
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")

@app.get("/datasets")
def list_datasets():
    """Return a list of dataset folder names.
    This simple endpoint can be used by the front‑end or other services to discover
    available dataset categories.
    """
    try:
        items = [name for name in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, name))]
        return jsonify({"datasets": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render provides the port via the $PORT environment variable.
    import os
    port = int(os.getenv("PORT", "8001"))
    app.run(host="0.0.0.0", port=port)
