import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import subprocess
import json


ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

# Configuration
model_name = "sentence-transformers/gtr-t5-xxl"
output_dir = os.path.join(ROOT_DIR,"embeddings")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
output_file = f"{output_dir}/paragraph_embeddings.npz"

# Example paragraphs (replace with your actual paragraphs)
paragraphs = [
    "Paragraph 1: This is the first example paragraph.",
    "Paragraph 2: This is the second example paragraph.",
    "Paragraph 3: This is the third example paragraph.",
]

## Read Json
with open(os.path.join(ROOT_DIR, "data", "passages.json")) as f:
    record = json.load(f)

electronic_record = record["PATIENT_ID"]["ER"]
paragraphs = []

for i in range(len(electronic_record)):
    for passage in electronic_record[i]["PASSAGES"]:
        paragraphs.append(passage)

# Load the model
model = SentenceTransformer(model_name)

# Encode paragraphs
print("Encoding paragraphs...")
paragraph_embeddings = model.encode(paragraphs, show_progress_bar=True, batch_size=16, convert_to_numpy=True)

# Save embeddings and paragraphs to a file
np.savez(output_file, embeddings=paragraph_embeddings, paragraphs=paragraphs)
print(f"Embeddings and paragraphs saved to {output_file}.")
