
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import subprocess


ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

# Configuration
model_name = "sentence-transformers/gtr-t5-xxl"
output_dir = os.path.join(ROOT_DIR,"embeddings")
embedding_file = f"{output_dir}/paragraph_embeddings.npz"

# Load the model
model = SentenceTransformer(model_name)

# Load embeddings and paragraphs
data = np.load(embedding_file)
paragraph_embeddings = data['embeddings']
paragraphs = data['paragraphs']

# Create FAISS index
print("Creating FAISS index...")
dimension = paragraph_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(paragraph_embeddings)

# Move index to GPUs
ngpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
co = faiss.GpuMultipleClonerOptions()
co.shard = True
search_index = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)

# Function to encode and search queries
def retrieve_paragraphs(queries, model, index, k=1):
    query_embeddings = model.encode(queries, show_progress_bar=True, batch_size=16, convert_to_numpy=True)
    D, I = index.search(query_embeddings, k)
    return D, I

# Example user queries
user_queries = [
    "I am feeling chest pain what should i do ?",
    "What is my height ?"
]

# Retrieve paragraphs for user queries
D, I = retrieve_paragraphs(user_queries, model, search_index)

# Process results
def create_retrieval_results(queries, indices, paragraphs, distances, set_type='query'):
    results = []
    for i, query in enumerate(tqdm(queries, desc=f"Processing {set_type} results")):
        retrieved_paragraphs = [paragraphs[idx] for idx in indices[i]]
        results.append({
            "query": query,
            "retrieved_paragraphs": retrieved_paragraphs,
            "distances": distances[i].tolist()
        })
    return results

# Create retrieval results
retrieval_results = create_retrieval_results(user_queries, I, paragraphs, D)

# Save results as JSON
output_file = f"{output_dir}/retrieval_results.json"
with open(output_file, "w") as f:
    json.dump(retrieval_results, f, indent=4)

print(f"Retrieval results saved to {output_file}.")
