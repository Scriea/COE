import os
import json
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import faiss
import torch

ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
DATA_DIR = os.path.join(ROOT_DIR, "data")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttributionModule:
    def __init__(self, model_name="sentence-transformers/gtr-t5-xxl", device=device, output_dir=EMBEDDINGS_DIR ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def vectorize_paragraphs(self, passages_file, output_file=os.path.join(EMBEDDINGS_DIR, "paragraph_embeddings.npz")):
        paragraphs = self.load_paragraphs(passages_file)
        
        print("Encoding paragraphs...")
        paragraph_embeddings = self.model.encode(paragraphs, show_progress_bar=True, batch_size=16, convert_to_numpy=True, device=self.device)
        
        output_path = os.path.join(self.output_dir, output_file)
        np.savez(output_path, embeddings=paragraph_embeddings, paragraphs=paragraphs)
        print(f"Embeddings and paragraphs saved to {output_path}.")
    
    def load_paragraphs(self, passages_file):
        with open(passages_file) as f:
            record = json.load(f)
        patient_electronic_record = record[0]["PATIENT_1"]["ER"]
        
        paragraphs = []
        for record in patient_electronic_record:
            for passage in record["PASSAGES"]:
                paragraphs.append(passage)
        
        return paragraphs

    def create_faiss_index(self, embedding_file_path, ngpu=-1, gpus=None):
        data = np.load(embedding_file_path)
        paragraph_embeddings = data['embeddings']
        
        dimension = paragraph_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(paragraph_embeddings)
        
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        search_index = faiss.index_cpu_to_gpus_list(index, co=co, ngpu=ngpu, gpus=gpus)
        
        return search_index, data['paragraphs']

    def retrieve_paragraphs(self, texts, search_index, paragraphs, k=1):
        query_embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=16, convert_to_numpy=True, device=self.device)
        D, I = search_index.search(query_embeddings, k)
        return self.create_retrieval_results(texts, I, paragraphs, D)

    def create_retrieval_results(self, queries, indices, paragraphs, distances):
        results = []
        for i, query in enumerate(tqdm(queries, desc="Processing query results")):
            retrieved_paragraphs = [paragraphs[idx] for idx in indices[i]]
            results.append({
                "query": query,
                "retrieved_paragraphs": retrieved_paragraphs,
                "distances": distances[i].tolist()
            })
        return results

    def save_results(self, results, output_file="retrieval_results.json"):
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Retrieval results saved to {output_path}.")

if __name__ == "__main__":

    module = AttributionModule(device="cuda:7")    
    passage_file = os.path.join(DATA_DIR,"passages.json")
    module.vectorize_paragraphs(passages_file=passage_file)

    # Create FAISS index
    embedding_file_path=os.path.join(EMBEDDINGS_DIR, "paragraph_embeddings.npz")
    search_index, paragraphs = module.create_faiss_index(embedding_file_path=embedding_file_path, gpus=[7])

    # Example user queries
    user_queries = [
        "I am feeling chest pain what should I do?",
        "What is my height?",
        "What should i do, i am feeling fever? "
    ]

    # Retrieve paragraphs for user queries
    retrieval_results = module.retrieve_paragraphs(user_queries, search_index, paragraphs)

    # Save results
    module.save_results(retrieval_results)
