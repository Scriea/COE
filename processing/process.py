import os
import sys
import json
import torch
import argparse
import numpy as np
import subprocess

# from huggingface_hub import login
# login()

ROOT_DIR = (
    subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)
sys.path.append(ROOT_DIR)
from src.attribution.attrb import AttributionModule
from src.generator.generator import Generator
from src.generator import prompts
from src.hallucination.hallucination import HalluCheck

# General Configuration
CHECPOINT_FILE = os.path.join(ROOT_DIR, "run", "checkpoint.json")
os.makedirs(os.path.dirname(CHECPOINT_FILE), exist_ok=True)

EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# chat_model = "google/gemma-2-9b"
# chat_model = "medalpaca/medalpaca-7b"
# chat_model = "meta-llama/Meta-Llama-3-8B"
# chat_model = "google/gemma-7b"
chat_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# hallucination_model = "meta-llama/Llama-2-7b-chat-hf"
# hallucination_model = "HPAI-BSC/Llama3-Aloe-8B-Alpha"
hallucination_model = "BioMistral/BioMistral-7B"


# Utility Functions
def save_checkpoint(data, checkpoint_file):
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference on a set of questions")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to test data in .json.",
        default=os.path.join(ROOT_DIR, "data", "test_data.json"),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to the model to use.",
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output file.",
        default=os.path.join(ROOT_DIR, "run", "output.json"),
    )

    args = parser.parse_args()
    # chat_model = args.model

    # Load Modules
    attribution_module = AttributionModule(device="cuda:2")
    generator = Generator(model_path=chat_model, device="cuda:2")
    hallucination_checker = HalluCheck(
        device="cuda:4", method="POS", model_path=hallucination_model
    )

    # Processing Document
    passages_file = os.path.join(ROOT_DIR, "data", "passages.json")
    passages = attribution_module.load_paragraphs(passages_file=passages_file)
    joint_passages = "\n".join(passages)

    embedding_file_path = os.path.join(
        attribution_module.output_dir, "paragraph_embeddings.npz"
    )
    if not os.path.exists(embedding_file_path):
        attribution_module.vectorize_paragraphs(passages_file, embedding_file_path)

    search_index, paragraphs = attribution_module.create_faiss_index(
        embedding_file_path=embedding_file_path, ngpu=1
    )

    def get_response(query):
        # Generate response
        prompt = prompts.medical_prompt.format(joint_passages, query)
        prompt_fs = prompts.medical_prompt_fs.format(joint_passages, query)
        answer = generator.generate_response(prompt)
        answer_rag1 = generator.vector_compare(prompt, joint_passages)
        answer_fs = generator.generate_response(prompt_fs)
        answer_rag = answer_rag1.response
        # Attribution
        attrb = {answer: "", answer_rag: "", answer_fs: ""}
        for _ in attrb.keys():
            attribution_query = f"Question: {query}\nAnswer: {_}"
            retrieval_results = attribution_module.retrieve_paragraphs(
                [attribution_query], search_index, paragraphs, k=1
            )
            retrieved_passages = retrieval_results[0]["retrieved_paragraphs"][0]
            attrb[_] = retrieved_passages
        # Check for hallucination
        hallucination_probability = hallucination_checker.hallucination_prop(
            answer, context="joint_passages"
        )
        # Compile the response message
        response = {
            "question": query,
            "answer": answer,
            "answer_rag": answer_rag,
            "answer_fs": answer_fs,
            "attribution": attrb[answer],
            "attribution_rag": attrb[answer_rag],
            "attribution_fs": attrb[answer_fs],
            "hallucination": hallucination_probability,
        }
        return response

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"File not found: {args.data}")

    with open(args.data, "r") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    eval_data = {}
    for item in data:
        query = item["question"]
        response = get_response(query)
        eval_data[query] = response
        print(response)
        save_checkpoint(eval_data, args.output)

    with open(args.output, "w") as f:
        json.dump(eval_data, f, indent=4)
