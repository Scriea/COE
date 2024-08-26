import os
import sys
import argparse
import json
import subprocess
import torch
import numpy as np
import tqdm

ROOT_DIR = (
    subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)
sys.path.append(ROOT_DIR)
from src.attribution.evaluation import AutoAIS

import warnings

warnings.filterwarnings("ignore")

device = "cuda:3" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AutoAIS on a given text and attribution."
    )
    parser.add_argument(
        "-d", "--data", type=str, help="Path to evaluation file json.", required=True
    )
    args = parser.parse_args()

    questions = []
    answers = []
    passages = []

    with open(args.data, "r") as f:
        data = json.load(f)

    for key, val in data.items():
        response = data[key]

        questions.append(response["question"])
        answers.append(response["answer"])
        passages.append(response["attribution"])

    score = AutoAIS(questions, answers, passages, device=device)
    print(f"AutoAIS : {score}")
