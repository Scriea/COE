import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import re
import json
import torch
import numpy as np
import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

AUTOAIS = "google/t5_xxl_true_nli_mixture"
device="cuda" if torch.cuda.is_available() else "cpu"

def AutoAIS(queries, answers, passages, model_path=None, device=device, output_path=None, device_map="auto"):
    """
    Calculate the AutoAIS score for the given text and attribution.

    Args:
        queries (list): List of questions.
        answers (list): List of answers.
        passages (list): List of passages.
        model_path (str): Path to the model.
        device (str): Device to use for computation.
        output_path (str): Path to save the output.

    Returns:
        float: The AutoAIS score.
    """
    autoais = 0

    if not model_path:
      hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS)
      hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS, device_map=device_map)
    else:
      hf_tokenizer = T5Tokenizer.from_pretrained(model_path)
      hf_model = T5ForConditionalGeneration.from_pretrained(model_path, device_map=device_map)

    ### Check if the input is a list of queries, texts and attributions
    assert len(queries) == len(answers) == len(passages)
    result = {}
    for i in tqdm.tqdm(range(len(queries)), desc="Calculating AutoAIS"):
      example = {}
      example["question"] = queries[i]
      example["answer"] = answers[i]
      example["passage"] = passages[i]

      inference = infer_autoais(example, hf_tokenizer, hf_model, device)
      autoais += inference == "Y"
      result[queries[i]] = example
    
    if not output_path:
       output_path = "output.json"
    with open(output_path, "w") as f:
      json.dump(result, f)
    
    score = autoais / len(queries) 
    return score


def format_example_for_autoais(example):
    """
      Formats an example for AutoAIS inference.
      example: Dict with the example data.
      Returns: A string representing the formatted example.

    
    """
    return "premise: {} hypothesis: The answer to the question '{}' is '{}'".format(
       example["passage"],
       example["question"],
       example["answer"]
    )


def infer_autoais(example, tokenizer, model, device=device):
  """Runs inference for assessing AIS between a premise and hypothesis.

  Args:
    example: Dict with the example data.
    tokenizer: A huggingface tokenizer object.
    model: A huggingface model object.

  Returns:
    A string representing the model prediction.
  """
  input_text = format_example_for_autoais(example)
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
  model.eval()
  with torch.no_grad():
    outputs = model.generate(input_ids)
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  inference = "Y" if result == "1" else "N"
  example["autoais"] = inference

  if torch.cuda.is_available():
    del input_ids, outputs
    torch.cuda.empty_cache()
  return inference


if __name__ == "__main__":
    text = "The captial of France is  'Paris'."
    question = 'What is the capital of France?'
    passage="""
    Paris (French pronunciation: [paʁi] ⓘ) is the capital and largest city of France. 
    With an official estimated population of 2,102,650 residents as of 1 January 2023 
    in an area of more than 105 km2 (41 sq mi),[5] Paris is the fourth-largest city in 
    the European Union and the 30th most densely populated city in the world in 2022.
    Since the 17th century, Paris has been one of the world's major centres of finance, 
    diplomacy, commerce, culture, fashion, and gastronomy. For its leading role in the
    arts and sciences, as well as its early and extensive system of street lighting, 
    in the 19th century, it became known as the City of Light.
    """

    score = AutoAIS([question], [text], [passage])
    print(score)


    



    
    
