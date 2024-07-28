import os
import re
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer

AUTOAIS = "google/t5_xxl_true_nli_mixture"
PASSAGE_FORMAT = re.compile("« ([^»]*) » « ([^»]*) » (.*)")
device="cuda:7" if torch.cuda.is_available() else "cpu"

def AutoAIS(text, attribution):
    """
    Calculate the AutoAIS score for the given text and attribution.

    Args:
        text (str): The input text.
        attribution (np.ndarray): The attribution of each token in the text.

    Returns:
        float: The AutoAIS score.
    """
    pass



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


def infer_autoais(example, tokenizer, model):
  """Runs inference for assessing AIS between a premise and hypothesis.

  Args:
    example: Dict with the example data.
    tokenizer: A huggingface tokenizer object.
    model: A huggingface model object.

  Returns:
    A string representing the model prediction.
  """
  input_text = format_example_for_autoais(example)
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids
  outputs = model.generate(input_ids)
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  inference = "Y" if result == "1" else "N"
  example["autoais"] = inference
  return inference


def score_predictions(predictions, nq_answers):
  """Scores model predictions against AutoAIS and NQ answers.

  Args:
    predictions: A dict from questions to prediction rows.
    nq_answers: A dict from questions to lists of NQ reference answers.
    passages: A dict from identifiers from the attribution corpus to the
      corresponding paragraphs.

  Returns:
    a dict of metric values, keyed by metric names
  """
  hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS)
  hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS)

  autoais = 0
  target_answers = []
  predicted_answers = []
  for question, answers in nq_answers.items():
    target_answers.append(answers)
    example = predictions.get(question, None)
    if example is None:
      print("Did not find prediction for '%s'", question)
      predicted_answers.append("")
      continue
    predicted_answers.append(example["answer"])
    if not example["passage"]:
      continue
    inference = infer_autoais(example, hf_tokenizer, hf_model)
    autoais += inference == "Y"

  scores = {}
  scores["AutoAIS"] = autoais / len(target_answers)

#   for metric, score in squad(target_answers, predicted_answers).items():
#     scores[f"SQuAD ({metric})"] = score
  return scores


if __name__ == "__main__":
    text = "The answer to the question 'What is the capital of France?' is 'Paris'."
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
    
    