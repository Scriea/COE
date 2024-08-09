import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from typing import List

import spacy
import scispacy
from scipy.stats import kstest
import numpy as np
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM, 
        pipeline, 
        AutoModelWithLMHead
    )

import torch
import pickle
from tqdm import tqdm
import stanza
import re
import argparse
import random
from datasets import load_dataset
from utils import compute_f1, softmax, find_subset_indices, extract_text_between_double_quotes

ALOE_8B = "HPAI-BSC/Llama3-Aloe-8B-Alpha"
BIO_MISTRAL_7B = "BioMistral/BioMistral-7B"

class HallucinationDetection:
    def __init__(self,):
        pass
    
    def hallucination_prop(self):
        """
        Returns sentence level hallucination detection results

        Input:
            Text (str): Input text for hallucination detection
            Other args might depend on the implementation
        Output:
            List of floats for each sentence in the input text

        """
        raise NotImplementedError
    

class HalluCheck(HallucinationDetection):
    def __init__(self, device=None, method="POS", model_path="HPAI-BSC/Llama3-Aloe-8B-Alpha"):
        self.method = method.upper()
        if self.method=="NER":
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
        elif self.method=="POS":
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
        elif self.method=="MED":
            self.nlp = spacy.load("en_ner_bc5cdr_md")

        if device is None:
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        self.tokenizer_ques_gen = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        self.model_ques_gen = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap", device_map=self.device)

        self.qa_model = pipeline("question-answering", device=self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding = True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            output_attentions=True, 
            device_map=self.device
        )

        self.model.to(self.device)
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.tokenizer.padding_side = "left"

    

    def hallucination_prop(self, text, context=""):
        sent_nlp = spacy.load("en_core_web_sm")
        sentences = [
            sent.text.strip() for sent in sent_nlp(text).sents
        ] 
        print("Sentences:", sentences)

        hallucination_probs = []
        for sentence in sentences:
            generated_question_answer_list = self.generate_questions_based_on_factual_parts(sentence=sentence)
            print("\n\nGenerated Questions", generated_question_answer_list)
            regenerated_answers, scores = self.generate_pinpointed_answers(generated_question_answer_list, context=context)
            print("\n\nRegenerated Answers", regenerated_answers)
            generated_questions = [generated_question_answer[0] for generated_question_answer in generated_question_answer_list]
            initial_hallu = self.compare_orig_and_regenerated(generated_questions, text, regenerated_answers)
            print("\n\nInitial Hallucination", initial_hallu)
            final_hallu = self.check_with_probability(regenerated_answers, initial_hallu[2], scores, initial_hallu[0])
            if final_hallu == []:
                prob_hallu = None
            else:
                prob_hallu = sum(final_hallu)/len(final_hallu)
            hallucination_probs.append(prob_hallu)
        return hallucination_probs

    def generate_questions_based_on_factual_parts(self, sentence:str)->List[List[str]]:
        """
        Description:
            This function generates questions based on the factual parts of the sentence.
        
        Args:
            sentence (str): The input sentence for which questions are to be generated.
        
        """
        def get_question(answer, context, max_length=128):
            input_text = "answer: %s  context: %s </s>" % (answer, context)
            features = self.tokenizer_ques_gen([input_text], return_tensors='pt')
            features = features.to(self.device)
            output = self.model_ques_gen.generate(input_ids=features['input_ids'], 
                        attention_mask=features['attention_mask'],
                        max_length=max_length)
            ques = self.tokenizer_ques_gen.decode(output[0])
            return ques
        
        if self.method == 'POS':
            double_quote_words = extract_text_between_double_quotes(sentence)
            text = sentence
            try:
                for i, double_quote_word in zip(range(len(double_quote_words)), double_quote_words):
                    # print(double_quote_word)
                    text = text.replace('"{}"'.format(double_quote_word), "DOUBLEQUOTES" + str(i))
            except:
                pass
            doc = self.nlp(text)
            is_factual = []
            split_text = []
            for sent in doc.sentences:
                for word in sent.words:
                    split_text.append(word.text)
                    if word.xpos == "NNP" or word.xpos == "NNPS" or word.xpos == "CD" or word.xpos == "RB":
                        # or word.xpos == "JJ" or word.xpos == "JJR" or word.xpos == "JJS"
                        is_factual.append(1)
                    elif word.upos == "PUNCT":
                        is_factual.append(2)
                    elif word.xpos == "IN":
                        is_factual.append(3)
                    else: is_factual.append(0)
            i = 0
            atomic_facts = []
            while (i < len(is_factual)):
                s = ""
                while i < len(is_factual) and (is_factual[i] ==1 or (is_factual[i] == 2 and is_factual[i-1]!=0  and i < (len(is_factual) - 1) and is_factual[i+1] !=0) or (is_factual[i] == 3 and is_factual[i-1]!=0  and i < (len(is_factual) - 1) and is_factual[i+1] !=0)):
                    s += split_text[i] + " "
                    i +=1
                if s != "":
                    atomic_facts.append(s)
                i += 1
            atomic_facts = [fact[:-1] for fact in atomic_facts]
            # print(atomic_facts)
            output_list = []
            for element in atomic_facts:
                if "DOUBLEQUOTES" in element:
                    # Extract the integer after "DOUBLEQUOTES"
                    index = int(element.split("DOUBLEQUOTES")[1].strip())
                    
                    # Replace with the corresponding element from double_quote_words
                    output_list.append(double_quote_words[index])
                else:
                    output_list.append(element)
        elif self.method == 'NER':
            ner_sent = self.nlp(sentence)
            output_list = [ent.text for sent in ner_sent.sentences for ent in sent.ents]
        elif self.method == 'RANDOM':
            ner_sent = self.nlp(sentence)
            num_random_facts = len([ent.text for sent in ner_sent.sentences for ent in sent.ents])
            random_facts_indices = random.sample(range(0, len(sentence.split())), num_random_facts)
            output_list = [sentence.split()[index] for index in random_facts_indices]
        elif self.method=="MED":
            ner_sent = self.nlp(sentence)
            output_list = [ent.text for ent in ner_sent.ents]
            ## Make Unique
            output_list = list(set(ent.lower() for ent in output_list))

        questions_answer_list = []
        pattern = r'<pad> question: (.+?)</s>'
        print("Atomic facts", output_list)


        for atomic_fact in output_list:
            gen_ques = get_question(atomic_fact, sentence)
            gen_ques = re.search(pattern, gen_ques).group(1)
            questions_answer_list.append([gen_ques, atomic_fact])
        return questions_answer_list
    

    def generate_pinpointed_answers(self, generated_question_answer_list, context):
        ## try with chat template
        # prompt = [f"{question_answer[0]}" for question_answer in generated_question_answer_list]
        if generated_question_answer_list == []:
            return [], []
        prompt = [f"<s>[INST]Background Information:{context} Question: {question_answer[0]}\n Answer with reasoning: [/INST]" for question_answer in generated_question_answer_list]
        tokenized_inputs = self.tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding = "longest", return_attention_mask = True)
        tokenized_inputs = tokenized_inputs.to(self.device)
        N=tokenized_inputs['input_ids'].shape[1]

        outputs = self.model.generate(
            **tokenized_inputs, 
            return_dict_in_generate=True, 
            output_scores=True, 
            max_new_tokens = 32, 
            # early_stopping=True,
            # num_beams=8,
            )
        
        predicted_token_ids = outputs['sequences']
        answers = self.tokenizer.batch_decode(predicted_token_ids[:, N:], skip_special_tokens=True)
        # print(answers)
        return answers, outputs['scores']
    
    def compare_orig_and_regenerated(self, generated_questions, orig_answer, reg_answers):
        not_match = []
        #not match is 1 if not matching [hallucinated] otherwise 0
        pin_point_orig_answers = []
        pin_point_reg_answers = []
        for question, reg_answer in zip(generated_questions, reg_answers):
            pin_orig_answer = self.qa_model(question = question, context = orig_answer)["answer"]
            pin_reg_answer = self.qa_model(question = question, context = reg_answer)["answer"]
            # f1_score = calculate_f1_score(pin_orig_answer, pin_reg_answer)
            f1_score = compute_f1(pin_orig_answer, pin_reg_answer)
            # print(pin_orig_answer, pin_reg_answer, f1_score)
            # print(f1_score)
            not_match.append(int(f1_score < 0.6))
            pin_point_orig_answers.append(pin_orig_answer)
            pin_point_reg_answers.append(pin_reg_answer)
        return not_match, pin_point_orig_answers, pin_point_reg_answers
    
    def check_with_probability(self, reg_answers, pin_point_orig_answers, scores, in_hallu):
        # print(pin_point_orig_answers, len(pin_point_orig_answers))
        not_match = [1]*len(pin_point_orig_answers)
        score = tuple(t.cpu() for t in scores)
        for j, orig_answer, pin_point_answer in zip(range(len(pin_point_orig_answers)), reg_answers, pin_point_orig_answers):
            if in_hallu[j] == 1:
                continue
            precise_answer_indices = find_subset_indices(orig_answer, pin_point_answer)
            precise_answer_tokens_ids = self.tokenizer(orig_answer[precise_answer_indices[0]: precise_answer_indices[0] + len(precise_answer_indices)])
            precise_answer_tokens = self.tokenizer.convert_ids_to_tokens(precise_answer_tokens_ids['input_ids'])
            # print(precise_answer_tokens)
            dist = []
            tokenized_words = []
            # print("len of score", len(score))
            for i in range(0, 50):
                try:
                    id  = torch.argmax(score[i][j])
                    probs = softmax(score[i][j].numpy())
                    probs_top = softmax(np.partition(score[i][j], -5)[-5:])
                    # ks_statistic, ks_p_value = kstest(probs, 'uniform', args=(probs.min(), probs.max()))
                    ks_statistic, ks_p_value = kstest(probs_top, 'uniform', args=(probs_top.min(), probs_top.max()))
                    tokenized_words.append(self.tokenizer.convert_ids_to_tokens(id.item()))
                    if ks_p_value > 0.05:
                        dist.append(1)
                        # print(tokenizer.convert_ids_to_tokens(id.item()), 1, "U")
                    else:
                        dist.append(0)
                        # print(tokenizer.convert_ids_to_tokens(id.item()), 0, "N-U")
                    
                except:
                    continue
            if len(precise_answer_tokens) > 1:
            # print(tokenized_words, precise_answer_tokens)
                indices = find_subset_indices(tokenized_words, precise_answer_tokens[1:])
                if len(indices) == 0:
                    not_match[j]  = 1
                    break
                indices = [indices[0]  - 1] + indices
                # print(indices)
            else: indices = find_subset_indices(tokenized_words, precise_answer_tokens)
            dist_concern = [dist[index] for index in indices]
            if sum(dist_concern) == 0:
                not_match[j] = 0
            else: not_match[j] = 1
        return not_match
    




class SelfCheckGPT(HallucinationDetection):
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None,
        device = None
    ):
        model = model if model is not None else ALOE_8B
        if device is None:
            device = torch.device("cpu")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map=device)
        # self.model.to(device)
        self.model.eval()

        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device {device}")

    def hallucination_prop(self, text:str, Passages):

        nlp = spacy.load("en_core_web_sm")
        sentences = [
            sent.text.strip() for sent in nlp(text).sents
        ]  # spacy sentence tokenization
        print("Sentences:", sentences)
        sent_scores = self.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=Passages  ,  # list of sampled passages
            verbose=True,  # whether to show a progress bar
        )
        return sent_scores
     
    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template


    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :param context: str -- context to be used in the prompt
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False, # hf's default for Llama2 is True
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]


if __name__=="__main__":
    ## Example Usage
    device="cuda:1"
    # HC = HalluCheck(device="cuda:2", method= "POS" )
    # hallucination_prop = HC.hallucination_prop("The capital of France is Mumbai.")
    # print("Probability of Hallucination : ", hallucination_prop)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct" ,
        # padding = True
        )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct" , 
            trust_remote_code=True, 
            output_attentions=True, 
            device_map=device
        )

    model.to(device)

    query = "What is the captial of France?"
    prompt = f"<s>[INST] Question: {query}\n Answer with reasoning: [/INST]"
    tokenized_inputs = tokenizer.batch_encode_plus(
        [prompt]*20,
        return_tensors="pt", 
        padding = "longest", 
        return_attention_mask = True
        )
    tokenized_inputs = tokenized_inputs.to(device)
    N=tokenized_inputs['input_ids'].shape[1]

    outputs = model.generate(
        **tokenized_inputs, 
        return_dict_in_generate=True, 
        output_scores=True, 
        max_new_tokens = 64, 
        # early_stopping=True,
        # num_beams=8,
        do_sample=True,
        temperature=0.9,
        )
    
    predicted_token_ids = outputs['sequences']
    answers = tokenizer.batch_decode(predicted_token_ids[:, N:], skip_special_tokens=True)
    text = """
        The capital of France is Paris.
        The capital of France is Mumbai.
    """
    SC = SelfCheckGPT(device="cuda:2")
    text = """
    Paris (French pronunciation: [paʁi] ⓘ) is the capital and largest city of France. 
    With an official estimated population of 2,102,650 residents in January 2023[2] in an area of more than 105 km2 (41 sq mi),[5] Paris is the fourth-largest city in the European Union and the 30th most densely populated city in the world in 2022.
    Since the 17th century, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy.
    For its leading role in the arts and sciences, as well as its early and extensive system of street lighting, in the 19th century, it became known as the City of Light.[7]
    The capital of France is Mumbai. The capital of France is Delhi. The capital of France is Paris
    """
    hallucination_prop = SC.hallucination_prop(text=text, Passages=answers, context="")
    print(hallucination_prop)