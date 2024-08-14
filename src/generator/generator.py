import os
import torch
from .prompts import medical_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from llama_index.core import Document
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex


class Arguments:
    def __init__(self):
        self.llama2chat = "meta-llama/Llama-2-7b-chat-hf"  # Default
        # self.llama3instruct = "meta-llama/Meta-Llama-3-8B-Instruct"


class Generator:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        if device is None:
            device = torch.device(
                'cuda:' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, device=self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, quantization_config=self.quantization_config, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.query_engine = None

    def generate_response(self, input_text):
        "Sequential generation of response"
        if isinstance(input_text, list):
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                input_text, return_tensors="pt", padding="longest", return_attention_mask=True)
            tokenized_inputs = tokenized_inputs.to(self.device)
            N = tokenized_inputs['input_ids'].shape[1]

            outputs = self.model.generate(
                **tokenized_inputs,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
                # early_stopping=True,
                # num_beams=8,
                do_sample=True,
                temperature=0.9,
            )
            predicted_token_ids = outputs['sequences']
            answers = self.tokenizer.batch_decode(
                predicted_token_ids[:, N:], skip_special_tokens=True)
            return answers

        elif isinstance(input_text, str):
            inputs = self.tokenizer(
                input_text, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            N = inputs['input_ids'].shape[1]
            # Generate a response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.4,
                    # num_beams=5,
                    do_sample=True,
                    max_new_tokens=512)

            # Decode the response
            response_text = self.tokenizer.decode(
                output_ids[0, N:], skip_special_tokens=True)
            return response_text

        else:
            raise ValueError(
                "Input text should be a string or a list of strings")

    def vector_compare(self, input_text):
        if isinstance(input_text, list):
            answers = []
            query_engine = self.store_docs()
            for string in input_text:
                response_text = query_engine.query(string)
                answers.append(response_text)
            return answers

        elif isinstance(input_text, str):
            query_engine = self.store_docs()

            response_text = query_engine.query(input_text)
            return response_text

    def store_docs(self):
        text = """
        ADVICE @ DISCHARGE :
        1. Regular medications & STOP SMOKING.
        2. Avoid Alcohol, Heavy exertion and lifting weights.
        3. Diet - High fiber, low cholesterol, low sugar (no sugar if diabetic), fruits, vegetables (5 servings
        per day).
        4. Exercise - Walk at least for 30 minutes daily. Avoid if Chest pain.
        5. TARGETS * LDL<70mg/dl *BP - 120/80mmHg * Sugar Fasting - 100mg/dl Post Breakfast – 150mg/dl
        * BMI<25kg/m2.
        6. IF CHEST PAIN – T.ANGISED 0.6 mg or T.SORBITRATE 5 mg keep under tongue. Repeat if no relief
        @ 5 minutes and report to nearest doctor for urgent ECG.
        """

        llm = HuggingFaceLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            query_wrapper_prompt=PromptTemplate(
                "<s> [INST] You are an experienced doctor and give honest assistant to user when a query is asked. Answer the question based on the context below. Keep the answer short. Respond 'Unsure about answer' if not sure about the answer. \t\t\t{query_str} [/INST] "),
            context_window=3900,
            model_kwargs={"quantization_config": self.quantization_config}
        )
        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
        documents = [Document(text=text)]
        vector_index = VectorStoreIndex.from_documents(documents)
        query_engine = vector_index.as_query_engine(response_mode="compact")
        return query_engine


if __name__ == "__main__":

    args = Arguments()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Usage example

    generator = Generator(model_path=args.llama2chat, device=device)

    question = "I am feeling chest pain, what should I do?"
    context = """
        ADVICE @ DISCHARGE :
        1. Regular medications & STOP SMOKING.
        2. Avoid Alcohol, Heavy exertion and lifting weights.
        3. Diet - High fiber, low cholesterol, low sugar (no sugar if diabetic), fruits, vegetables (5 servings
        per day).
        4. Exercise - Walk at least for 30 minutes daily. Avoid if Chest pain.
        5. TARGETS * LDL<70mg/dl *BP - 120/80mmHg * Sugar Fasting - 100mg/dl Post Breakfast – 150mg/dl
        * BMI<25kg/m2.
        6. IF CHEST PAIN – T.ANGISED 0.6 mg or T.SORBITRATE 5 mg keep under tongue. Repeat if no relief
        @ 5 minutes and report to nearest doctor for urgent ECG.
    """

    input_text = medical_prompt.format(context, question)

    response = generator.generate_response(input_text)
    response_vector = generator.vector_compare(input_text)

    print(f"Input Text: {input_text}")
    print(f"Answer: {response}")
    print(f"Vector RAG Answer: {response_vector}")
