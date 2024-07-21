import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Arguments:
    def __init__(self):
        self.llama2chat = "meta-llama/Llama-2-7b-chat-hf"           ## Default
        self.llam

args = Arguments()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator:
    def __init__(self, model_name=args.llama2chat, device=device):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

if __name__=="__main__":
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llama2chat)
    model = AutoModelForCausalLM.from_pretrained(args.llama2chat).to(device)
    tokenizer.pad_token = tokenizer.eos_token

    def generate_response(question):
        input_text = f"User: {question}\nAI:"
        
        # Tokenize the input and create attention mask
        inputs = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate a response
        with torch.no_grad():
            output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=1024, pad_token_id=tokenizer.pad_token_id)

        # Decode the response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the actual response text
        response_text = response.split("AI:")[-1].strip()

        return response_text

    # Example usage
    question = "What is the capital of France?"
    response = generate_response(question)

    print(f"Question: {question}")
    print(f"Answer: {response}")
