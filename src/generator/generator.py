import os
import torch
from .prompts import medical_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

class Arguments:
    def __init__(self):
        self.llama2chat = "meta-llama/Llama-2-7b-chat-hf"           ## Default
        # self.llama3instruct = "meta-llama/Meta-Llama-3-8B-Instruct"


class Generator:
    def __init__(self, model_name, device=torch.device('cuda')):
        self.model_path = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self,input_text):
        # Tokenize the input and create attention mask
        inputs = self.tokenizer.encode_plus(input_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate a response
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0.9,
                do_sample=True)

        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the actual response text
        response_text = response[len(input_text):].strip()

        return response_text

if __name__=="__main__":

    args = Arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Usage example
    
    generator = Generator(model_name=args.llama2chat, device=device)
        
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

    print(f"Input Text: {input_text}")
    print(f"Answer: {response}")