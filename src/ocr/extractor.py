import os
import fitz  # PyMuPDF
from PIL import Image
import torch
import cv2
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import json

from .td import TableDetector
from .util import pdf_to_images, split_text_by_keywords


class DocumentReader:
    def __init__(self, device=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def extract_passages(self, input_pdf, phrases=None, output_path=None):
        text = self.extract_text(input_pdf)
        table = self.extract_table(input_pdf)
        if not phrases:
            phrases = [
                "Discharge Summary",
                "HISTORY",
                "RISK FACTORS",
                "CLINICAL FINDINGS",
                "ADMISSION DIAGNOSIS",
                "PREV. INTERVENTION",
                "PROCEDURES DETAILS",
                "RESULT :",
                "IN HOSPITAL COURSE :",
                "FINAL DIAGNOSIS",
                "CONDITION AT DISCHARGE",
                "ADVICE @ DISCHARGE",
                "DIET ADVICE",
                "DISCHARGE MEDICATIONS",
            ]
        passages = split_text_by_keywords(text, phrases)
        try:
            passages[1] = table
        except IndexError as e:
            print(f"Upload correct discharge summary")
        ## Save the extracted passages to a JSON file with key = "Dicharge Summary" and value as list of passages
        json_data = {"EHR": passages}
        if output_path:
            if output_path.endswith(".json"):
                with open(os.path.join(output_path), "w") as f:
                    json.dump(json_data, f, indent=4)
        #     else:
        #         raise ValueError("Output path should be a JSON file.")
        return passages

    def extract_text(self, input_pdf):
        # Open the input PDF
        doc = fitz.open(input_pdf)
        text = ""
        # Iterate through all the pages
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text

    def extract_table(self, input_pdf):

        # Convert PDF to images
        images = pdf_to_images(input_pdf)
        first_page = images[0]

        first_page.save("first_page.png")
        image = cv2.imread("first_page.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## Table detection
        table_det = TableDetector()
        dets = table_det.predict(image=image)
        if dets == []:
            return ""
        det = dets[0]
        x1, y1, x2, y2 = map(int, det)  # Convert coordinates to integers
        cropped_img = image[y1:y2, x1:x2]  # Crop the image using the bounding box
        # plt.imsave("cropped_img_new.jpg", cropped_img)

        # Load the model and tokenizer
        model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-Llama3-V-2_5-int4", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-Llama3-V-2_5-int4",
            trust_remote_code=True,
            device_map=self.device,
        )
        model.eval()

        # Define the questions and prompts
        question = "Extract the information from the given tabular image in json format (key-value pairs)?"
        system_prompt = """You are a highly accurate and detail-oriented document reader specializing in extracting patient information from medical documents.
        Your primary task is to identify and extract complete text provided image. If the image contains tables, parse tables in json key-value pair format.
        Please strictly follow these instructions and return the information in a json text format.
        """
        # Perform inference

        msgs = [{"role": "user", "content": question}]

        res = model.chat(
            image=Image.fromarray(cropped_img),
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=False,  # if sampling=False, beam_search will be used by default
            temperature=0.3,
            num_beams=8,
            system_prompt=system_prompt,
        )

        # print(res)

        # try:
        #     data = json.loads(res)
        #     with open('output.json', 'w') as json_file:
        #         json.dump(data, json_file, indent=4)
        # except ValueError:
        #     print("Invalid JSON")
        #     return None

        return res


if __name__ == "__main__":
    dr = DocumentReader(device="cuda:4")
    out = dr.extract_text("/home/iitb_admin_user/COE/Anonymized_EHR/EHR_masked2.pdf")
    # print(out)
    phrases = [
        "Discharge Summary",
        "HISTORY",
        "RISK FACTORS",
        "CLINICAL FINDINGS",
        "ADMISSION DIAGNOSIS",
        "PREV. INTERVENTION",
        "PROCEDURES DETAILS",
        "RESULT :",
        "IN HOSPITAL COURSE :",
        "FINAL DIAGNOSIS",
        "CONDITION AT DISCHARGE",
        "ADVICE @ DISCHARGE",
        "DIET ADVICE",
        "DISCHARGE MEDICATIONS",
    ]
    passages = dr.extract_passages(
        "/home/iitb_admin_user/COE/Anonymized_EHR/EHR_masked2.pdf"
    )

    print(len(passages))
    for passage in passages[1:]:
        print(f'"""\n{passage}\n"""\n\n\n\n\n\n\n\n\n\n\n\n')

    json_data = json.dumps(passages[1:], indent=4)
    with open("passages.json", "w") as json_file:
        json_file.write(json_data)
