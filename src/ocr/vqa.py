import os
import argparse
import fitz  # PyMuPDF
from PIL import Image
import torch
import cv2
from transformers import AutoModel, AutoTokenizer
import pdf2image
from td import TableDetector
import matplotlib.pyplot as plt
import json

def pdf_to_images(input_pdf):
    # Open the input PDF
    images = pdf2image.convert_from_path(input_pdf)
    
    return images

def mask_keywords(input_pdf, output_pdf, keywords_to_mask):
    # Open the input PDF
    doc = fitz.open(input_pdf)
    
    # Define the font to be used
    fontname = "helv"  # Helvetica
    fontsize = 9  # Default font size
    
    # Iterate through all the pages
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        
        # Extract the text as a dictionary
        text_dict = page.get_text("dict")
        
        # Iterate through blocks of text in the page
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        original_text = span["text"]
                        new_text = original_text
                        for keyword in keywords_to_mask:
                            if keyword in original_text:
                                # Replace the keyword with 'XYZ'
                                new_text = new_text.replace(keyword, "XYZ")
                        
                        if new_text != original_text:
                            # Calculate the position to insert the new text
                            bbox = span["bbox"]
                            # print("Bounding_BOX: ", bbox, end="\n\n")
                            x0, y0 = bbox[0], bbox[1]  # Top-left corner of the text
                            # Erase the original text by drawing a white rectangle over it
                            page.draw_rect(bbox, color=(1, 1, 1), fill=(1, 1, 1))
                            y_mid = (bbox[1] + bbox[3]) / 2
                            # Insert the new text
                            page.insert_text((x0, y_mid), new_text, fontname=fontname, fontsize=fontsize, color=(0, 0, 0))
    
    # Save the output PDF
    doc.save(output_pdf)
    print(f"Masked PDF saved as: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Mask specific keywords in a PDF file")
    parser.add_argument("-p", "--pdf", type=str, required=True, help="Path to input PDF file")

    args = parser.parse_args()

    # Convert PDF to images
    input_pdf = args.pdf
    images = pdf_to_images(input_pdf)
    first_page = images[0]

    first_page.save("first_page.png")
    # exit()
    image = cv2.imread("first_page.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Table detection
    table_det = TableDetector()   
    dets = table_det.predict(image=image)
    if dets ==[]:
        return
    det = dets[0]
    x1, y1, x2, y2 = map(int, det)  # Convert coordinates to integers
    cropped_img = image[y1:y2, x1:x2]  # Crop the image using the bounding box
    plt.imsave("cropped_img_new.jpg", cropped_img)
    
    # Load the model and tokenizer
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, device_map="cuda:4")
    model.eval()

    # Define the questions and prompts
    question = "Extract the information from the given tabular image in json format (key-value pairs)?"
    system_prompt = """You are a highly accurate and detail-oriented document reader specializing in extracting patient information from medical documents.
    Your primary task is to identify and extract complete text provided image. If the image contains tables, parse tables in json key-value pair format.
    Please strictly follow these instructions and return the information in a json text format.
    """
    # Perform inference

    msgs = [{'role': 'user', 'content': question}]
    
    res = model.chat(
          image=Image.fromarray(cropped_img),
          msgs=msgs,
          tokenizer=tokenizer,
          sampling=False,  # if sampling=False, beam_search will be used by default
          temperature=0.3,
          num_beams=8,
          system_prompt=system_prompt
    )

    print(res)

    try:
        data = json.loads(res)
        with open('output.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except ValueError:
        print("Invalid JSON")
        return
    # print("Words to be masked -> :")
    # words_to_mask = []
    # # print(res)
    # print("\n\n")
    # res_split = res.split(',')

    # for j in range(len(res_split)):
    #     for word in res_split[j].split():
    #         words_to_mask.append(word.strip())
    # print(words_to_mask)

    # # keywords_to_mask = res_split
    # # keywords_to_mask = [x.strip() for x in keywords_to_mask]
    # # print(keywords_to_mask)
    
    
    # output_pdf = os.path.join(os.path.dirname(input_pdf), f'{os.path.basename(input_pdf).split(".")[0]}_masked.pdf')
    # mask_keywords(input_pdf, output_pdf, words_to_mask)

if __name__ == "__main__":
    main()