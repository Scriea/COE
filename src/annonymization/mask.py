import os
import argparse
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import pdf2image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pdf_to_images(input_pdf):
    # Convert each page of the input PDF to an image
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

def process_pdf_file(input_pdf, output_dir, model, tokenizer, system_prompt):
    # Convert PDF to images
    images = pdf_to_images(input_pdf)
    first_page = images[0]

    # first_page.save("first_page.png")

    # Perform inference
    msgs = [{'role': 'user', 'content': "What is the full name of the patient, hospital number and the mobile number as visible in the image?"}]
    
    res = model.chat(
          image=first_page,
          msgs=msgs,
          tokenizer=tokenizer,
          sampling=False,  # if sampling=False, beam_search will be used by default
          temperature=0.3,
          num_beams=4,
          system_prompt=system_prompt
    )

    print("Words to be masked -> :")
    words_to_mask = []
    # print(res)
    print("\n\n")
    res_split = res.split(',')

    for j in range(len(res_split)):
        for word in res_split[j].split():
            words_to_mask.append(word.strip())
    print(words_to_mask)
    
    output_pdf = os.path.join(output_dir, f'{os.path.basename(input_pdf).split(".")[0]}_masked.pdf')
    mask_keywords(input_pdf, output_pdf, words_to_mask)

def main():
    parser = argparse.ArgumentParser(description="Mask specific keywords in PDF files within a directory")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to the directory containing PDF files")
    parser.add_argument("-o", "--output_dir", type=str,default="./processed", help="Path to the directory to save processed PDF files")

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model and tokenizer
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    model.eval()

    # Define the system prompt
    system_prompt = """You are a highly accurate and detail-oriented document reader specializing in extracting patient information from medical documents.
    Your primary task is to identify and extract the complete Patient Name and Hospital Number, Mobile Number from the provided image.
    1. Ensure you capture the full Patient Name, including the first, middle (if present), and last names without any changes in spelling or case.
    2. Ensure you accurately extract the Hospital Number without any alterations.
    3. Ensure you take the 10 digit mobile no. from pdf
    
    Please strictly follow these instructions and return the information in a comma-separated format: Patient Name, Hospital Number, Mobile Number
    """

    # Process each PDF file in the specified directory
    for filename in os.listdir(args.directory):
        if filename.lower().endswith(".pdf"):
            input_pdf = os.path.join(args.directory, filename)
            process_pdf_file(input_pdf, args.output_dir, model, tokenizer, system_prompt)

if __name__ == "__main__":
    main()