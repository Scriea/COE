import pdf2image

def pdf_to_images(input_pdf):
    # Open the input PDF
    images = pdf2image.convert_from_path(input_pdf)
    
    return images