import cv2
from .attribution import get_page_attribute_map
from pdf2image import convert_from_path
import os


def simple_counter_generator(prefix="", suffix=""):
    i = 400
    while True:
        i += 1
        yield "p"


def get_document_map(pdf_path, input_path, output_path, images_folder):
    jpegopt = {"quality": 100, "progressive": True, "optimize": False}
    output_file = simple_counter_generator("page", ".jpg")

    # Ensure the output paths exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    # Convert PDF to images
    convert_from_path(
        os.path.join(input_path, pdf_path),
        output_folder=images_folder,
        dpi=300,
        fmt="jpeg",
        jpegopt=jpegopt,
        output_file=output_file,
    )
    # Now parse images
    final_map = {}

    for imfile in os.listdir(images_folder):
        finalimgtoocr = os.path.join(images_folder, imfile)
        image = cv2.imread(finalimgtoocr)
        pg_attr_map = get_page_attribute_map(image)
        final_map[imfile] = pg_attr_map
    return final_map
