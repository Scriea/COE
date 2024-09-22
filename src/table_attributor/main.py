import os
import cv2
from fuzzywuzzy import fuzz

from .doc_handler import get_document_map
from difflib import SequenceMatcher

ROOT = os.path.abspath(
    os.path.join((os.path.dirname(os.path.relpath(__file__))), "../../")
)
TEMP_DIR = os.path.join(ROOT, "temp")


def similarity(a, b):
    return fuzz.partial_ratio(a.lower(), b.lower())
    # SequenceMatcher(None, a.lower(), b.lower()).ratio())


def get_attributed_image(pdf_path, answer):
    # TEMP_DIR = "/home/iitb_admin_user/ashutosh/COE/temp"
    input_path = os.path.join(TEMP_DIR, "documents")
    output_path = os.path.join(input_path, "output")
    pdf_base = os.path.basename(pdf_path)

    pdf_folder = os.path.join(output_path, pdf_base[:-4])
    images_folder = os.path.join(pdf_folder, "images")
    # print(input_path, output_path, pdf_base, pdf_folder, images_folder)
    # Ensure the directories exist
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    doc_map = get_document_map(pdf_path, input_path, output_path, images_folder)
    max_simi = -1
    final_page = ""
    final_bbox = [0, 0, 0, 0]

    for page in doc_map.keys():
        for text in doc_map[page].keys():
            if text == "":
                continue
            simi = similarity(answer, text)
            if simi > max_simi:
                max_simi = simi
                final_page = page
                final_bbox = doc_map[page][text]

    actual_page_to_render = os.path.join(images_folder, final_page)
    final_image = cv2.imread(actual_page_to_render)

    # Draw a rectangle on the identified area
    cv2.rectangle(
        final_image,
        (final_bbox[0], final_bbox[1]),
        (final_bbox[2], final_bbox[3]),
        (0, 255, 128),
        4,
    )
    return final_image


if __name__ == "__main__":
    pdf_path = "/home/iitb_admin_user/ashutosh/COE/data/EHR_P1.pdf"
    answer = "Dhanvantarinagar, Puducherry-605006"
    get_attributed_image(pdf_path, answer)
