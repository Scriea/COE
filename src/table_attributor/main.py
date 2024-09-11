import cv2
from fuzzywuzzy import fuzz

from .doc_handler import get_document_map
from difflib import SequenceMatcher


def similarity(a, b):
    return fuzz.partial_ratio(a.lower(), b.lower())
    # SequenceMatcher(None, a.lower(), b.lower()).ratio())


def get_attributed_image(pdf_path, answer):
    input_path = "documents/"
    output_path = "resources/"
    images_folder = output_path + pdf_path[:-4] + "/images/"
    doc_map = get_document_map(pdf_path, input_path, output_path, images_folder)
    max_simi = -1
    final_page = ""
    # final_text = ""
    final_bbox = [0, 0, 0, 0]

    for page in doc_map.keys():
        for text in doc_map[page].keys():
            simi = similarity(answer, text)
            if simi > max_simi:
                max_simi = simi
                final_page = page
                # final_text = text
                final_bbox = doc_map[page][text]

    actual_page_to_render = images_folder + final_page
    final_image = cv2.imread(actual_page_to_render)
    cv2.rectangle(
        final_image,
        (final_bbox[0], final_bbox[1]),
        (final_bbox[2], final_bbox[3]),
        (0, 255, 128),
        4,
    )
    cv2.imwrite("temp.jpg", final_image)
    return final_image


# pdf_path = 'test.pdf'
# ### Here replace with LLM answer
# answer = "Dhanvantarinagar, Puducherry-605006"
#
# get_attributed_image(pdf_path, answer)
