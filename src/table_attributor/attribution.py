import pytesseract
from bs4 import BeautifulSoup

def get_page_attribute_map(image):
    #image = cv2.imread(image_file)
    hocr = pytesseract.image_to_pdf_or_hocr(image, lang = 'eng', extension='hocr')
    soup = BeautifulSoup(hocr, 'html.parser')
    paras = soup.find_all('p')
    attribution_map = {}
    
    for p in paras:
        p_bbox = p['title'].split(' ')[1:]
        final_box = list(map(int, p_bbox))
        final_text = ""

        p_spans = p.find_all('span', attrs = {'class': 'ocrx_word'})
        for word in p_spans:
            append = word.text + " "
            final_text += append

        attribution_map[final_text] = final_box
    return attribution_map