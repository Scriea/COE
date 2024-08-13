from pdfminer.high_level import extract_text

def extract_pdf_text(pdf_path):
    with open(pdf_path, 'r') as f:
        raw_text = extract_text(f)
    return raw_text.strip()


if __name__ == "__main__":
    text = extract_text("/raid/ganesh/vishak/ashutosh/COE/data/Patient_1_Discharge summary_Final.pdf")

    print(text)