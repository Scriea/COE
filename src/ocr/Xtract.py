import os
import json
import argparse
import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def split_text_into_passages(text, keywords):
    # Create a pattern that matches any of the keywords with an optional space before the colon
    pattern = '|'.join(re.escape(keyword).replace(r'\:', r'\s*:') for keyword in keywords)
    print(pattern)
    # Find all matches in the text
    matches = list(re.finditer(f'({pattern})', text))

    passages = []
    last_index = 0

    for match in matches:
        start, end = match.span()
        # Capture the text before the current match as a passage
        if last_index < start:
            passages.append(text[last_index:start].strip())
        # Capture the current match and text after it as a passage
        passages.append(text[start:end].strip())
        last_index = end

    # Add the remaining text as the last passage
    if last_index < len(text):
        passages.append(text[last_index:].strip())

    # Filter out any empty passages
    passages = [passage for passage in passages if passage]

    # Add keywords at the beginning of their corresponding passages
    for i, passage in enumerate(passages):
        for keyword in keywords:
            if passage.startswith(keyword):
                passages[i] = f"{keyword} {passage[len(keyword):].strip()}"

    return passages

def main():
    parser = argparse.ArgumentParser(description="Run OCR on a PDF file")
    parser.add_argument("-p", "--pdf", type=str, help="Path to PDF file", default="")
    args = parser.parse_args()

    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_pdf(args.pdf)
    result = model(doc)
    json_output = result.export()

    print(json_output['pages'][0].keys())


    print("\n\n"*10)

    print(json_output)

    # lines = []
    # for page in json_output['pages']:
    #     for block in page['blocks']:
    #         for line in block['lines']:
    #             lines.append(" ".join(word['value'] for word in line['words']))

    # text = "\n".join(lines)

    # keywords = [
    #     'Discharge Summary',
    #     'HISTORY',
    #     'RISK FACTORS',
    #     'Physical Exam',
    #     'Lab Findings',
    #     'Echocardiography',
    #     'ADMISSION DIAGNOSIS:',
    #     'PROCEDURES DETAILS :',
    #     'RESULT',
    #     'IN HOSTPITAL COURSE',
    #     'FINAL DIAGNOSIS',
    #     'CONDITION AT DISCHARGE :',
    #     'ADVICE @ DISCHARGE :',
    #     'DIET ADVICE :',
    #     'DISCHARGE MEDICATIONS :',
    #     'FOLLOW-UP :'
    # ]

    # print(f"OCR TEXT: {text}\n\n")

    # passages = split_text_into_passages(text, keywords)

    # print(len(keywords), len(passages))
    # print("\n\n")
    # for passage in passages:
    #     print(f"Passage: {passage}\n\n")

if __name__ == "__main__":
    main()
