
import os
import json
import argparse
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

parser = argparse.ArgumentParser(description="Run OCR on a PDF file")
parser.add_argument("-p","--pdf", type=str, help="Path to PDF file", default="")

args = parser.parse_args()

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_pdf(args.pdf)
result = model(doc)
json_output = result.export()
# print(json_output)


blocks = []
lines = []
data = json_output
for page in data['pages']:
    for block in page['blocks']:
        blocks.append([])
        for line in block['lines']:
            lines.append([])
            for word in line['words']:
                blocks[-1].append(word['value'])
                lines[-1].append(word['value'])
        # blocks.append(block)

# for block in blocks:
#   print(block)
print("===============================================================")
# for line in lines:
#   print(line)


output_string = "\n".join(" ".join(sublist) for sublist in lines)

print(output_string)
