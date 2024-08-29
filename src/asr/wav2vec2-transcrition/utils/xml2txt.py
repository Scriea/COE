import os
import xml.etree.ElementTree as ET

def extract_text_from_xml(xml_path, output_txt_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract words in order
    words = []
    for line in root.findall('.//line'):
        for word in line.findall('.//word'):
            words.append(word.text)
    
    # Join words into a single paragraph
    paragraph = ' '.join(words)
    
    # Save the paragraph to the output text file
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write(paragraph)

def process_xml_directory(xml_dir, output_txt_dir):
    # Ensure the output directory exists
    os.makedirs(output_txt_dir, exist_ok=True)
    
    # Iterate over each XML file in the input directory
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            output_txt_path = os.path.join(output_txt_dir, os.path.splitext(xml_file)[0] + '.txt')
            extract_text_from_xml(xml_path, output_txt_path)
            print(f"Extracted text saved to {output_txt_path}")

# Paths to your XML input directory and output text directory
xml_dir = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/w2v_gt'
output_txt_dir = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/w2v_gt_txt'

process_xml_directory(xml_dir, output_txt_dir)
