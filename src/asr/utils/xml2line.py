import os
import xml.etree.ElementTree as ET

def extract_paragraph_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    words = []

    for line in root.findall('line'):
        words.extend([word.text for word in line.findall('word') if word.get('is_valid') == '1'])
    
    paragraph = ' '.join(words)
    return paragraph

def write_paragraph_to_txt(paragraph, txt_file):
    with open(txt_file, 'a') as f:
        f.write(paragraph + '\n')

def process_xml_folder(xml_folder, output_txt):
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(xml_folder, filename)
            paragraph = extract_paragraph_from_xml(xml_file_path)
            write_paragraph_to_txt(paragraph, output_txt)

# Example usage
xml_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/transcripts/wav2vec2/hyp_audio/hindi'
output_txt = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/hindi_whisper.txt'

# Clear the content of the output file before appending new content
open(output_txt, 'w').close()

process_xml_folder(xml_folder, output_txt)
