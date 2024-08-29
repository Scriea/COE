# process_xmls.py
import os
import re
import xml.etree.ElementTree as ET

def sort_files(file_list):
    def extract_number(filename):
        match = re.search(r'_(\d+)_chunks_merged\.xml$', filename)
        return int(match.group(1)) if match else -1
    
    return sorted(file_list, key=extract_number)

def extract_text_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    words = []

    for line in root.findall('.//word[@is_valid="1"]'):
        words.append(line.text)
    
    return ' '.join(words)

def process_xml_folder(xml_folder, mode, output_txt):
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('_chunks_merged.xml')]
    sorted_xml_files = sort_files(xml_files)

    with open(output_txt, 'w') as f:
        for xml_file in sorted_xml_files:
            file_number = int(re.search(r'_(\d+)_chunks_merged\.xml$', xml_file).group(1))
            if (mode == 'odd' and file_number % 2 != 0) or (mode == 'even' and file_number % 2 == 0):
                xml_file_path = os.path.join(xml_folder, xml_file)
                text = extract_text_from_xml(xml_file_path)
                f.write(text + '\n')

# Example usage
xml_folder = r'/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/transcript_whisper_medium/telugu_2'
mode = 'odd'  # or 'even'
output_txt = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/transcript_whisper_medium/hyp/Telugu_2_Prompt_hyp.txt'

process_xml_folder(xml_folder, mode, output_txt)
