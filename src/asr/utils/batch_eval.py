import os
import jiwer
import xml.etree.ElementTree as ET
import pandas as pd

def extract_text_from_xml(xml_path):
    # Parse the XML file and extract text
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Combine text from all relevant tags (you may need to adjust this based on your XML structure)
    text_elements = [elem.text for elem in root.iter() if elem.text]
    return ' '.join(text_elements).strip()

def add_spaces_to_text(text):
    # Add a space between each character
    return ' '.join(text)

def calculate_wer(groundtruth_text, hyp_text):
    # Calculate WER
    return jiwer.wer(groundtruth_text, hyp_text)

def calculate_cer_using_wer(groundtruth_text, hyp_text):
    # Add spaces between characters
    spaced_groundtruth = add_spaces_to_text(groundtruth_text)
    spaced_hyp = add_spaces_to_text(hyp_text)
    
    # Calculate CER using WER
    cer = calculate_wer(spaced_groundtruth, spaced_hyp)
    
    return cer

def process_text_files(groundtruth_dir, hyp_dir, output_excel_path):
    results = []

    # Iterate over each hypothesis file in the hypothesis directory
    for hyp_file in os.listdir(hyp_dir):
        if hyp_file.endswith('_merged.xml'):
            # Construct the corresponding ground truth file name
            base_name = hyp_file.split('_merged.xml')[0]
            groundtruth_file = base_name + '_w2v.xml'
            groundtruth_path = os.path.join(groundtruth_dir, groundtruth_file)
            hyp_path = os.path.join(hyp_dir, hyp_file)
            
            if os.path.exists(groundtruth_path):
                # Extract text from XML files
                groundtruth_text = extract_text_from_xml(groundtruth_path)
                hyp_text = extract_text_from_xml(hyp_path)
                
                # Calculate WER and CER (using WER on spaced texts)
                wer = calculate_wer(groundtruth_text, hyp_text)
                cer = calculate_cer_using_wer(groundtruth_text, hyp_text)
                
                # Append results
                results.append({
                    'Ground Truth File': groundtruth_file,
                    'Hypothesis File': hyp_file,
                    'Word Error Rate (WER)': wer,
                    'Character Error Rate (CER)': cer
                })
                
                print(f"Ground Truth: {groundtruth_file}, Hypothesis: {hyp_file}")
                print(f"Word Error Rate (WER): {wer}")
                print(f"Character Error Rate (CER): {cer}")
                print("-" * 60)
            else:
                print(f"Ground truth file {groundtruth_file} does not exist.")
    
    # Create a DataFrame and save to an Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"Results have been saved to {output_excel_path}")

# Paths to your ground truth and hypothesis directories
groundtruth_dir = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/w2v_gt'
hyp_dir = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/transcripts/wav2vec2/hyp_audio/hindi'
output_excel_path = 'wer_cer_results.xlsx'

process_text_files(groundtruth_dir, hyp_dir, output_excel_path)
