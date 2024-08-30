import re

def clean_text(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    # Remove all types of timestamps and join lines with spaces
    cleaned_lines = []
    timestamp_pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{1,3}\]'
    punctuation_pattern = r'[,.?!;:"\'()\-]'  # Add any other punctuation marks you want to remove
    for line in lines:
        cleaned_line = re.sub(timestamp_pattern, '', line).strip()
        cleaned_line = re.sub(punctuation_pattern, '', cleaned_line)
        cleaned_lines.append(cleaned_line)
    
    paragraph = ' '.join(cleaned_lines)
    
    # Save the cleaned paragraph to the output file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(paragraph)

# Path to your input and output text files
input_file_path = '/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/utils/desired_transcript.txt'
output_file_path = '/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/utils/groundtruth_transcript.txt'

clean_text(input_file_path, output_file_path)
print(f"Cleaned text saved to {output_file_path}")
