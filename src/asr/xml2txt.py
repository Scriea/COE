import xml.etree.ElementTree as ET

def xml_to_text(xml_path, output_txt_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Check if the root element is 'transcript'
    if root.tag != 'transcript':
        raise ValueError("Invalid XML format. Root element should be 'transcript'.")
    
    # Initialize an empty list to store words
    passages = []
    
    # Iterate through each line in the XML
    for line in root.findall('line'):
        # Concatenate all words in the current line
        passage = ' '.join(word.text for word in line.findall('word') if word.get('is_valid') == "1")
        passages.append(passage)
    
    # Combine all passages into a single text
    full_text = '\n\n'.join(passages)
    
    # Write the output to a .txt file
    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        output_file.write(full_text)

    print(f"Text has been successfully extracted and saved to {output_txt_path}.")

if __name__ == "__main__":
    # Specify the XML input file path and output text file path
    xml_path = '/home/iitb_admin_user/COE/COE/src/asr/wav2vec2-transcrition/audio_files/hindi/temp_merged.xml'  # Replace with your XML file path
    output_txt_path = './output_passage.txt'  # Replace with your desired output path

    # Call the function to convert XML to text
    xml_to_text(xml_path, output_txt_path)
