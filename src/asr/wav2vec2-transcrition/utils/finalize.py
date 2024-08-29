import os
import shutil

# Define the paths to the folders
xml_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/transcripts/wav2vec2/audio_files_1401_1500/hindi'
mp3_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/audio_files_1401_1500/hindi'
output_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/final_1401_1500'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Debug: Print paths
print(f"XML Folder: {xml_folder}")
print(f"MP3 Folder: {mp3_folder}")
print(f"Output Folder: {output_folder}")

# Function to process files
def process_files(mp3_file_path, base_name):
    # Construct the expected XML file name with '_w2v' suffix
    xml_file_name = f"{base_name}_merged.xml"
    xml_path = os.path.join(xml_folder, xml_file_name)
    
    # Check if the corresponding XML file exists
    if os.path.isfile(xml_path):
        # Create a subfolder named with the base name in the output folder
        subfolder_path = os.path.join(output_folder, base_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Define the full paths for the destination files in the subfolder
        mp3_output_path = os.path.join(subfolder_path, os.path.basename(mp3_file_path))
        xml_output_path = os.path.join(subfolder_path, xml_file_name[:-10] + '.xml')
        
        # Copy the files to the subfolder
        shutil.copy(mp3_file_path, mp3_output_path)
        shutil.copy(xml_path, xml_output_path)
        
        # Debug: Print success message
        print(f"Copied {os.path.basename(mp3_file_path)} and {xml_file_name} to {subfolder_path}")
    else:
        # Debug: Print message if XML file is not found
        print(f"XML file {xml_file_name} not found for {os.path.basename(mp3_file_path)}")

# Recursively search for MP3 files in all subdirectories
for root, _, files in os.walk(mp3_folder):
    for mp3_file in files:
        if mp3_file.endswith('.wav'):
            mp3_file_path = os.path.join(root, mp3_file)
            # Extract the base name (without extension)
            base_name = os.path.splitext(mp3_file)[0]
            process_files(mp3_file_path, base_name)

print("Files have been processed.")
