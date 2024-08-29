import os
import shutil

def create_output_structure(hyp_xml_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each file in hyp_xml_dir
    for file_name in os.listdir(hyp_xml_dir):
        if file_name.endswith('_chunks_merged.xml'):
            # Extract ID from the filename
            
            # Define source and destination paths for each file to be copied
            hyp_source_path = os.path.join(hyp_xml_dir, file_name)
            hyp_dest_path = os.path.join(output_dir, file_name)
 
            
            # Copy and rename hyp file
            shutil.copy(hyp_source_path, hyp_dest_path)
        


# Example usage:
hyp_xml_dir = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/transcripts/whisper_l/xmls_hi2/english"
output_dir = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/transcript_whisper_largev3/hindi_2"

create_output_structure(hyp_xml_dir, output_dir)
