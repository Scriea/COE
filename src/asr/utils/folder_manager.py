import os
import shutil

def create_output_structure(hyp_xml_dir, audio_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each file in hyp_xml_dir
    for file_name in os.listdir(hyp_xml_dir):
        if file_name.endswith('_merged.xml'):
            # Extract ID from the filename
            file_id = file_name.split('_merged')[0]
            
            # Create the subfolder in output_dir
            subfolder_path = os.path.join(output_dir, file_id)
            os.makedirs(subfolder_path, exist_ok=True)
            
            # Define source and destination paths for each file to be copied
            hyp_source_path = os.path.join(hyp_xml_dir, file_name)
            hyp_dest_path = os.path.join(subfolder_path, f"{file_id}_w2v.xml")
            
            audio_source_path = os.path.join(audio_dir, f"{file_id}.mp3")
            audio_dest_path = os.path.join(subfolder_path, f"{file_id}.mp3")
            
            # groundtruth_source_path = os.path.join(groundtruth_dir, f"{file_id}.xml")
            # groundtruth_dest_path = os.path.join(subfolder_path, f"{file_id}.xml")
            
            # Copy and rename hyp file
            shutil.copy(hyp_source_path, hyp_dest_path)
            
            # Copy audio file if it exists
            if os.path.exists(audio_source_path):
                shutil.copy(audio_source_path, audio_dest_path)
            
            # Copy groundtruth file if it exists
            # if os.path.exists(groundtruth_source_path):
            #     shutil.copy(groundtruth_source_path, groundtruth_dest_path)
# Example usage:
hyp_xml_dir = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/transcripts/wav2vec2/audio_files_1601_1700/hindi"
# audio_dir = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/final_1401_1500_audios"
audio_dir="/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/chunked_audio_files_1601_1700"
# groundtruth_dir = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/utils/gt_xml_40"
output_dir = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/output/audio_1601_1700"

create_output_structure(hyp_xml_dir, audio_dir, output_dir)
