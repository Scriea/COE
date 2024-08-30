import os
import shutil

def process_audio_files(input_dir, batch_size, output_dir):
    # Get all file names in the input directory
    file_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # Extract file IDs and sort them
    file_ids = [(int(f[2:-8]), f) for f in file_names]
    file_ids.sort()

    # Create output directories and copy files
    for i in range(0, len(file_ids), batch_size):
        batch_files = file_ids[i:i + batch_size]
        if batch_files:
            start_id = batch_files[0][0]
            end_id = batch_files[-1][0]
            batch_dir_name = f"hindi_{start_id}_{end_id}"
            batch_dir_path = os.path.join(output_dir, batch_dir_name)
            os.makedirs(batch_dir_path, exist_ok=True)
            
            for _, file_name in batch_files:
                src_path = os.path.join(input_dir, file_name)
                dst_path = os.path.join(batch_dir_path, file_name)
                shutil.copy(src_path, dst_path)

# Example usage
input_directory = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/audio_files_401_500'
batch_size = 20
output_directory = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos'

process_audio_files(input_directory, batch_size, output_directory)
