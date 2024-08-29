import os
from pydub import AudioSegment

# Define the paths
audio_directory = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/audio_files_1401_1500/hindi'
output_directory = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/final_1401_1500_audios'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate through each subdirectory in the audio directory
for subdir in os.listdir(audio_directory):
    subdir_path = os.path.join(audio_directory, subdir)
    if os.path.isdir(subdir_path) and subdir.endswith('_chunks'):
        # Create a new AudioSegment instance to store the merged audio for this subdirectory
        merged_audio = AudioSegment.empty()
        
        # Iterate through each chunk file in the subdirectory
        for chunk_file in sorted(os.listdir(subdir_path)):
            if chunk_file.endswith('.wav'):
                chunk_path = os.path.join(subdir_path, chunk_file)
                # Load the chunk file and append it to the merged audio
                chunk_audio = AudioSegment.from_wav(chunk_path)
                merged_audio += chunk_audio
        
        # Define the output path for the merged audio file
        output_file_name = f"{subdir[:-7]}.wav"
        output_path = os.path.join(output_directory, output_file_name)
        
        # Export the merged audio to the output directory
        merged_audio.export(output_path, format='wav')
        
        print(f"Merged audio for {subdir} saved to {output_path}")

print("All subdirectories have been processed.")
