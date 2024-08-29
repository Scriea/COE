import os
from pydub import AudioSegment

def convert_wav_to_mp3(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            # Define the full path to the input and output files
            wav_path = os.path.join(input_folder, filename)
            mp3_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mp3')
            
            # Load the WAV file
            audio = AudioSegment.from_wav(wav_path)
            
            # Export the file as MP3
            audio.export(mp3_path, format="mp3")
            print(f"Converted {filename} to MP3")

# Example usage
input_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/hindi2'
output_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/hindi2_mp3'
convert_wav_to_mp3(input_folder, output_folder)
