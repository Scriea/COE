# process_audios.py
import os
import re
from transcriber import transcribe_audio

def sort_audios(audio_files):
    def extract_number(filename):
        match = re.search(r'_(\d+)\.wav$', filename)
        return int(match.group(1)) if match else -1
    
    return sorted(audio_files, key=extract_number)

def process_audio_folder(audio_folder, output_txt):
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')] 
    sorted_audio_files = sort_audios(audio_files)
    
    with open(output_txt, 'w') as f:
        for audio_file in sorted_audio_files:
            audio_path = os.path.join(audio_folder, audio_file)
            transcript = transcribe_audio(audio_path)
            f.write(transcript + '\n')

# Example usage
audio_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/test_audio'
output_txt = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/test/test_audio.txt'

process_audio_folder(audio_folder, output_txt)
