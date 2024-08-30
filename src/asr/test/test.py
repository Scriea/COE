import re
import os
from force_alignment import transcribe_audio
# Assuming `transcribe_audio` function is already defined

def sort_audio_files(audio_dir):
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
    sorted_files = sorted(audio_files, key=lambda x: int(re.search(r'(\d+)', x).group()))
    return sorted_files

def transcribe_and_combine(audio_dir, source_lang, json_dir, model):
    sorted_files = sort_audio_files(audio_dir)
    combined_tokens = []

    for audio_file in sorted_files:
        audio_path = os.path.join(audio_dir, audio_file)
        print("audio_file: ", audio_file)
        word_offsets = transcribe_audio(audio_path, source_lang, json_dir, model)
        if len(word_offsets)>0:
            for token_info in word_offsets:
                combined_tokens.append(token_info['token'])
    
    paragraph = ' '.join(combined_tokens)
    return paragraph

# Example usage:
audio_dir = '/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/chunk_by_diarization_20'
source_lang = 'Hindi' 
json_dir = '/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/transcripts/jsons'
model = 'wav2vec2'

paragraph = transcribe_and_combine(audio_dir, source_lang, json_dir, model)
print(paragraph)

with open("hyp_indic_wav2vec2_diarize_20.txt", 'w', encoding='utf-8') as f:
    f.write(paragraph)

