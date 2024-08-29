import os
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
def split_audio_on_silence(file_path, output_dir, min_silence_len=500, silence_thresh=-40, min_chunk_len=15*60*1000, max_chunk_len=20*60*1000):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)

    if duration <= max_chunk_len:
        # Copy the file to the output directory if its length is less than or equal to the maximum chunk length
        shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))
        return

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200  # Adjust to include some silence at the beginning and end of each chunk
    )

    output_chunks = []
    current_chunk = AudioSegment.empty()

    for chunk in chunks:
        if len(current_chunk) + len(chunk) < min_chunk_len:
            current_chunk += chunk
        else:
            output_chunks.append(current_chunk)
            current_chunk = chunk

        if len(current_chunk) >= max_chunk_len:
            output_chunks.append(current_chunk)
            current_chunk = AudioSegment.empty()

    if len(current_chunk) > 0:
        output_chunks.append(current_chunk)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    for i, chunk in enumerate(output_chunks):
        chunk.export(os.path.join(output_dir, f"{base_name}_{i + 1}.mp3"), format="mp3")

def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(input_dir), desc="Processing files"):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(input_dir, file_name)
            split_audio_on_silence(file_path, output_dir)

input_dir = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/audio_files_401_1400'
output_dir = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/hindi_401_1400'

process_folder(input_dir, output_dir)
