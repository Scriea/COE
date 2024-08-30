from diarization import diarize_audio
import os
import shutil
from pydub import AudioSegment

def split_audio(diarization_result, main_audio_path, audio_chunk_dir, base_name, min_chunk_length, max_chunk_length):
    # Load the main audio file
    audio = AudioSegment.from_file(main_audio_path)

    # Create the audio chunk directory if it doesn't exist
    os.makedirs(audio_chunk_dir, exist_ok=True)

    # Convert diarization result to a sorted list of tuples
    diarization_items = sorted(diarization_result.items(), key=lambda x: x[0][0])
    print("sorted: ", diarization_items)
    
    # Initialize variables for chunking
    chunk_start_time = None
    chunk_end_time = None
    accumulated_duration = 0
    chunk_id = 1
    min_duration = min_chunk_length * 60 * 1000  # Convert to milliseconds
    max_duration = max_chunk_length * 60 * 1000  # Convert to milliseconds

    # Iterate through the diarization result and split the audio
    for i in range(len(diarization_items)):
        time_range, _ = diarization_items[i]
        start_time = time_range[0] * 1000  # Convert to milliseconds
        end_time = time_range[1] * 1000  # Convert to milliseconds
        
        # Initialize the chunk start time
        if chunk_start_time is None:
            chunk_start_time = start_time

        # Update the chunk end time
        chunk_end_time = end_time

        # Calculate the accumulated duration
        accumulated_duration = chunk_end_time - chunk_start_time
        
        # Check if accumulated duration exceeds the minimum chunk length
        if accumulated_duration >= min_duration:
            # Find the maximum gap within the range [min_duration, max_duration]
            max_gap = 0
            split_index = i
            for j in range(i + 1, len(diarization_items)):
                next_start_time = diarization_items[j][0][0] * 1000
                gap = next_start_time - chunk_end_time
                if gap > max_gap:
                    max_gap = gap
                    split_index = j
                if accumulated_duration + (next_start_time - chunk_end_time) > max_duration:
                    break
                chunk_end_time = diarization_items[j][0][1] * 1000
                accumulated_duration = chunk_end_time - chunk_start_time
            
            # Split at the point with the maximum gap
            chunk_end_time = diarization_items[split_index][0][0] * 1000
            audio_chunk = audio[chunk_start_time:chunk_end_time]

            # Create the chunk filename with two-digit chunk ID
            chunk_filename = f"{base_name}_{chunk_id:02d}.mp3"
            chunk_path = os.path.join(audio_chunk_dir, chunk_filename)

            # Export the audio chunk
            audio_chunk.export(chunk_path, format="mp3")
            print(f"Exported: {chunk_path}")

            # Reset for the next chunk
            chunk_id += 1
            chunk_start_time = diarization_items[split_index][0][0] * 1000
            accumulated_duration = 0

    # Handle any remaining audio if there is left
    if chunk_start_time is not None and chunk_end_time is not None:
        audio_chunk = audio[chunk_start_time:chunk_end_time]
        chunk_filename = f"{base_name}_{chunk_id:02d}.mp3"
        chunk_path = os.path.join(audio_chunk_dir, chunk_filename)
        audio_chunk.export(chunk_path, format="mp3")
        print(f"Exported: {chunk_path}")

def process_files(input_folder, output_folder, min_chunk_length, max_chunk_length):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(input_folder, file_name)
            audio = AudioSegment.from_file(file_path)
            duration_in_minutes = len(audio) / (1000 * 60)  # Convert milliseconds to minutes
            
            # Check the duration of the file
            if duration_in_minutes > max_chunk_length:
                # Diarize and split the audio file
                dia_res = diarize_audio(file_path, num_speakers=2)
                base_name = os.path.splitext(file_name)[0]
                try:
                    split_audio(dia_res, file_path, output_folder, base_name, min_chunk_length, max_chunk_length)
                except Exception as e:
                    print(f"Error splitting {file_path}: {e}")
                    continue
            else:
                # Copy the file to the output folder
                shutil.copy(file_path, output_folder)
                print(f"Copied: {file_path} to {output_folder}")

# Example usage
# input_folder = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/sample"
input_folder = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/audio_files_1601_1700"
output_folder = "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/chunked_audio_files_1601_1700"
min_chunk_length = 15  # in minutes
max_chunk_length = 20  # in minutes

process_files(input_folder, output_folder, min_chunk_length, max_chunk_length)
