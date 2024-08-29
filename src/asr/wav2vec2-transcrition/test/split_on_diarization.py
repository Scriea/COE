from diarization import diarize_audio
import os
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
    max_duration = max_chunk_length * 60 * 1000  # Convert to milliseconds
    min_duration = min_chunk_length * 60 * 1000  # Convert to milliseconds

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
            # Find the optimal split point
            if accumulated_duration >= max_duration or (i + 1 < len(diarization_items) and diarization_items[i + 1][0][0] * 1000 - chunk_end_time > max_duration):
                # Extract the audio chunk
                audio_chunk = audio[chunk_start_time:chunk_end_time]

                # Create the chunk filename
                chunk_filename = f"{base_name}-{chunk_id}.mp3"
                chunk_path = os.path.join(audio_chunk_dir, chunk_filename)

                # Export the audio chunk
                audio_chunk.export(chunk_path, format="mp3")
                print(f"Exported: {chunk_path}")

                # Reset for the next chunk
                chunk_id += 1
                chunk_start_time = None
                accumulated_duration = 0

    # Handle any remaining audio if there is left
    if chunk_start_time is not None and chunk_end_time is not None:
        audio_chunk = audio[chunk_start_time:chunk_end_time]
        chunk_filename = f"{base_name}-{chunk_id}.mp3"
        chunk_path = os.path.join(audio_chunk_dir, chunk_filename)
        audio_chunk.export(chunk_path, format="mp3")
        print(f"Exported: {chunk_path}")

# Example usage
dia_res = diarize_audio("/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/sample/02007350001.mp3", num_speakers=2)
min_chunk_length = 15  # in minutes
max_chunk_length = 20  # in minutes

print(dia_res)
split_audio(dia_res, "/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/sample/02007350001.mp3", "chunk_by_diarization_" + str(min_chunk_length), "02007350001", min_chunk_length=min_chunk_length, max_chunk_length=max_chunk_length)
