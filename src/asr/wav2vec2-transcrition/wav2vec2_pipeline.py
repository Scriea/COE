import time
import os
import librosa
import argparse
from audio_extractor import process_video_audio
from force_alignment import transcribe_audio
from xml_creation import create_transcript_xml
from xml_merger import merge_xml_files, update_merged_xml_with_diarization
from diarization import diarize_audio
from xml2txt import xml_to_text

def process_single_file(file_path, audio_dir, lang, json_dir, transcript_xml_dir, model, chunk_length_s, split):
    # Process the video file to extract audio and split into chunks
    diarization_results = process_video_audio(file_path, audio_dir, chunk_length_s, lang, split, diarize_audio)
    print(diarization_results)
    
    # Define the language-specific directory for audio
    lang_audio_dir = os.path.join(audio_dir, lang)
    
    main_audio_filename = os.path.splitext(os.path.basename(file_path))[0]
    chunk_dir = os.path.join(lang_audio_dir, main_audio_filename + '_chunks')

    print(main_audio_filename, chunk_dir)
    
    for root, _, files in os.walk(chunk_dir):
        for file in files:
            if file.endswith('.wav'):
                chunk_audio_path = os.path.join(root, file)
                duration = librosa.get_duration(filename=chunk_audio_path)
                if duration < 0.01:
                    print(f"Skipping {chunk_audio_path} due to short duration: {duration:.4f} seconds")
                    continue
                # Transcribe the chunked audio file
                word_offset = transcribe_audio(chunk_audio_path, lang.capitalize(), json_dir, model)
                
                if len(word_offset) > 0:
                    chunk_number = int(file.split('_chunk_')[-1].split('.')[0])
                    
                    # Directory to save XML files for this main audio file
                    main_audio_xml_dir = os.path.join(transcript_xml_dir, lang, main_audio_filename + '_chunks')
                    os.makedirs(main_audio_xml_dir, exist_ok=True)
                    
                    # Create XML transcript with blank speaker field
                    create_transcript_xml(word_offset, {}, main_audio_filename, chunk_number, main_audio_xml_dir, lang)
    
    # Merge the XML files for the main audio file
    mrg_main_audio_xml_dir = os.path.join(transcript_xml_dir, lang, main_audio_filename + '_chunks')
    output_file = os.path.join(transcript_xml_dir, lang, f"{main_audio_filename}_merged.xml")
    merge_xml_files(mrg_main_audio_xml_dir, output_file, lang, chunk_length_s)
    
    # Update the merged XML file with the diarization results of the whole audio
    diarization_result = diarization_results.get(main_audio_filename, {})
    update_merged_xml_with_diarization(output_file, diarization_result)
    xml_to_text(output_file, os.path.join(transcript_xml_dir, lang, f"{main_audio_filename}_merged.txt"))

def run_pipeline(video_input, audio_dir, lang, json_dir, transcript_xml_dir, model, chunk_length_s=120, split=True):
    if os.path.isdir(video_input):
        video_files = [f for f in os.listdir(video_input) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.ogg'))]
        for video_file in video_files:
            video_path = os.path.join(video_input, video_file)
            try:
                process_single_file(video_path, audio_dir, lang, json_dir, transcript_xml_dir, model, chunk_length_s, split)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
    elif os.path.isfile(video_input):
        process_single_file(video_input, audio_dir, lang, json_dir, transcript_xml_dir, model, chunk_length_s, split)
    else:
        raise ValueError("The provided video_input path is neither a directory nor a file.")
    

def main():
    parser = argparse.ArgumentParser(description='Process video or audio files for transcription and diarization.')
    parser.add_argument('-m','--media_input', type=str, help='Path to the input video or audio file, or directory containing files.')
    parser.add_argument('--audio_dir', type=str, default=os.path.abspath('./audio_files'), help='Directory to save extracted audio files.')
    parser.add_argument('--lang', type=str, default='hindi', help='Language of the audio content (default: hindi).')
    parser.add_argument('--json_dir', type=str, default=os.path.abspath('./transcripts/jsons'), help='Directory to save transcript JSON files.')
    parser.add_argument('--transcript_xml_dir', type=str, default=os.path.abspath('./transcripts/xml'), help='Directory to save transcript XML files.')
    parser.add_argument('--model', type=str, default='wav2vec2', help='Model name for transcription (default: wav2vec2).')
    parser.add_argument('--chunk_length_s', type=int, default=30, help='Chunk length in seconds for processing (default: 120).')
    parser.add_argument('--split', action='store_true', help='Whether to split audio files into chunks (default: True).', default=True)

    args = parser.parse_args()

    # Create directories if they do not exist
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.json_dir, exist_ok=True)
    os.makedirs(args.transcript_xml_dir, exist_ok=True)

    start_time = time.time()
    run_pipeline(args.media_input, args.audio_dir, args.lang, args.json_dir, args.audio_dir, args.model, args.chunk_length_s, args.split)
    end_time = time.time()
    transcription_time = end_time - start_time
    print(f"Time taken to transcribe the directory: {transcription_time:.2f} seconds")

if __name__ == '__main__':
    main()
