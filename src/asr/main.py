import os
import time
import librosa
from .audio_extractor import process_video_audio
from .force_alignment import transcribe_audio
from .xml_creation import create_transcript_xml
from .xml_merger import merge_xml_files, update_merged_xml_with_diarization
from .diarization import diarize_audio
from .xml2txt import xml_to_text

class ASR:
    def __init__(self, audio_dir='./audio_files', json_dir='./transcripts/jsons', transcript_xml_dir='./transcripts/xml', model='wav2vec2', chunk_length_s=120, split=True):
        """
        Initialize the ASR class with default parameters for directories and model settings.

        Parameters:
        - audio_dir: Directory to save extracted audio files.
        - json_dir: Directory to save transcript JSON files.
        - transcript_xml_dir: Directory to save transcript XML files.
        - model: Model name for transcription (default: 'wav2vec2').
        - chunk_length_s: Chunk length in seconds for processing (default: 120).
        - split: Whether to split audio files into chunks (default: True).
        """
        self.audio_dir = os.path.abspath(audio_dir)
        self.json_dir = os.path.abspath(json_dir)
        self.transcript_xml_dir = os.path.abspath(transcript_xml_dir)
        self.model = model
        self.chunk_length_s = chunk_length_s
        self.split = split

        # Ensure directories exist
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.transcript_xml_dir, exist_ok=True)

    def process_single_file(self, file_path, lang):
        """
        Process a single audio or video file to extract speech and convert it to text.

        Parameters:
        - file_path: Path to the input file (audio or video).
        - lang: Language of the audio content.

        Returns:
        - output_text_path: Path to the file containing the final transcribed text.
        """
        # Process the video file to extract audio and split into chunks
        diarization_results = process_video_audio(file_path, self.audio_dir, self.chunk_length_s, lang, self.split, diarize_audio)
        print(diarization_results)
        
        # Define the language-specific directory for audio
        lang_audio_dir = os.path.join(self.audio_dir, lang)
        
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
                    word_offset = transcribe_audio(chunk_audio_path, lang.capitalize(), self.json_dir, self.model)
                    
                    if len(word_offset) > 0:
                        chunk_number = int(file.split('_chunk_')[-1].split('.')[0])
                        
                        # Directory to save XML files for this main audio file
                        main_audio_xml_dir = os.path.join(self.transcript_xml_dir, lang, main_audio_filename + '_chunks')
                        os.makedirs(main_audio_xml_dir, exist_ok=True)
                        
                        # Create XML transcript with blank speaker field
                        create_transcript_xml(word_offset, {}, main_audio_filename, chunk_number, main_audio_xml_dir, lang)
        
        # Merge the XML files for the main audio file
        mrg_main_audio_xml_dir = os.path.join(self.transcript_xml_dir, lang, main_audio_filename + '_chunks')
        output_file = os.path.join(self.transcript_xml_dir, lang, f"{main_audio_filename}_merged.xml")
        merge_xml_files(mrg_main_audio_xml_dir, output_file, lang, self.chunk_length_s)
        
        # Update the merged XML file with the diarization results of the whole audio
        diarization_result = diarization_results.get(main_audio_filename, {})
        update_merged_xml_with_diarization(output_file, diarization_result)
        output_text_path = os.path.join(self.transcript_xml_dir, lang, f"{main_audio_filename}_merged.txt")
        xml_to_text(output_file, output_text_path)

        return output_text_path

    def speech_to_text(self, media_input, lang):
        """
        Main function to process the input file(s) and return the transcribed text.

        Parameters:
        - media_input: Path to the input media file or directory containing media files.
        - lang: Language of the audio content.

        Returns:
        - A dictionary with file names as keys and their corresponding transcribed text as values.
        """
        if os.path.isfile(media_input):
            try:
                output_text_path = self.process_single_file(media_input, lang)
                with open(output_text_path, 'r') as f:
                    transcription_text = f.read()
                return transcription_text
            except Exception as e:
                print(f"Error processing {media_input}: {e}")
                results[os.path.basename(media_input)] = None
                return None
        else:
            raise ValueError("The provided media_input path is neither a directory nor a file.")
    

if __name__=="__main__":
    asr = ASR()
    media_input = './audios/Tamil_1.mp3'
    lang = 'Tamil'
    results = asr.speech_to_text(media_input, lang)
    print(results)
