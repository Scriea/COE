from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os

def convert_to_mp3(input_file, output_file):
    # Extract the file extension
    file_extension = os.path.splitext(input_file)[1].lower()

    # Handle video files
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        # Extract audio from video
        video = VideoFileClip(input_file)
        audio = video.audio
        temp_audio_path = "temp_audio.wav"
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')

        # Convert the extracted audio to MP3
        sound = AudioSegment.from_file(temp_audio_path)
        sound.export(output_file, format="mp3")
        os.remove(temp_audio_path)  # Clean up the temporary audio file

    # Handle audio files
    elif file_extension in ['.wav', '.flac', '.ogg', '.m4a']:
        # Convert the audio to MP3 if not already in MP3 format
        sound = AudioSegment.from_file(input_file)
        sound.export(output_file, format="mp3")

    # If the file is already in MP3 format, just copy it
    elif file_extension == '.mp3':
        os.rename(input_file, output_file)

    else:
        raise ValueError("Unsupported file format!")

# Example usage
input_file = "samples\Video.mp4"  # Change this to your input file
output_file = "Video.mp3"  # Change this to your desired output file name
convert_to_mp3(input_file, output_file)
