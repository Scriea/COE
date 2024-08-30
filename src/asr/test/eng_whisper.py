# process_videos.py
import os
import re
from moviepy.editor import VideoFileClip
from transcriber import transcribe_audio

def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path, codec='pcm_s16le')

def sort_videos(video_files):
    def extract_number(filename):
        match = re.search(r'_(\d+)\.mp4$', filename)
        return int(match.group(1)) if match else -1
    
    return sorted(video_files, key=extract_number)

def process_video_folder(video_folder, audio_folder, output_txt):
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    sorted_video_files = sort_videos(video_files)
    
    with open(output_txt, 'w') as f:
        for video_file in sorted_video_files:
            video_path = os.path.join(video_folder, video_file)
            audio_path = os.path.join(audio_folder, f"{os.path.splitext(video_file)[0]}.wav")
            
            extract_audio_from_video(video_path, audio_path)
            transcript = transcribe_audio(audio_path)
            f.write(transcript + '\n')

# Example usage
video_folder = '/raid/ganesh/pdadiga/anindya/Chinese_1_Song'
audio_folder = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/test/chinese_1_audio'
output_txt = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/test/hyp-chinese-1.txt'

process_video_folder(video_folder, audio_folder, output_txt)
