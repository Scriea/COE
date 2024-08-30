import os
from get_silence import extract_silence_level
import shutil

def split_audio_on_silences(filepath, chunks_dir):
    shutil.rmtree(chunks_dir)
    os.remove("mp3splt.log")
    os.system(f"mp3splt -s -p th=-35,min=0.4,rm=50_50,trackjoin=2.5 {filepath} -o @f-@n -d {chunks_dir} > time_o.txt")
    silence_level = extract_silence_level("time_o.txt")
    print(silence_level)

    shutil.rmtree(chunks_dir)
    os.remove("mp3splt.log")
    os.system(f"mp3splt -s -p th={silence_level},min=0.4,rm=50_50,trackjoin=2.5 {filepath} -o @f-@n -d {chunks_dir} > time_o.txt")


split_audio_on_silences("/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/videos/hindi/input_audio.mp3" ,"/home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/utils/temp")

