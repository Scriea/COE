import numpy as np
import os
import sys
import subprocess

class ASR:
    def __init__(self, model):
        self.model = model

    def convert_audio_to_text(self, audio_path):
        pass

if __name__ == "__main__":
    asr = ASR("whisper")
    asr.convert_audio_to_text("audio.wav")