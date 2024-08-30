from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import librosa

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("vasista22/ccc-wav2vec2-base-100h")
model = Wav2Vec2ForCTC.from_pretrained("vasista22/ccc-wav2vec2-base-100h")
    
# load dummy dataset and read soundfiles
file_path = '/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/test_audio/02001010001.mp3'

# Load the audio file
audio, sample_rate = librosa.load(file_path)

# tokenize
input_values = processor(audio, return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)
