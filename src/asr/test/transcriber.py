# import torchaudio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def transcribe_audio(audio_path):

    result = pipe(audio_path, return_timestamps="word", generate_kwargs={"language": "english"})
    text = result["text"]
    TRANSCRIPT = text.strip()
    return TRANSCRIPT

# print(transcribe_audio("/raid/ganesh/pdadiga/anindya/Russian Song/WhatsApp Audio 2024-07-29 at 10.24.05 AM (1).ogg", "english", "whisper"))