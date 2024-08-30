from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp

pipeline = FlaxWhisperPipline('parthiv11/indic_whisper_hi_multi_gpu', dtype=jnp.bfloat16)
transcript= pipeline('/raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/wav2vec2/hyp_audio/hindi/02000100001_chunks/02000100001_chunk_1.wav')
print(transcript)