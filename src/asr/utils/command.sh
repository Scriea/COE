path C:\Program Files (x86)\mp3splt;%PATH%

mp3splt -s -p th=-37.12,min=0.4,rm=50_50,trackjoin=2.5 /home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/videos/hindi/input_audio.mp3 -o @f-@n -d ./temp

mp3splt -s -P -p th=-37.12,min=0.4,rm=50_50,trackjoin=2.5 -o _@m:@s.@h_@M:@S.@H /home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/videos/hindi/input_audio.mp3 > time_o.txt

wer -i /home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/utils/samples/02004160001.txt /home/kcdh_admin/anindya/Indic-ASR-Transcript-Generator/utils/samples/hyp_indicwhisper.txt 2>&1 | tee hyp_indicwhisper_analysis.txt

CUDA_VISIBLE_DEVICES=6 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/Indicw2v_91_100_log.txt
CUDA_VISIBLE_DEVICES=6 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/w2vhi_111_120_log_v1.txt
CUDA_VISIBLE_DEVICES=6 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/w2vhi_16_19_log.txt

CUDA_VISIBLE_DEVICES=1 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/w2vhi_401_420_log.txt
CUDA_VISIBLE_DEVICES=1 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/w2vhi_421_440_log.txt
CUDA_VISIBLE_DEVICES=2 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/w2vhi_441_460_log.txt
CUDA_VISIBLE_DEVICES=2 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/w2vhi_461_480_log.txt
CUDA_VISIBLE_DEVICES=1 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/w2vhi_test_log_v2.txt

CUDA_VISIBLE_DEVICES=6 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/split_A_log.txt
CUDA_VISIBLE_DEVICES=7 python wav2vec2_pipeline.py 2>&1 | tee ${PWD}/logs/indicwhisper_test_log.txt

python3 infer.py /raid/ganesh/pdadiga/anindya/SPRING_INX_ccc_wav2vec2_Hindi.pt /raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/hyp_audio/02000020001.mp3 2>&1 | tee ccc_wav2vec2_log1.txt

scp -r /raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/split_AB ganesh@10.129.6.170:/home/ganesh/asr_bgpt/anindya/Indic-whisper-wav2vec2-Transcription/transcripts
               
python /raid/ganesh/pdadiga/rishabh/asr/IndicWav2Vec/w2v_inference/scripts/sfi.py --audio-file /raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/videos/hyp_audio/02000020001.mp3 --ft-model /raid/ganesh/pdadiga/anindya/SPRING_INX_wav2vec2_Hindi.pt --w2l-decoder viterbi

python /raid/ganesh/pdadiga/anindya/IndicWav2Vec/lang_wise_manifest_creation.py /raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/audios --dest /raid/ganesh/pdadiga/anindya/Indic-whisper-wav2vec2-Transcription/test --ext mp3 --valid-percent 0.0

CUDA_VISIBLE_DEVICES=4 python infer_copy.py 